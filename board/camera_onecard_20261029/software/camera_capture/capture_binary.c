/* SPDX-License-Identifier: MIT */
#include "capture_binary.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DETECT_MAX 512u
#define DETECT_PIXELS (DETECT_MAX * DETECT_MAX)
#define POINTS_MAX (DETECT_MAX * 2u)
#define CANDIDATES_MAX 32u
#define TILE_MAX 16u

typedef struct { double x, y; } point;
typedef struct {
    point q[4];
    double area, rectangularity, contrast[4], score;
} candidate;
typedef struct {
    uint8_t preview[DETECT_PIXELS], smooth[DETECT_PIXELS];
    uint8_t mask[DETECT_PIXELS], temp[DETECT_PIXELS];
    uint32_t queue[DETECT_PIXELS];
    unsigned row_min[DETECT_MAX], row_max[DETECT_MAX];
    point points[POINTS_MAX], hull[POINTS_MAX * 2u], polygon[POINTS_MAX * 2u];
    candidate candidates[CANDIDATES_MAX];
    uint8_t row_gray[4][CAPTURE_BINARY_MAX_DIMENSION];
    uint32_t horizontal[CAPTURE_BINARY_MAX_DIMENSION];
    uint64_t vertical[CAPTURE_BINARY_MAX_DIMENSION];
    uint32_t columns[CAPTURE_BINARY_MAX_DIMENSION];
    unsigned cubic_x[CAPTURE_BINARY_MAX_DIMENSION][4];
    double cubic_weight[CAPTURE_BINARY_MAX_DIMENSION][4];
    uint8_t tile_lut[TILE_MAX * TILE_MAX][256];
} binary_scratch;
_Static_assert(sizeof(binary_scratch) <= CAPTURE_PAPER_WORKSPACE_BYTES,
               "paper detection workspace is too small");
typedef struct {
    const capture_color *color;
    const uint8_t *source;
    unsigned sw, sh;
    size_t source_stride;
    const capture_binary_config *config;
    capture_binary_result *result;
    binary_scratch *scratch;
    void (*service)(void *);
    void *context;
} run_context;

static void tick(run_context *r) { if (r->service) r->service(r->context); }
static unsigned smaller(unsigned a, unsigned b) { return a < b ? a : b; }
static unsigned larger(unsigned a, unsigned b) { return a > b ? a : b; }
static int clamp_index(int i, unsigned n) { return i < 0 ? 0 : (i >= (int)n ? (int)n - 1 : i); }
static uint8_t byte_round(double v)
{
    if (v <= 0.0) return 0;
    if (v >= 255.0) return 255;
    return (uint8_t)(v + 0.5);
}
static uint8_t corrected_gray(const capture_color *color, const uint8_t *p)
{
    uint8_t bgr[3];
    capture_color_apply_bgr(color, p, bgr);
    /* OpenCV's standard fixed-point BGR -> gray coefficients. */
    return (uint8_t)((1868u*bgr[0] + 9617u*bgr[1] + 4899u*bgr[2] + 8192u) >> 14);
}
static double cross(point a, point b, point c)
{
    return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
}
static double distance(point a, point b)
{
    double dx=a.x-b.x, dy=a.y-b.y;
    return sqrt(dx*dx+dy*dy);
}
static double polygon_area(const point *p, unsigned n)
{
    double a=0.0;
    unsigned i;
    for (i=0;i<n;++i) a+=p[i].x*p[(i+1u)%n].y-p[i].y*p[(i+1u)%n].x;
    return fabs(a)*0.5;
}
static int compare_point(const void *va, const void *vb)
{
    const point *a=(const point *)va, *b=(const point *)vb;
    if (a->x < b->x) return -1;
    if (a->x > b->x) return 1;
    return a->y < b->y ? -1 : (a->y > b->y ? 1 : 0);
}
static int compare_byte(const void *a, const void *b)
{ return (int)*(const uint8_t *)a - (int)*(const uint8_t *)b; }

void capture_binary_default_config(capture_binary_config *c)
{
    if (!c) return;
    memset(c,0,sizeof(*c));
    c->target_width=1920u;
    c->clahe_clip_limit=2.0;
    c->clahe_tiles_x=c->clahe_tiles_y=8u;
    c->adaptive_block_size=51u;
    c->adaptive_c=10;
    c->mode=CAPTURE_BINARY_A4;
    c->whiten_outside_paper=1u;
}

int capture_a4_crop(unsigned w,unsigned h,unsigned xywh[4])
{
    unsigned short_edge;
    if(!xywh) return CAPTURE_BINARY_INVALID;
    memset(xywh,0,4u*sizeof(*xywh));
    if(!w || !h || w>CAPTURE_BINARY_MAX_DIMENSION || h>CAPTURE_BINARY_MAX_DIMENSION ||
       (uint64_t)w*h>CAPTURE_BINARY_MAX_PIXELS) return CAPTURE_BINARY_INVALID;
    short_edge=(w*70u+49u)/99u;
    if(!short_edge || short_edge>h) return CAPTURE_BINARY_INVALID;
    xywh[2]=w;xywh[3]=short_edge;
    xywh[0]=0u;xywh[1]=(h-short_edge)/2u;
    return CAPTURE_BINARY_OK;
}

static unsigned histogram_percentile(const unsigned h[256], unsigned count, unsigned percent)
{
    unsigned i,sum=0,goal=(count*percent+99u)/100u;
    for(i=0;i<256u;++i) { sum+=h[i]; if(sum>=goal) return i; }
    return 255u;
}
static unsigned otsu_threshold(const unsigned h[256], unsigned count)
{
    unsigned i,n=0,best=0;
    double all=0.0,sum=0.0,maximum=-1.0;
    for(i=0;i<256u;++i) all+=(double)i*h[i];
    for(i=0;i<255u;++i) {
        double a,b,d,v;
        n+=h[i]; sum+=(double)i*h[i];
        if(!n || n==count) continue;
        a=sum/n; b=(all-sum)/(count-n); d=a-b;
        v=(double)n*(count-n)*d*d;
        if(v>maximum) { maximum=v; best=i; }
    }
    return best;
}

static void make_preview(run_context *r, unsigned pw, unsigned ph)
{
    unsigned x,y;
    binary_scratch *s=r->scratch;
    /* Four subpixel samples are enough for localization; they are NEVER used
     * as the image to crop/resize. Final pixels come from the native frame. */
    for(y=0;y<ph;++y) {
        for(x=0;x<pw;++x) {
            unsigned a,b,sum=0;
            for(b=0;b<2u;++b) for(a=0;a<2u;++a) {
                unsigned sx=(unsigned)(((uint64_t)(4u*x+1u+2u*a)*r->sw)/(4u*pw));
                unsigned sy=(unsigned)(((uint64_t)(4u*y+1u+2u*b)*r->sh)/(4u*ph));
                sx=smaller(sx,r->sw-1u); sy=smaller(sy,r->sh-1u);
                sum+=corrected_gray(r->color,r->source+(size_t)sy*r->source_stride+3u*sx);
            }
            s->preview[y*pw+x]=(uint8_t)((sum+2u)/4u);
        }
        if((y&15u)==15u) tick(r);
    }
    /* 3x3 binomial blur suppresses isolated print/noise without moving edges. */
    for(y=0;y<ph;++y) {
        for(x=0;x<pw;++x) {
            unsigned sum=0; int dy,dx;
            for(dy=-1;dy<=1;++dy) for(dx=-1;dx<=1;++dx) {
                unsigned weight=(dx==0?2u:1u)*(dy==0?2u:1u);
                sum+=weight*s->preview[(unsigned)clamp_index((int)y+dy,ph)*pw+
                                       (unsigned)clamp_index((int)x+dx,pw)];
            }
            s->smooth[y*pw+x]=(uint8_t)((sum+8u)/16u);
        }
        if((y&15u)==15u) tick(r);
    }
}

static void morphology(run_context *r, unsigned w, unsigned h, int dilate)
{
    binary_scratch *s=r->scratch;
    unsigned x,y;
    /* Separable 5x5 closing fills print strokes; four external edge tests
     * below prevent a printed staff rectangle from masquerading as paper. */
    for(y=0;y<h;++y) {
        for(x=0;x<w;++x) {
            unsigned v=dilate?0u:1u; int k;
            for(k=-2;k<=2;++k) {
                unsigned q=s->mask[y*w+(unsigned)clamp_index((int)x+k,w)];
                v=dilate?(v|q):(v&q);
            }
            s->temp[y*w+x]=(uint8_t)v;
        }
        if((y&15u)==15u) tick(r);
    }
    for(y=0;y<h;++y) {
        for(x=0;x<w;++x) {
            unsigned v=dilate?0u:1u; int k;
            for(k=-2;k<=2;++k) {
                unsigned q=s->temp[(unsigned)clamp_index((int)y+k,h)*w+x];
                v=dilate?(v|q):(v&q);
            }
            s->mask[y*w+x]=(uint8_t)v;
        }
        if((y&15u)==15u) tick(r);
    }
}

static unsigned convex_hull(binary_scratch *s, unsigned n)
{
    unsigned i,k=0,t;
    qsort(s->points,n,sizeof(point),compare_point);
    for(i=0;i<n;++i) {
        while(k>=2u && cross(s->hull[k-2u],s->hull[k-1u],s->points[i])<=0.0) --k;
        s->hull[k++]=s->points[i];
    }
    t=k+1u;
    for(i=n-1u;i>0u;--i) {
        while(k>=t && cross(s->hull[k-2u],s->hull[k-1u],s->points[i-1u])<=0.0) --k;
        s->hull[k++]=s->points[i-1u];
    }
    return k>1u?k-1u:0u;
}

static int quad_from_hull(binary_scratch *s, unsigned n, point q[4])
{
    unsigned count=n,i,first=0;
    double perimeter=0.0,tolerance;
    if(n<4u) return 0;
    memcpy(s->polygon,s->hull,n*sizeof(point));
    for(i=0;i<n;++i) perimeter+=distance(s->hull[i],s->hull[(i+1u)%n]);
    while(count>4u) {
        unsigned remove=0;
        double minimum=1e30;
        for(i=0;i<count;++i) {
            point a=s->polygon[(i+count-1u)%count],b=s->polygon[i],c=s->polygon[(i+1u)%count];
            double d=fabs(cross(a,c,b))/fmax(distance(a,c),1.0);
            if(d<minimum) { minimum=d; remove=i; }
        }
        memmove(s->polygon+remove,s->polygon+remove+1u,(count-remove-1u)*sizeof(point));
        --count;
    }
    /* A rounded blob cannot pass merely because we reduced it to four points. */
    tolerance=fmax(2.5,perimeter*0.018);
    for(i=0;i<n;++i) {
        unsigned j; double best=1e30;
        for(j=0;j<4u;++j) {
            point a=s->polygon[j],b=s->polygon[(j+1u)%4u];
            double dx=b.x-a.x,dy=b.y-a.y;
            double t=((s->hull[i].x-a.x)*dx+(s->hull[i].y-a.y)*dy)/(dx*dx+dy*dy);
            point p;
            t=fmax(0.0,fmin(1.0,t)); p.x=a.x+t*dx; p.y=a.y+t*dy;
            best=fmin(best,distance(p,s->hull[i]));
        }
        if(best>tolerance) return 0;
    }
    for(i=1;i<4u;++i)
        if(s->polygon[i].x+s->polygon[i].y<s->polygon[first].x+s->polygon[first].y) first=i;
    for(i=0;i<4u;++i) q[i]=s->polygon[(first+i)%4u];
    return 1;
}

static double minimum_rectangle_area(const point q[4])
{
    unsigned i,j; double best=1e30;
    for(i=0;i<4u;++i) {
        double len=distance(q[i],q[(i+1u)%4u]);
        double ux=(q[(i+1u)%4u].x-q[i].x)/len,uy=(q[(i+1u)%4u].y-q[i].y)/len;
        double minx=1e30,maxx=-1e30,miny=1e30,maxy=-1e30;
        for(j=0;j<4u;++j) {
            double x=q[j].x*ux+q[j].y*uy,y=-q[j].x*uy+q[j].y*ux;
            minx=fmin(minx,x);maxx=fmax(maxx,x);miny=fmin(miny,y);maxy=fmax(maxy,y);
        }
        best=fmin(best,(maxx-minx)*(maxy-miny));
    }
    return best;
}

static int edge_contrasts(run_context *r, unsigned w, unsigned h, candidate *c)
{
    unsigned side,good=0;
    double short_side=1e30;
    for(side=0;side<4u;++side) short_side=fmin(short_side,distance(c->q[side],c->q[(side+1u)%4u]));
    for(side=0;side<4u;++side) {
        uint8_t inside[135],outside[135];
        point a=c->q[side],b=c->q[(side+1u)%4u];
        double len=distance(a,b),nx=-(b.y-a.y)/len,ny=(b.x-a.x)/len;
        unsigned i,k,n=0;
        for(i=0;i<45u;++i) for(k=0;k<3u;++k) {
            double t=0.12+0.76*i/44.0,d=2.0+2.0*k;
            int ix=(int)floor(a.x+(b.x-a.x)*t+d*nx+0.5);
            int iy=(int)floor(a.y+(b.y-a.y)*t+d*ny+0.5);
            int ox=(int)floor(a.x+(b.x-a.x)*t-d*nx+0.5);
            int oy=(int)floor(a.y+(b.y-a.y)*t-d*ny+0.5);
            if(ix<0 || iy<0 || ox<0 || oy<0 || ix>=(int)w || ox>=(int)w || iy>=(int)h || oy>=(int)h)
                return 0;
            inside[n]=r->scratch->preview[(unsigned)iy*w+(unsigned)ix];
            outside[n]=r->scratch->preview[(unsigned)oy*w+(unsigned)ox]; ++n;
        }
        qsort(inside,n,1u,compare_byte);qsort(outside,n,1u,compare_byte);
        c->contrast[side]=(double)inside[n/2u]-outside[n/2u];
        if(c->contrast[side]<6.0) return 0;
        /* A thick printed box has a dark rule just outside its bright interior
         * but white paper farther out. Probe beyond typical print widths too.
         * Samples beyond the camera view are unavailable, not fabricated dark. */
        {
            uint8_t farther[45];unsigned count=0;
            double d=fmax(8.0,short_side*0.12);
            for(i=0;i<45u;++i) {
                double t=0.12+0.76*i/44.0;
                int ox=(int)floor(a.x+(b.x-a.x)*t-d*nx+0.5);
                int oy=(int)floor(a.y+(b.y-a.y)*t-d*ny+0.5);
                if(ox>=0 && oy>=0 && ox<(int)w && oy<(int)h)
                    farther[count++]=r->scratch->preview[(unsigned)oy*w+(unsigned)ox];
            }
            if(count>=15u) {
                qsort(farther,count,1u,compare_byte);
                if((double)inside[n/2u]-farther[count/2u]<6.0) return 0;
            }
        }
        if(c->contrast[side]>=12.0) ++good;
    }
    return good>=3u;
}

static int qualify_candidate(run_context *r, unsigned w, unsigned h, unsigned points,
                             unsigned component_pixels, candidate *c)
{
    binary_scratch *s=r->scratch;
    unsigned i,n=convex_hull(s,points);
    double sides[4],hull_area,aspect,minimum=1e30,mean_contrast=0.0;
    if(n<4u) return 0;
    hull_area=polygon_area(s->hull,n);
    if(hull_area<=0.0 || component_pixels/hull_area<0.78) return 0;
    if(!quad_from_hull(s,n,c->q)) return 0;
    c->area=polygon_area(c->q,4u);
    if(c->area<(double)w*h*0.18 || c->area>(double)w*h*0.96) return 0;
    for(i=0;i<4u;++i) {
        point a=c->q[(i+3u)%4u],b=c->q[i],d=c->q[(i+1u)%4u];
        double cosine=fabs(((a.x-b.x)*(d.x-b.x)+(a.y-b.y)*(d.y-b.y))/
                           fmax(distance(a,b)*distance(d,b),1.0));
        if(cosine>0.65) return 0;
        sides[i]=distance(b,d); minimum=fmin(minimum,sides[i]);
    }
    if(minimum<smaller(w,h)*0.18) return 0;
    aspect=(sides[0]+sides[2])/(sides[1]+sides[3]);
    if(aspect<0.48 || aspect>2.1) return 0;
    c->rectangularity=c->area/fmax(minimum_rectangle_area(c->q),1.0);
    if(c->rectangularity<0.72 || !edge_contrasts(r,w,h,c)) return 0;
    for(i=0;i<4u;++i) mean_contrast+=c->contrast[i]*0.25;
    c->score=c->area*c->rectangularity*(0.6+0.4*fmin(mean_contrast/80.0,1.0));
    return 1;
}

static double intersection_area(const point a[4], const point b[4])
{
    point p[16],q[16]; unsigned n=4u,edge,i;
    memcpy(p,a,4u*sizeof(point));
    for(edge=0;edge<4u && n;++edge) {
        unsigned m=0;
        point start=b[edge],end=b[(edge+1u)%4u];
        for(i=0;i<n;++i) {
            point x=p[i],y=p[(i+1u)%n];
            double dx=cross(start,end,x),dy=cross(start,end,y);
            if(dx>=0.0) q[m++]=x;
            if((dx>=0.0)!=(dy>=0.0)) {
                double t=dx/(dx-dy);
                q[m].x=x.x+t*(y.x-x.x);q[m].y=x.y+t*(y.y-x.y);++m;
            }
        }
        memcpy(p,q,m*sizeof(point));n=m;
    }
    return n>=3u?polygon_area(p,n):0.0;
}

static int detect_paper(run_context *r)
{
    binary_scratch *s=r->scratch;
    capture_binary_result *out=r->result;
    unsigned longest=larger(r->sw,r->sh),pw,ph,x,y,i,level,levels[3],number=0;
    unsigned hist[256]={0},border_hist[256]={0},border_count=0,background,bright;
    candidate found;
    pw=longest>DETECT_MAX?(unsigned)(((uint64_t)r->sw*DETECT_MAX+longest/2u)/longest):r->sw;
    ph=longest>DETECT_MAX?(unsigned)(((uint64_t)r->sh*DETECT_MAX+longest/2u)/longest):r->sh;
    if(pw<32u || ph<32u) return CAPTURE_BINARY_NOT_FOUND;
    out->detection_width=pw;out->detection_height=ph;
    make_preview(r,pw,ph);
    for(y=0;y<ph;++y) for(x=0;x<pw;++x) {
        unsigned v=s->smooth[y*pw+x],border=larger(1u,smaller(pw,ph)/30u);
        ++hist[v];
        if(x<border || y<border || x>=pw-border || y>=ph-border) { ++border_hist[v];++border_count; }
    }
    background=histogram_percentile(border_hist,border_count,50u);
    bright=histogram_percentile(hist,pw*ph,95u);
    levels[0]=otsu_threshold(hist,pw*ph);
    levels[1]=background+larger(6u,bright>background?(bright-background)*6u/100u:0u);
    levels[2]=background+larger(10u,bright>background?(bright-background)*15u/100u:0u);
    for(level=0;level<3u;++level) {
        if(levels[level]>=255u) continue;
        for(i=0;i<pw*ph;++i) s->mask[i]=(uint8_t)(s->smooth[i]>levels[level]);
        morphology(r,pw,ph,1);morphology(r,pw,ph,0);
        for(i=0;i<pw*ph;++i) if(s->mask[i]) {
            unsigned head=0,tail=0,points=0; int clipped=0;
            for(y=0;y<ph;++y) { s->row_min[y]=pw;s->row_max[y]=0; }
            s->mask[i]=0;s->queue[tail++]=i;
            while(head<tail) {
                unsigned p=s->queue[head++],xx=p%pw,yy=p/pw,k;
                unsigned near[4],neighbors=0;
                if(xx<2u || yy<2u || xx>=pw-2u || yy>=ph-2u) clipped=1;
                s->row_min[yy]=smaller(s->row_min[yy],xx);s->row_max[yy]=larger(s->row_max[yy],xx);
                if(xx) near[neighbors++]=p-1u;
                if(xx+1u<pw) near[neighbors++]=p+1u;
                if(yy) near[neighbors++]=p-pw;
                if(yy+1u<ph) near[neighbors++]=p+pw;
                for(k=0;k<neighbors;++k) if(s->mask[near[k]]) {
                    s->mask[near[k]]=0;s->queue[tail++]=near[k];
                }
                if((head&4095u)==0u) tick(r);
            }
            if(clipped || (uint64_t)tail*100u<(uint64_t)pw*ph*15u) continue;
            for(y=0;y<ph;++y) if(s->row_min[y]<pw) {
                s->points[points].x=s->row_min[y];s->points[points++].y=y;
                if(s->row_max[y]!=s->row_min[y]) {
                    s->points[points].x=s->row_max[y];s->points[points++].y=y;
                }
            }
            if(points>=4u && qualify_candidate(r,pw,ph,points,tail,&found)) {
                if(number>=CANDIDATES_MAX) return CAPTURE_BINARY_AMBIGUOUS;
                s->candidates[number++]=found;
            }
            tick(r);
        }
    }
    out->candidates=number;
    if(!number) return CAPTURE_BINARY_NOT_FOUND;
    /* First keep enclosing pages, regardless of interior brightness. */
    for(i=0;i<number;++i) {
        unsigned j;
        for(j=i+1u;j<number;++j) if(s->candidates[j].area>s->candidates[i].area) {
            candidate tmp=s->candidates[i];s->candidates[i]=s->candidates[j];s->candidates[j]=tmp;
        }
    }
    {
        unsigned distinct=0,best=0;
        for(i=0;i<number;++i) {
            unsigned j;int same=0;
            for(j=0;j<distinct;++j)
                if(intersection_area(s->candidates[i].q,s->candidates[j].q)/
                   fmin(s->candidates[i].area,s->candidates[j].area)>=0.88) { same=1;break; }
            if(!same) s->candidates[distinct++]=s->candidates[i];
        }
        for(i=1;i<distinct;++i) if(s->candidates[i].score>s->candidates[best].score) best=i;
        for(i=0;i<distinct;++i)
            if(i!=best && s->candidates[i].area>=s->candidates[best].area*0.65)
                return CAPTURE_BINARY_AMBIGUOUS;
        found=s->candidates[best];
    }
    {
        double minx=1e30,miny=1e30,maxx=-1e30,maxy=-1e30,side=1e30;
        double sx=(double)r->sw/pw,sy=(double)r->sh/ph;
        point native[4]; unsigned pad,x0,y0,x1,y1;
        for(i=0;i<4u;++i) {
            native[i].x=(found.q[i].x+0.5)*sx-0.5;native[i].y=(found.q[i].y+0.5)*sy-0.5;
            out->paper_quad_xy[2u*i]=native[i].x;out->paper_quad_xy[2u*i+1u]=native[i].y;
            minx=fmin(minx,native[i].x);maxx=fmax(maxx,native[i].x);
            miny=fmin(miny,native[i].y);maxy=fmax(maxy,native[i].y);
            out->edge_contrast[i]=found.contrast[i];
        }
        for(i=0;i<4u;++i) side=fmin(side,distance(native[i],native[(i+1u)%4u]));
        pad=(unsigned)ceil(fmax(3.0*fmax(sx,sy),0.01*side));
        x0=(unsigned)fmax(0.0,floor(minx)-pad);y0=(unsigned)fmax(0.0,floor(miny)-pad);
        x1=(unsigned)fmin(r->sw,ceil(maxx)+pad+1u);y1=(unsigned)fmin(r->sh,ceil(maxy)+pad+1u);
        out->crop_xywh[0]=x0;out->crop_xywh[1]=y0;out->crop_xywh[2]=x1-x0;out->crop_xywh[3]=y1-y0;
        out->has_paper_quad=1u;out->crop_padding_px=pad;
        out->whiten_margin_px=r->config->whiten_outside_paper?(unsigned)ceil(4.0*fmax(sx,sy)):0u;
        out->area_fraction=found.area/((double)pw*ph);out->rectangularity=found.rectangularity;
    }
    return CAPTURE_BINARY_OK;
}

static void render_gray_row(run_context *r, unsigned row, uint8_t *gray)
{
    capture_binary_result *o=r->result;
    unsigned x,cx=o->crop_xywh[0],cy=o->crop_xywh[1],cw=o->crop_xywh[2];
    double minimum=-1e30,maximum=1e30;
    const uint8_t *source=r->source+(size_t)(cy+row)*r->source_stride+3u*cx;
    if(o->has_paper_quad && r->config->whiten_outside_paper) {
        unsigned i;
        for(i=0;i<4u;++i) {
            unsigned j=(i+1u)%4u;
            double ax=o->paper_quad_xy[2u*i],ay=o->paper_quad_xy[2u*i+1u];
            double dx=o->paper_quad_xy[2u*j]-ax,dy=o->paper_quad_xy[2u*j+1u]-ay;
            double rhs=dx*((double)(cy+row)-ay)+o->whiten_margin_px*sqrt(dx*dx+dy*dy);
            if(dy>1e-12) maximum=fmin(maximum,ax+rhs/dy);
            else if(dy < -1e-12) minimum=fmax(minimum,ax+rhs/dy);
            else if(rhs<0.0) { minimum=1.0;maximum=0.0; }
        }
    }
    for(x=0;x<cw;++x) {
        if(cx+x<minimum || cx+x>maximum) { gray[x]=255u;++o->whitened_pixels; }
        else gray[x]=corrected_gray(r->color,source+3u*x);
    }
}

static void resize_area(run_context *r, uint8_t *dest)
{
    binary_scratch *s=r->scratch;
    unsigned sw=r->result->crop_xywh[2],sh=r->result->crop_xywh[3];
    unsigned dw=r->result->width,dh=r->result->height,sy,dy=0,x;
    uint64_t denominator=(uint64_t)sw*sh;
    memset(s->vertical,0,dw*sizeof(s->vertical[0]));
    for(sy=0;sy<sh;++sy) {
        uint64_t ystart=(uint64_t)sy*dh,yend=(uint64_t)(sy+1u)*dh;
        render_gray_row(r,sy,s->row_gray[0]);
        for(x=0;x<dw;++x) {
            uint64_t start=(uint64_t)x*sw,end=(uint64_t)(x+1u)*sw;
            unsigned sx=(unsigned)(start/dw);uint64_t position=start,sum=0;
            while(position<end) {
                uint64_t next=(uint64_t)(sx+1u)*dw;
                if(next>end) next=end;
                sum+=(next-position)*s->row_gray[0][sx];position=next;++sx;
            }
            s->horizontal[x]=(uint32_t)sum;
        }
        while(ystart<yend) {
            uint64_t edge=(uint64_t)(dy+1u)*sh,next=edge<yend?edge:yend,weight=next-ystart;
            for(x=0;x<dw;++x) s->vertical[x]+=(uint64_t)s->horizontal[x]*weight;
            ystart=next;
            if(next==edge) {
                for(x=0;x<dw;++x) {
                    dest[(size_t)dy*dw+x]=(uint8_t)((s->vertical[x]+denominator/2u)/denominator);
                    s->vertical[x]=0;
                }
                ++dy;
            }
        }
        if((sy&15u)==15u) tick(r);
    }
    tick(r);
}

static double cubic(double x)
{
    const double a=-0.75;
    x=fabs(x);
    if(x<=1.0) return (a+2.0)*x*x*x-(a+3.0)*x*x+1.0;
    if(x<2.0) return a*x*x*x-5.0*a*x*x+8.0*a*x-4.0*a;
    return 0.0;
}
static void resize_cubic(run_context *r, uint8_t *dest)
{
    binary_scratch *s=r->scratch;
    unsigned sw=r->result->crop_xywh[2],sh=r->result->crop_xywh[3];
    unsigned dw=r->result->width,dh=r->result->height,x,y,k;
    int row_tag[4]={-1,-1,-1,-1};
    for(x=0;x<dw;++x) {
        double position=(x+0.5)*sw/dw-0.5;int base=(int)floor(position);
        for(k=0;k<4u;++k) {
            s->cubic_x[x][k]=(unsigned)clamp_index(base+(int)k-1,sw);
            s->cubic_weight[x][k]=cubic(position-(base+(int)k-1));
        }
    }
    for(y=0;y<dh;++y) {
        double position=(y+0.5)*sh/dh-0.5,weight[4];
        int base=(int)floor(position);unsigned rows[4];
        for(k=0;k<4u;++k) {
            unsigned sy=(unsigned)clamp_index(base+(int)k-1,sh),slot=sy&3u;
            rows[k]=slot;weight[k]=cubic(position-(base+(int)k-1));
            if(row_tag[slot]!=(int)sy) { render_gray_row(r,sy,s->row_gray[slot]);row_tag[slot]=(int)sy; }
        }
        for(x=0;x<dw;++x) {
            double v=0.0;unsigned j;
            for(k=0;k<4u;++k) for(j=0;j<4u;++j)
                v+=weight[k]*s->cubic_weight[x][j]*s->row_gray[rows[k]][s->cubic_x[x][j]];
            dest[(size_t)y*dw+x]=byte_round(v);
        }
        if((y&15u)==15u) tick(r);
    }
    tick(r);
}

static unsigned reflect101(unsigned x, unsigned size)
{
    if(size<=1u) return 0u;
    x%=2u*size-2u;
    return x<size?x:2u*size-2u-x;
}
static void apply_clahe(run_context *r, const uint8_t *input, uint8_t *output)
{
    binary_scratch *s=r->scratch;
    unsigned w=r->result->width,h=r->result->height;
    unsigned nx=r->config->clahe_tiles_x,ny=r->config->clahe_tiles_y;
    unsigned tw=(w+nx-1u)/nx,th=(h+ny-1u)/ny,tx,ty,x,y;
    uint64_t area=(uint64_t)tw*th;
    unsigned clip=(unsigned)fmax(1.0,floor(r->config->clahe_clip_limit*(double)area/256.0));
    for(ty=0;ty<ny;++ty) for(tx=0;tx<nx;++tx) {
        unsigned hist[256]={0},i,excess=0,sum=0;
        for(y=0;y<th;++y) {
            const uint8_t *row=input+(size_t)reflect101(ty*th+y,h)*w;
            for(x=0;x<tw;++x) ++hist[row[reflect101(tx*tw+x,w)]];
            if((y&31u)==31u) tick(r);
        }
        for(i=0;i<256u;++i) if(hist[i]>clip) { excess+=hist[i]-clip;hist[i]=clip; }
        for(i=0;i<256u;++i) hist[i]+=excess/256u;
        excess%=256u;
        if(excess) {
            unsigned step=larger(1u,256u/excess);
            for(i=0;i<256u && excess;i+=step,--excess) ++hist[i];
        }
        for(i=0;i<256u;++i) {
            sum+=hist[i];s->tile_lut[ty*nx+tx][i]=(uint8_t)(((uint64_t)sum*255u+area/2u)/area);
        }
        tick(r);
    }
    for(y=0;y<h;++y) {
        double fy=(y+0.5)/th-0.5;int y0=(int)floor(fy),y1=y0+1;
        double wy=fy-y0;unsigned iy0=(unsigned)clamp_index(y0,ny),iy1=(unsigned)clamp_index(y1,ny);
        for(x=0;x<w;++x) {
            double fx=(x+0.5)/tw-0.5;int x0=(int)floor(fx),x1=x0+1;
            double wx=fx-x0;unsigned ix0=(unsigned)clamp_index(x0,nx),ix1=(unsigned)clamp_index(x1,nx);
            unsigned v=input[(size_t)y*w+x];
            double top=s->tile_lut[iy0*nx+ix0][v]*(1.0-wx)+s->tile_lut[iy0*nx+ix1][v]*wx;
            double bottom=s->tile_lut[iy1*nx+ix0][v]*(1.0-wx)+s->tile_lut[iy1*nx+ix1][v]*wx;
            output[(size_t)y*w+x]=byte_round(top*(1.0-wy)+bottom*wy);
        }
        if((y&15u)==15u) tick(r);
    }
    tick(r);
}

static void adaptive_mean(run_context *r, const uint8_t *input, uint8_t *output)
{
    unsigned w=r->result->width,h=r->result->height,block=r->config->adaptive_block_size;
    unsigned radius=block/2u,x,y,k,area=block*block;
    uint32_t *columns=r->scratch->columns;
    memset(columns,0,w*sizeof(*columns));
    for(k=0;k<block;++k) {
        unsigned sy=(unsigned)clamp_index((int)k-(int)radius,h);
        for(x=0;x<w;++x) columns[x]+=input[(size_t)sy*w+x];
    }
    for(y=0;y<h;++y) {
        uint64_t sum=0;
        for(k=0;k<block;++k) sum+=columns[(unsigned)clamp_index((int)k-(int)radius,w)];
        for(x=0;x<w;++x) {
            unsigned mean=(unsigned)((sum+area/2u)/area);
            output[(size_t)y*w+x]=((int)input[(size_t)y*w+x]>(int)mean-r->config->adaptive_c)?255u:0u;
            if(x+1u<w) {
                sum-=columns[(unsigned)clamp_index((int)x-(int)radius,w)];
                sum+=columns[(unsigned)clamp_index((int)x+(int)radius+1,w)];
            }
        }
        if(y+1u<h) {
            const uint8_t *remove=input+(size_t)clamp_index((int)y-(int)radius,h)*w;
            const uint8_t *add=input+(size_t)clamp_index((int)y+(int)radius+1,h)*w;
            for(x=0;x<w;++x) columns[x]=columns[x]-remove[x]+add[x];
        }
        if((y&15u)==15u) tick(r);
    }
    tick(r);
}

int capture_paper_detect(const capture_color *color, const uint8_t *source,
                         unsigned w, unsigned h, size_t stride,
                         void *workspace, size_t workspace_bytes,
                         capture_binary_result *result,
                         void (*service)(void *), void *context)
{
    capture_binary_config config;
    run_context r;
    size_t source_bytes;
    uintptr_t src=(uintptr_t)source, work=(uintptr_t)workspace;
    uint8_t check[3];
    if(!result) return CAPTURE_BINARY_INVALID;
    memset(result,0,sizeof(*result));
    if(!source || !color || !w || !h || w>CAPTURE_BINARY_MAX_DIMENSION || h>CAPTURE_BINARY_MAX_DIMENSION ||
       (uint64_t)w*h>CAPTURE_BINARY_MAX_PIXELS || stride<(size_t)w*3u ||
       (size_t)(h-1u)>(SIZE_MAX-(size_t)w*3u)/stride) return CAPTURE_BINARY_INVALID;
    source_bytes=(size_t)(h-1u)*stride+(size_t)w*3u;
    if(src>UINTPTR_MAX-source_bytes) return CAPTURE_BINARY_INVALID;
    if(!workspace || (work&7u) || workspace_bytes<CAPTURE_PAPER_WORKSPACE_BYTES || work>UINTPTR_MAX-workspace_bytes)
        return CAPTURE_BINARY_WORKSPACE;
    if(src<work+workspace_bytes && work<src+source_bytes) return CAPTURE_BINARY_INVALID;
    if(capture_color_row_rgb(color,source,check,1u)!=CAPTURE_COLOR_OK) return CAPTURE_BINARY_INVALID;
    capture_binary_default_config(&config);
    memset(&r,0,sizeof(r)); r.color=color; r.source=source; r.sw=w; r.sh=h;
    r.source_stride=stride; r.config=&config; r.result=result;
    r.scratch=(binary_scratch *)workspace; r.service=service; r.context=context;
    result->source_width=w; result->source_height=h;
    return detect_paper(&r);
}

int capture_binary_run(const capture_color *color, const uint8_t *source,
                       unsigned w, unsigned h, size_t stride,
                       const capture_binary_config *config,
                       void *workspace, size_t workspace_bytes,
                       capture_binary_result *result,
                       void (*service)(void *), void *context)
{
    capture_binary_config defaults;
    run_context r;
    size_t source_bytes,pixels,used;
    uintptr_t src=(uintptr_t)source,work=(uintptr_t)workspace;
    unsigned out_h;int status;
    uint8_t check[3],*gray,*clahe,*binary;
    if(!result) return CAPTURE_BINARY_INVALID;
    memset(result,0,sizeof(*result));
    if(!config) { capture_binary_default_config(&defaults);config=&defaults; }
    if(!source || !color || !w || !h || w>CAPTURE_BINARY_MAX_DIMENSION || h>CAPTURE_BINARY_MAX_DIMENSION ||
       (uint64_t)w*h>CAPTURE_BINARY_MAX_PIXELS || stride<(size_t)w*3u ||
       (size_t)(h-1u)>(SIZE_MAX-(size_t)w*3u)/stride ||
       !config->target_width || config->target_width>CAPTURE_BINARY_MAX_DIMENSION ||
       !isfinite(config->clahe_clip_limit) || config->clahe_clip_limit<=0.0 || config->clahe_clip_limit>256.0 ||
       !config->clahe_tiles_x || !config->clahe_tiles_y || config->clahe_tiles_x>TILE_MAX || config->clahe_tiles_y>TILE_MAX ||
       config->adaptive_block_size<3u || config->adaptive_block_size>255u || !(config->adaptive_block_size&1u) ||
       config->adaptive_c < -255 || config->adaptive_c>255 || config->mode>CAPTURE_BINARY_A4 ||
       config->whiten_outside_paper>1u)
        return CAPTURE_BINARY_INVALID;
    source_bytes=(size_t)(h-1u)*stride+(size_t)w*3u;
    if(src>UINTPTR_MAX-source_bytes) return CAPTURE_BINARY_INVALID;
    if(!workspace || (work&7u) || workspace_bytes<CAPTURE_BINARY_WORKSPACE_BYTES || work>UINTPTR_MAX-workspace_bytes)
        return CAPTURE_BINARY_WORKSPACE;
    if(src<work+workspace_bytes && work<src+source_bytes) return CAPTURE_BINARY_INVALID;
    /* Use public validated color API rather than depend on its private magic. */
    if(capture_color_row_rgb(color,source,check,1u)!=CAPTURE_COLOR_OK) return CAPTURE_BINARY_INVALID;
    memset(&r,0,sizeof(r));r.color=color;r.source=source;r.sw=w;r.sh=h;r.source_stride=stride;
    r.config=config;r.result=result;r.scratch=(binary_scratch *)workspace;r.service=service;r.context=context;
    result->source_width=w;result->source_height=h;
    if(config->mode==CAPTURE_BINARY_AUTO) {
        status=detect_paper(&r);if(status!=CAPTURE_BINARY_OK) return status;
    } else if(config->mode==CAPTURE_BINARY_A4) {
        status=capture_a4_crop(w,h,result->crop_xywh);if(status!=CAPTURE_BINARY_OK) return status;
    } else if(config->mode==CAPTURE_BINARY_FULL) {
        result->crop_xywh[2]=w;result->crop_xywh[3]=h;
    } else {
        unsigned x=config->roi_xywh[0],y=config->roi_xywh[1],rw=config->roi_xywh[2],rh=config->roi_xywh[3];
        if(!rw || !rh || x>=w || y>=h || rw>w-x || rh>h-y) return CAPTURE_BINARY_INVALID;
        memcpy(result->crop_xywh,config->roi_xywh,sizeof(result->crop_xywh));
    }
    result->width=config->target_width;
    out_h=(unsigned)(((uint64_t)result->crop_xywh[3]*result->width+result->crop_xywh[2]/2u)/result->crop_xywh[2]);
    if(!out_h) out_h=1u;
    if(out_h>CAPTURE_BINARY_MAX_DIMENSION || (uint64_t)out_h*result->width>CAPTURE_BINARY_MAX_PIXELS)
        return CAPTURE_BINARY_OUTPUT_BOUNDS;
    result->height=out_h;result->stride=result->width;
    result->scale_xy[0]=(double)result->width/result->crop_xywh[2];
    result->scale_xy[1]=(double)result->height/result->crop_xywh[3];
    pixels=(size_t)result->width*result->height;
    used=(sizeof(binary_scratch)+7u)&~(size_t)7u;
    if(used>workspace_bytes || pixels>(workspace_bytes-used)/3u) return CAPTURE_BINARY_WORKSPACE;
    gray=(uint8_t *)workspace+used;clahe=gray+pixels;binary=clahe+pixels;
    if(result->width>result->crop_xywh[2]) {
        result->interpolation=CAPTURE_BINARY_CUBIC;resize_cubic(&r,gray);
    } else {
        result->interpolation=result->width==result->crop_xywh[2]?CAPTURE_BINARY_NONE:CAPTURE_BINARY_AREA;
        resize_area(&r,gray);
    }
    apply_clahe(&r,gray,clahe);adaptive_mean(&r,clahe,binary);
    result->gray_resized=gray;result->clahe=clahe;result->binary=binary;
    return CAPTURE_BINARY_OK;
}

const char *capture_binary_result_string(int status)
{
    switch(status) {
    case CAPTURE_BINARY_OK:return "ok";
    case CAPTURE_BINARY_INVALID:return "invalid image, color context, or configuration";
    case CAPTURE_BINARY_WORKSPACE:return "workspace too small, unaligned, or invalid";
    case CAPTURE_BINARY_NOT_FOUND:return "no reliable complete paper boundary; show all four edges on a contrasting background";
    case CAPTURE_BINARY_AMBIGUOUS:return "multiple plausible pages; show one page only";
    case CAPTURE_BINARY_OUTPUT_BOUNDS:return "paper aspect ratio exceeds output dimension/pixel limits";
    default:return "unknown binary preprocessing error";
    }
}
