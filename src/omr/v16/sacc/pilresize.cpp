// Pillow's Image.resize(size) for 8-bit RGB (default filter BICUBIC, no box, no reducing_gap) in C++, so that the
// two passes can run on row ranges in several threads (see pil_resize.py; Pillow 12.3.0, libImaging/Resample.c).
//
// pr_coeffs   precompute_coeffs + normalize_coeffs_8bpc: bounds (xmin, n) and 22-bit fixed-point taps per output
//             pixel.  Returns ksize, or -1 when a tap lies within `margin` of its rounding boundary.  Pillow's build
//             may contract a*b+c into fused multiply-adds: with in0 = 0 the centre (0 + (xx + 0.5) * scale) is one
//             rounded product either way, so the bounds are bit-identical; the bicubic polynomial may differ in its
//             last bits, which can only matter for a tap near a boundary -> the caller uses Pillow there.
// pr_horiz    horizontal pass over source rows [y0, y1) -> temp rows [y0 - first, y1 - first)
// pr_vert     vertical pass for output rows [yy0, yy1)
// The pixel passes are integer arithmetic only, as Pillow's.
// Build: g++ -O3 -ffp-contract=off -fPIC -shared -std=c++17 pilresize.cpp -o libpilresize.so
//        cl /O2 /fp:precise /LD /EHsc /std:c++17 pilresize.cpp /Fe:pilresize.dll
#include <cmath>
#include <cstdint>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

static const int PRECISION_BITS = 32 - 8 - 2;

static inline double bicubic_filter(double x) {
    const double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

static inline bool near_int(double v, double margin) {
    double f = v - std::floor(v);
    return f < margin || f > 1.0 - margin;
}

static inline uint8_t clip8(int in) {
    if (in >= (1 << PRECISION_BITS << 8)) return 255;
    if (in <= 0) return 0;
    return (uint8_t)(in >> PRECISION_BITS);
}

// bounds[2 * outSize], kk[outSize * kmax]
API int pr_coeffs(int inSize, float in0, float in1, int outSize, int kmax, int* bounds, int* kk, double margin) {
    if (in0 != 0.0f) return -1;                 // see above: bounds are exact only for a box starting at 0
    double filterscale, scale;
    filterscale = scale = (double)(in1 - in0) / outSize;
    if (filterscale < 1.0) filterscale = 1.0;
    double support = 2.0 * filterscale;
    int ksize = (int)std::ceil(support) * 2 + 1;
    if (ksize > kmax || ksize > 64) return -1;
    double k[64];
    for (int xx = 0; xx < outSize; xx++) {
        double center = in0 + (xx + 0.5) * scale;
        double ww = 0.0;
        double ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        xmax -= xmin;
        for (int x = 0; x < xmax; x++) {
            double w = bicubic_filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; x++) {
            if (ww != 0.0) k[x] /= ww;
        }
        for (int x = 0; x < ksize; x++) {
            int v = 0;
            if (x < xmax) {
                double p = k[x] * (1 << PRECISION_BITS);
                double q = k[x] < 0 ? -0.5 + p : 0.5 + p;
                if (near_int(q, margin)) return -1;
                v = (int)q;
            }
            kk[xx * ksize + x] = v;
        }
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    return ksize;
}

// src: rows of w_in pixels x 3 bytes (stride sstride); tmp row r = source row r + first
API void pr_horiz(const uint8_t* src, int sstride, int y0, int y1, int first, uint8_t* tmp, int tstride, int w_out,
                  const int* bounds, const int* kk, int ksize) {
    for (int y = y0; y < y1; y++) {
        const uint8_t* row = src + (size_t)y * sstride;
        uint8_t* out = tmp + (size_t)(y - first) * tstride;
        for (int xx = 0; xx < w_out; xx++) {
            int xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1];
            const int* k = kk + xx * ksize;
            int ss0 = 1 << (PRECISION_BITS - 1), ss1 = ss0, ss2 = ss0;
            const uint8_t* p = row + xmin * 3;
            for (int x = 0; x < xmax; x++) {
                ss0 += p[x * 3 + 0] * k[x];
                ss1 += p[x * 3 + 1] * k[x];
                ss2 += p[x * 3 + 2] * k[x];
            }
            out[xx * 3 + 0] = clip8(ss0);
            out[xx * 3 + 1] = clip8(ss1);
            out[xx * 3 + 2] = clip8(ss2);
        }
    }
}

// tmp: rows of w pixels x 3 (bounds already shifted by `first`); output rows [yy0, yy1)
API void pr_vert(const uint8_t* tmp, int tstride, int w, uint8_t* dst, int dstride, int yy0, int yy1,
                 const int* bounds, const int* kk, int ksize) {
    int n = w * 3;
    for (int yy = yy0; yy < yy1; yy++) {
        int ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1];
        const int* k = kk + yy * ksize;
        uint8_t* out = dst + (size_t)yy * dstride;
        for (int i = 0; i < n; i++) {
            int ss = 1 << (PRECISION_BITS - 1);
            for (int y = 0; y < ymax; y++) ss += tmp[(size_t)(ymin + y) * tstride + i] * k[y];
            out[i] = clip8(ss);
        }
    }
}
