// Exact C++ kernels for Homr 0.7.0 staff detection (see homr_fast.py and README.md).
//
// hg_screen    superset pre-screen for AngledBoundingBox._can_shapes_possibly_touch:
//              candidate i passes when |c_i - q| <= major_i + major_q (+ margin); the caller
//              then runs Homr's own exact test on the candidates only.
// hg_connect   homr.staff_detection.connect_staff_lines (grouping only): same visiting order,
//              same arithmetic as RotatedBoundingBox.is_overlapping_extrapolated.
// hg_resample  homr.staff_detection.resample_staff_segment for one anchor and one x range: the
//              get_at / get_center_extrapolated per line and Homr's checks, in the same order.
// hg_crossing  homr.staff_detection.are_lines_crossing with cv::pointPolygonTest's integer branch (hg_ppt);
//              the centre-distance screen is decided only outside a margin, else the caller runs Homr's code.
// Build: g++ -O3 -ffp-contract=off -fPIC -shared -std=c++17 homrgeo.cpp -o libhomrgeo.so
//        cl /O2 /fp:precise /LD /EHsc /std:c++17 homrgeo.cpp /Fe:homrgeo.dll
// No fused multiply-add and no fast-math: every product and sum is rounded like CPython's.
#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

// candidates of one query, ascending index; returns their number (<= n)
API int hg_screen(int n, const double* cx, const double* cy, const double* mj, double qx, double qy, double qm,
                  double margin, int* out) {
    int k = 0;
    for (int i = 0; i < n; i++) {
        double dx = cx[i] - qx, dy = cy[i] - qy;
        double s = mj[i] + qm;
        if (std::sqrt(dx * dx + dy * dy) <= s + margin * (1.0 + s)) out[k++] = i;
    }
    return k;
}

namespace {
// per fragment: F[7*i + ...]
enum { BLX = 0, BRX = 1, CX = 2, CY = 3, SX = 4, TAN = 5, NF = 7 };

// RotatedBoundingBox.is_overlapping_extrapolated(self=a, other=b)
bool extrapolated(const double* F, int a, int b, double max_gap, double tol) {
    const double* A = F + (size_t)NF * a;
    const double* B = F + (size_t)NF * b;
    const double *L, *R;
    if (A[CX] > B[CX]) { L = B; R = A; } else { L = A; R = B; }
    double center_x = (L[CX] + R[CX]) * 0.5;
    // python: size[0] // 2 on a float (sizes are >= 0, so this is floor(size / 2))
    if (center_x - L[CX] - std::floor(L[SX] / 2.0) > max_gap || R[CX] - center_x - std::floor(R[SX] / 2.0) > max_gap)
        return false;
    double left_y = (center_x - L[CX]) * L[TAN] + L[CY];
    double right_y = (center_x - R[CX]) * R[TAN] + R[CY];
    return std::fabs(left_y - right_y) <= tol;
}
}  // namespace

// groups written top to bottom: out_len[g] members each, concatenated in out_idx (input indices,
// in the order python appends them).  Returns the number of groups, or -1 if a buffer is too small.
API int hg_connect(int n, const double* F, double max_gap, double tol, double short_len, int max_groups,
                   int* out_len, int cap, int* out_idx) {
    // sorted(key=bottom_left x, reverse=True) is stable, and the loop pops from the end:
    // ascending x, equal x in descending input order
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        double xa = F[(size_t)NF * a + BLX], xb = F[(size_t)NF * b + BLX];
        if (xa < xb) return true;
        if (xb < xa) return false;
        return a > b;
    });
    std::vector<std::vector<int>> groups;
    std::vector<int> active;
    double last_cleanup = 0.0;
    for (int i : order) {
        double x = F[(size_t)NF * i + BLX];
        if (x - last_cleanup > max_gap) {
            std::vector<int> keep;
            for (int g : active)
                if (x - F[(size_t)NF * groups[g].back() + BRX] < max_gap) keep.push_back(g);
            active.swap(keep);
            last_cleanup = x;
        }
        if (F[(size_t)NF * i + SX] < short_len) continue;
        bool connected = false;
        for (int g : active)
            if (extrapolated(F, groups[g].back(), i, max_gap, tol)) {
                groups[g].push_back(i);
                connected = true;
            }
        if (!connected) {
            groups.push_back({i});
            active.push_back((int)groups.size() - 1);
        }
    }
    std::vector<int> top(groups.size());
    for (size_t g = 0; g < groups.size(); g++) top[g] = (int)g;
    std::stable_sort(top.begin(), top.end(), [&](int a, int b) {
        return F[(size_t)NF * groups[a][0] + CY] < F[(size_t)NF * groups[b][0] + CY];
    });
    if ((int)groups.size() > max_groups) return -1;
    int k = 0;
    for (size_t r = 0; r < top.size(); r++) {
        const std::vector<int>& g = groups[top[r]];
        if (k + (int)g.size() > cap) return -1;
        out_len[r] = (int)g.size();
        for (int v : g) out_idx[k++] = v;
    }
    return (int)groups.size();
}

// resample_staff_segment's loop.  L lines, line l holds nf[l] fragments (staff_fragments order), each as
// G[6*f ...] = lo, hi (StaffLineSegment.get_at: x >= lo and x <= hi, lo/hi computed in Python by Homr's
// expressions), bx, by, tan (get_center_extrapolated: (x - bx) * tan + by), angle.  prev_y: the anchor's
// dummy point (L values).  For every x in order: ok[q] = 1 and ys[q*L ...], angles[q] if Homr yields a point
// there, and units[q] = StaffPoint's np.mean(np.diff(y)).  Returns 0, or -1 if L is out of range.
API int hg_resample(int L, const int* nf, const double* G, double unit, const double* prev_y, int nx,
                    const double* xs, int* ok, double* ys, double* angles, double* units) {
    const int MAXL = 64;
    if (L < 1 || L > MAXL) return -1;
    int start[MAXL + 1];
    start[0] = 0;
    for (int l = 0; l < L; l++) start[l + 1] = start[l] + nf[l];
    double prev[MAXL];
    for (int l = 0; l < L; l++) prev[l] = prev_y[l];
    const double half = 0.5 * unit;
    for (int q = 0; q < nx; q++) {
        const double x = xs[q];
        ok[q] = 0;
        int frag[MAXL];
        double ac[MAXL];
        bool has[MAXL];
        bool any = false;
        for (int l = 0; l < L; l++) {
            frag[l] = -1;
            has[l] = false;
            for (int f = start[l]; f < start[l + 1]; f++) {
                const double* g = G + (size_t)6 * f;
                if (x >= g[0] && x <= g[1]) { frag[l] = f; break; }
            }
            if (frag[l] >= 0) {
                const double* g = G + (size_t)6 * frag[l];
                ac[l] = (x - g[2]) * g[4] + g[3];
                has[l] = true;
                any = true;
            }
        }
        if (!any) continue;                      // all centres None
        // deltas of the compacted centres; Homr applies flag i to axis_center[i] and [i + 1]
        double cv[MAXL];
        int m = 0;
        for (int l = 0; l < L; l++)
            if (has[l]) cv[m++] = ac[l];
        bool bad[MAXL];
        for (int i = 0; i + 1 < m; i++) bad[i] = (cv[i + 1] - cv[i]) < half;
        for (int i = 0; i + 1 < m; i++)
            if (bad[i]) { has[i] = false; has[i + 1] = false; }
        for (int i = 0; i < L; i++)
            if (has[i] && std::fabs(ac[i] - prev[i]) > half) has[i] = false;
        // forward then backward; prev_center is not reset between the passes nor moved by a filled entry
        int pc = -1;
        for (int k = 0; k < 2 * L; k++) {
            int i = k < L ? k : 2 * L - 1 - k;
            if (has[i]) pc = i;
            else if (pc >= 0) { ac[i] = ac[pc] + unit * (double)(i - pc); has[i] = true; }
        }
        bool complete = true;
        for (int l = 0; l < L; l++) complete = complete && has[l];
        if (!complete) continue;
        // np.mean of the found fragments' angles: numpy adds fewer than 8 values in order, from 0.0
        double s = 0.0;
        int c = 0;
        for (int l = 0; l < L; l++)
            if (frag[l] >= 0) { s += G[(size_t)6 * frag[l] + 5]; c++; }
        ok[q] = 1;
        angles[q] = s / (double)c;
        for (int l = 0; l < L; l++) { ys[(size_t)q * L + l] = ac[l]; prev[l] = ac[l]; }
        double du = 0.0;                         // np.mean(np.diff(y)), fewer than 8 values: in order from 0.0
        for (int l = 0; l + 1 < L; l++) du += ac[l + 1] - ac[l];
        units[q] = du / (double)(L - 1);
    }
    return 0;
}

// cv::pointPolygonTest(contour, pt, measureDist=false) for an integer contour and an integer point (OpenCV's
// integer ray-crossing branch): 1 inside, 0 on an edge or vertex, -1 outside.
static int ppt_int(const int* poly, int n, int px, int py) {
    if (n <= 0) return -1;
    int counter = 0;
    int vx = poly[2 * (n - 1)], vy = poly[2 * (n - 1) + 1];
    for (int i = 0; i < n; i++) {
        int v0x = vx, v0y = vy;
        vx = poly[2 * i];
        vy = poly[2 * i + 1];
        if ((v0y <= py && vy <= py) || (v0y > py && vy > py) || (v0x < px && vx < px)) {
            if (py == vy && (px == vx || (py == v0y && ((v0x <= px && px <= vx) || (vx <= px && px <= v0x)))))
                return 0;
            continue;
        }
        long long dist = (long long)(py - v0y) * (long long)(vx - v0x) - (long long)(px - v0x) * (long long)(vy - v0y);
        if (dist == 0) return 0;
        if (vy < v0y) dist = -dist;
        counter += dist > 0;
    }
    return counter % 2 == 0 ? -1 : 1;
}

API int hg_ppt(const int* poly, int n, int px, int py) { return ppt_int(poly, n, px, py); }

// homr.staff_detection.are_lines_crossing: lines l = 0..nl-1 own fragments [ls[l], ls[l+1]); fragment f has centre
// (cx, cy), major axis mj and polygon pts [ps[f], ps[f+1]) of xy (int).  A pair is tested like
// RotatedBoundingBox.is_overlapping: _can_shapes_possibly_touch (distance > mj1 + mj2 -> no), then
// do_polygons_overlap (a vertex of one inside or on the other).  The distance test is decided only outside a
// relative margin around equality (python computes it with ** 0.5); a pair inside the margin returns -1 and
// the caller runs Homr's function.  Returns 1 crossing, 0 not.
API int hg_crossing(int nl, const int* ls, const double* cx, const double* cy, const double* mj, const int* ps,
                    const int* xy, double margin) {
    for (int a = 0; a < nl; a++)
        for (int b = a + 1; b < nl; b++)
            for (int i = ls[a]; i < ls[a + 1]; i++)
                for (int j = ls[b]; j < ls[b + 1]; j++) {
                    double dx = cx[i] - cx[j], dy = cy[i] - cy[j];
                    double d = std::sqrt(dx * dx + dy * dy), s = mj[i] + mj[j];
                    if (d > s + margin * (1.0 + s)) continue;          // certainly no touch
                    if (!(d < s - margin * (1.0 + s))) return -1;       // too close to call here
                    const int* p = xy + 2 * (size_t)ps[i];
                    const int* q = xy + 2 * (size_t)ps[j];
                    int np = ps[i + 1] - ps[i], nq = ps[j + 1] - ps[j];
                    for (int k = 0; k < np; k++)
                        if (ppt_int(q, nq, p[2 * k], p[2 * k + 1]) >= 0) return 1;
                    for (int k = 0; k < nq; k++)
                        if (ppt_int(p, np, q[2 * k], q[2 * k + 1]) >= 0) return 1;
                }
    return 0;
}
