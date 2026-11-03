// Exact C++ kernels for Codex route_trials (relation_fast.py, redecode_fast.py).
//
// hr_stroke_groups: the grouping loop of strokes() over the Hough segments, already sorted by
// -|dy| (stable).  Python recomputes a group's statistics after every append as
// (np.mean of the segment x centres, min y1, max y2); the centres are half-integers, so every
// partial sum is exact in double and np.mean == exact_sum / n with one rounding, which is what
// is computed here.  Same comparisons, same order, first matching group.
// Build: g++ -O3 -ffp-contract=off -fPIC -shared -std=c++17 relgeo.cpp -o librelgeo.so
//        cl /O2 /fp:precise /LD /EHsc /std:c++17 relgeo.cpp /Fe:relgeo.dll
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

// seg: n x 4 int32 (x1, y1, x2, y2) in the sorted order; out_group[i] = group of segment i or -1
// (filtered).  Returns the number of groups, or -1 if a sum could leave the exact range.
API int hr_stroke_groups(int n, const int32_t* seg, double space, double minimum, int* out_group) {
    std::vector<double> sum, mean;           // exact sum of centres, sum / count
    std::vector<long long> cnt, lo, hi;
    const double len_min = minimum * space;
    const double tx = .35 * space, ty = .4 * space;
    for (int i = 0; i < n; i++) {
        long long x1 = seg[4 * i], y1 = seg[4 * i + 1], x2 = seg[4 * i + 2], y2 = seg[4 * i + 3];
        long long dy = std::llabs(y2 - y1), dx = std::llabs(x2 - x1);
        // python: abs(y2-y1) < minimum*space or abs(x2-x1) > max(2, .08*abs(y2-y1))
        if ((double)dy < len_min || (double)dx > std::max(2.0, .08 * (double)dy)) {
            out_group[i] = -1;
            continue;
        }
        if (y1 > y2) { std::swap(x1, x2); std::swap(y1, y2); }
        double x = (double)(x1 + x2) / 2.0;
        int selected = -1;
        for (size_t g = 0; g < mean.size(); g++) {
            if (std::fabs(x - mean[g]) < tx && (double)y1 < (double)hi[g] + ty && (double)y2 > (double)lo[g] - ty) {
                selected = (int)g;
                break;
            }
        }
        if (selected < 0) {
            selected = (int)mean.size();
            sum.push_back(0.0); mean.push_back(0.0); cnt.push_back(0); lo.push_back(y1); hi.push_back(y2);
        }
        sum[selected] += x;
        cnt[selected] += 1;
        if (std::fabs(sum[selected]) > 4.0e15) return -1;      // stay far inside 2^53 / 2
        mean[selected] = sum[selected] / (double)cnt[selected];
        lo[selected] = std::min(lo[selected], y1);
        hi[selected] = std::max(hi[selected], y2);
        out_group[i] = selected;
    }
    return (int)mean.size();
}

// ---------------------------------------------------------------- redecode kernels (redecode_fast.py)
namespace {
// python round(float): round half to even
inline long long pyround(double v) { return (long long)std::nearbyint(v); }

// python slice a[start:stop] on an axis of length n -> [lo, hi)
inline void pyslice(long long start, long long stop, long long n, long long& lo, long long& hi) {
    if (start < 0) { start += n; if (start < 0) start = 0; }
    if (start > n) start = n;
    if (stop < 0) { stop += n; if (stop < 0) stop = 0; }
    if (stop > n) stop = n;
    lo = start; hi = stop < start ? start : stop;
}

// repair_unison_voices._stem_end; returns 1 and (col, last), or 0 for None
int stem_end(const uint8_t* ink, long long H, long long W, double s, double x, double y, int up, long long& out_col,
             long long& out_last) {
    long long nominal = pyround(x + (up ? 0.6 * s : -0.6 * s));
    bool found = false;
    long long bl = 0, bc = 0, bla = 0;                 // max over (length, col, last)
    for (long long col = nominal - 2; col < nominal + 3; col++) {
        long long start = pyround(y + (up ? -0.4 * s : 0.4 * s));
        long long sign = up ? -1 : 1;
        long long last = start, blanks = 0;
        long long nd = pyround(s * 7);
        for (long long distance = 0; distance < nd; distance++) {
            long long row = start + sign * distance;
            if (!(0 <= row && row < H) || !(0 <= col && col < W)) break;
            if (ink[row * W + col]) { last = row; blanks = 0; }
            else { blanks += 1; if (blanks >= 2) break; }
        }
        long long length = std::llabs(last - start);
        if ((double)length >= 1.4 * s) {
            if (!found || length > bl || (length == bl && (col > bc || (col == bc && last > bla)))) {
                bl = length; bc = col; bla = last; found = true;
            }
        }
    }
    if (!found) return 0;
    out_col = bc; out_last = bla;
    return 1;
}
}  // namespace

API int hr_stem_end(const uint8_t* ink, long long H, long long W, double s, double x, double y, int up, long long* out) {
    return stem_end(ink, H, W, s, x, y, up, out[0], out[1]);
}

// repair_tuplets.continuous_beams(a, b, ink, up, count, maximum_gap); sa = staff space of a (thresholds).
// 1 / 0 = True / False; -1 = python would raise (caller falls back to the original).
API int hr_continuous_beams(const uint8_t* ink, long long H, long long W, double sa, double ax, double ay, double sb,
                            double bx, double by, int up, long long count, double maximum_gap) {
    double s = sa;
    long long xa = 0, ya = 0, xb = 0, yb = 0;
    int fa = stem_end(ink, H, W, sa, ax, ay, up, xa, ya);
    int fb = stem_end(ink, H, W, sb, bx, by, up, xb, yb);
    if (!(count == 1 || count == 2) || !(4 <= maximum_gap && maximum_gap <= 8) || !fa || !fb) return 0;
    if (!(.7 * s <= (double)(xb - xa) && (double)(xb - xa) <= maximum_gap * s) || (double)std::llabs(ya - yb) > s) return 0;
    static const double F8[7] = {.125, .25, .375, .5, .625, .75, .875}, F4[3] = {.25, .5, .75};
    const double* fr = maximum_gap > 4 ? F8 : F4;
    int nf = maximum_gap > 4 ? 7 : 3;
    for (int k = 0; k < nf; k++) {
        double fraction = fr[k];
        long long x = pyround(xa + (xb - xa) * fraction);
        double y = ya + (yb - ya) * fraction;
        double radius = 1.35;
        long long top = std::max(0LL, pyround(y - radius * s));
        long long bottom = std::min(H, pyround(y + radius * s) + 1);
        long long xc = x;
        if (xc < 0) xc += W;                           // python int index (negative wraps)
        if (xc < 0 || xc >= W) return -1;              // python would raise IndexError
        long long lo_r, hi_r;
        pyslice(top, bottom, H, lo_r, hi_r);
        std::vector<std::pair<long long, long long>> spans;
        long long n = hi_r - lo_r;
        long long r = 0;
        while (r < n) {
            if (ink[(lo_r + r) * W + xc] > 0) {
                long long a = r;
                while (r < n && ink[(lo_r + r) * W + xc] > 0) r++;
                long long lo = a, hi = r;
                double mid = top + (lo + hi - 1) / 2.0;
                if (.2 * s <= (double)(hi - lo) && (double)(hi - lo) <= .8 * s &&
                    (up ? (mid >= y - .4 * s) : (mid <= y + .4 * s)))
                    spans.push_back({top + lo, top + hi});
            } else r++;
        }
        if ((long long)spans.size() != count) return 0;
        double outer = 0;
        for (size_t q = 0; q < spans.size(); q++) {
            double c = (spans[q].first + spans[q].second - 1) / 2.0;
            if (q == 0 || (up ? c < outer : c > outer)) outer = c;
        }
        if (std::fabs(outer - y) > .55 * s) return 0;
        if (count == 2) {
            double gap = (double)(spans[1].first - spans[0].second);
            if (!(.1 * s <= gap && gap <= .6 * s)) return 0;
        }
    }
    return 1;
}

// relation_grace.head_width(image, xy, space, quantile=1.0): max over rows (np.quantile(.., 1) == max)
API double hr_head_width(const uint8_t* img, long long H, long long W, double x, double y, double space) {
    long long x0 = std::max(0LL, pyround(x - 1.1 * space));
    long long x1 = std::min(W, pyround(x + 1.1 * space) + 1);
    long long centre = pyround(x) - x0;
    long long lo_c, hi_c;
    pyslice(x0, x1, W, lo_c, hi_c);
    long long y0 = std::max(0LL, pyround(y - .32 * space)), y1 = std::min(H, pyround(y + .32 * space) + 1);
    bool any = false;
    double best = 0.0;
    for (long long yy = y0; yy < y1; yy++) {
        long long n = hi_c - lo_c;
        if (n <= 0) continue;
        const uint8_t* row = img + yy * W + lo_c;
        long long best_span = -1;
        long long r = 0;
        while (r < n) {
            if (row[r] < 155) {
                long long a = r;
                while (r < n && row[r] < 155) r++;
                long long b = r;
                if ((double)a - .15 * space <= (double)centre && (double)centre < (double)b + .15 * space &&
                    .55 * space < (double)(b - a) && (double)(b - a) < 1.65 * space)
                    best_span = std::max(best_span, b - a);
            } else r++;
        }
        if (best_span >= 0) {
            double v = best_span / space;
            if (!any || v > best) best = v;
            any = true;
        }
    }
    return any ? best : 0.0;
}

// relation_voices.beam_chains: union roots per group.  rel: m x 3 (y_range[0], y_range[1], stem_x) per
// beam relation, rel_off[i]..rel_off[i+1] those of group i.  img: grayscale, ink = img < 155.
API void hr_beam_chains(int n, const double* gx, const int* beams, const int* grace, const int* rel_off,
                        const double* rel, const uint8_t* img, long long H, long long W, double space, int* out_root) {
    std::vector<int> parent(n);
    for (int i = 0; i < n; i++) parent[i] = i;
    auto root = [&](int i) { while (parent[i] != i) i = parent[i]; return i; };
    for (int i = 0; i < n; i++) {
        if (!beams[i]) continue;
        for (int j = i + 1; j < n; j++) {
            if (gx[j] - gx[i] > 8 * space) break;
            if (!beams[j] || (grace[i] != 0) != (grace[j] != 0)) continue;
            for (int p = rel_off[i]; p < rel_off[i + 1]; p++)
                for (int q = rel_off[j]; q < rel_off[j + 1]; q++) {
                    const double* ea = rel + 3 * (size_t)p;
                    const double* eb = rel + 3 * (size_t)q;
                    double ya = (ea[0] + ea[1]) / 2.0, yb = (eb[0] + eb[1]) / 2.0;
                    if (std::fabs(ya - yb) > 1.5 * space) continue;
                    long long num = std::max(4LL, pyround(std::fabs(eb[2] - ea[2])));
                    double div = (double)(num - 1);
                    double dx = eb[2] - ea[2], dy = yb - ya, stx = dx / div, sty = dy / div;
                    long long hits = 0;
                    for (long long k = 0; k < num; k++) {
                        double xk, yk;
                        if (k == num - 1) { xk = eb[2]; yk = yb; }
                        else {
                            xk = stx == 0 ? ((double)k / div) * dx + ea[2] : (double)k * stx + ea[2];
                            yk = sty == 0 ? ((double)k / div) * dy + ya : (double)k * sty + ya;
                        }
                        long long yy = pyround(yk), xx = pyround(xk);
                        long long r0, r1, c0, c1;
                        pyslice(std::max(0LL, yy - 2), std::min(H, yy + 3), H, r0, r1);
                        pyslice(std::max(0LL, xx - 1), std::min(W, xx + 2), W, c0, c1);
                        bool hit = false;
                        for (long long r = r0; r < r1 && !hit; r++)
                            for (long long c = c0; c < c1; c++)
                                if (img[r * W + c] < 155) { hit = true; break; }
                        hits += hit;
                    }
                    if ((double)hits / (double)num >= .88) parent[root(j)] = root(i);
                }
        }
    }
    for (int i = 0; i < n; i++) out_root[i] = root(i);
}

// relation_grace.head_width(image, xy, space, quantile) for any quantile: numpy's default (linear)
// quantile of the row widths, reproduced: virtual index n*q + (1 - q) - 1, floor / next clipped to
// the last element, gamma = index - floor, lerp a + (b - a) * gamma, or b - (b - a) * (1 - gamma) where
// gamma >= 0.5.  Returns 0.0 when no row qualifies (python: `... if rows else 0.`).
API double hr_head_width_q(const uint8_t* img, long long H, long long W, double x, double y, double space,
                           double q) {
    long long x0 = std::max(0LL, pyround(x - 1.1 * space));
    long long x1 = std::min(W, pyround(x + 1.1 * space) + 1);
    long long centre = pyround(x) - x0;
    long long lo_c, hi_c;
    pyslice(x0, x1, W, lo_c, hi_c);
    long long y0 = std::max(0LL, pyround(y - .32 * space)), y1 = std::min(H, pyround(y + .32 * space) + 1);
    std::vector<double> rows;
    for (long long yy = y0; yy < y1; yy++) {
        long long n = hi_c - lo_c;
        if (n <= 0) continue;
        const uint8_t* row = img + yy * W + lo_c;
        long long best_span = -1;
        long long r = 0;
        while (r < n) {
            if (row[r] < 155) {
                long long a = r;
                while (r < n && row[r] < 155) r++;
                long long b = r;
                if ((double)a - .15 * space <= (double)centre && (double)centre < (double)b + .15 * space &&
                    .55 * space < (double)(b - a) && (double)(b - a) < 1.65 * space)
                    best_span = std::max(best_span, b - a);
            } else r++;
        }
        if (best_span >= 0) rows.push_back(best_span / space);
    }
    if (rows.empty()) return 0.0;
    std::sort(rows.begin(), rows.end());
    long long n = (long long)rows.size();
    double vi = ((double)n * q + (1.0 + q * -1.0)) - 1.0;
    long long prev = (long long)std::floor(vi), next = prev + 1;
    if (vi >= (double)(n - 1)) { prev = n - 1; next = n - 1; }
    if (vi < 0) { prev = 0; next = 0; }
    double gamma = vi - (double)prev;
    double a = rows[prev], b = rows[next], d = b - a;
    return gamma >= 0.5 ? b - d * (1.0 - gamma) : a + d * gamma;
}
