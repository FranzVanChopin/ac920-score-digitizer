// C++ port of search.solve (stage A beam search) with the LearnedPicker MLP and the pair
// successor scorer, for the production configuration (prod.PROD + pair_w/close_cost).
//
// Follows src/search.py, src/pickers.py, src/losses.py, src/pair_succ.py and src/visual.py
// line by line: same branches, comparisons, tie rules (stable sorts) and evaluation order.
// Unavoidable differences from the numpy reference: BLAS summation order in the MLP and the
// pair dot product, libm exp/log vs numpy's, and numpy argsort's unstable order for tied x
// with more than 16 candidates (here: stable).  Sums that numpy does with pairwise
// summation (np.sum of the successor exps) use the same pairwise algorithm.
// Plain C API for ctypes.  Build: MSVC /O2 /fp:precise, g++ -O3 -ffp-contract=off.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

#ifdef DEC_PROF
#include <chrono>
// profiling build only (-DDEC_PROF): seconds and calls per step, read with dec_prof()
static double g_prof_t[16];
static long g_prof_n[16];
struct ProfScope {
    int id;
    std::chrono::steady_clock::time_point t0;
    explicit ProfScope(int i) : id(i), t0(std::chrono::steady_clock::now()) {}
    ~ProfScope() {
        g_prof_t[id] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        g_prof_n[id]++;
    }
};
#define PROF_CAT2(a, b) a##b
#define PROF_CAT(a, b) PROF_CAT2(a, b)
#define PROF(i) ProfScope PROF_CAT(prof_scope_, __LINE__)(i)
#else
#define PROF(i)
#endif

namespace {

const double EPS = 1e-6;
const double WHOLE = 1920.0;
const int CAND_DIM = 31, STATE_DIM = 6, N_D = 7, N_DOTS = 3, SCALE = 12;

struct Model {
    int H = 0, H2 = 0, nw = 0;
    std::vector<double> W1, W1s, b1, W2, b2, w2, W3, b3, W4, b4, W5, b5, we, beta, wt;
    double be = 0, bt = 0;
    std::vector<int> wa, wn;                 // warp vocab (actual, normal)
    double pmu[10], psd[10], pw[10], pb = 0;  // pair successor model
    int idx11 = 0, idx32 = -1;
};

struct Cand {
    int cid, staff, is_rest, hint_d, hint_dots, hwa, hwn, hwi, midi, stem;
    double x;
};

struct Ev { int cid; double x, onset, dur; int d, o, w; };
typedef std::vector<Ev> Voice;
typedef std::vector<Voice> Voices;

struct Params {
    int beam, k_attr, warp_top, d_top_n, o_top_n, max_expansions, iso_max;
    double pretense_tol, lam_struct, progress_w, voice_w, dev_w, tau_w, canon_pen, iso_w,
        iso_tol, lam_iso, hint_w, pair_w, close_cost, stem_w, staff_w, width, lam_hint;
};

// ---------------------------------------------------------------- visual.py
double duration_of(int d, int o, int a, int n) {
    double written = WHOLE * std::pow(2.0, -d) * (2.0 - std::pow(2.0, -o));
    return written * ((double)n / (double)a);
}

// numpy pairwise summation (loops_utils.h.src), contiguous double
double pairwise_sum(const double* a, long n) {
    if (n < 8) {
        double res = -0.0;
        for (long i = 0; i < n; i++) res += a[i];
        return res;
    } else if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j];
        long i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    } else {
        long n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
    }
}

struct Table {
    int nw;
    std::vector<double> dur;                 // [d][o][w]
    std::vector<int> canon;                  // canonical key index, -1 none
    std::vector<char> canon_ok;
    std::vector<double> dur_vals;            // sorted (stable on insertion order)
    std::vector<int> dur_keys;
    double min_one;
    int key(int d, int o, int w) const { return (d * 3 + o) * nw + w; }
};

int canonical(double dur, const Model& M, const Table& T) {
    // preference: no time-warp first, then simpler division, then fewer dots
    int best = -1, bp = 0, bd = 0, bo = 0;
    for (int w = 0; w < M.nw; w++) {
        int plain = (M.wa[w] == 1 && M.wn[w] == 1) ? 0 : 1;
        for (int d = 0; d < 7; d++)
            for (int o = 0; o < 3; o++)
                if (std::fabs(T.dur[T.key(d, o, w)] - dur) <= 1e-6) {
                    if (best < 0 || plain < bp || (plain == bp && (d < bd || (d == bd && o < bo)))) {
                        best = T.key(d, o, w); bp = plain; bd = d; bo = o;
                    }
                }
    }
    return best;
}

Table make_table(const Model& M) {
    Table T;
    T.nw = M.nw;
    T.dur.resize(7 * 3 * M.nw);
    for (int d = 0; d < 7; d++)
        for (int o = 0; o < 3; o++)
            for (int w = 0; w < M.nw; w++) T.dur[T.key(d, o, w)] = duration_of(d, o, M.wa[w], M.wn[w]);
    T.min_one = 1e300;
    for (int w = 0; w < M.nw; w++) T.min_one = std::min(T.min_one, T.dur[T.key(6, 0, w)]);
    T.canon.resize(T.dur.size());
    T.canon_ok.resize(T.dur.size());
    // insertion order of python's table dict: d, o, w
    std::vector<std::pair<double, int>> pairs;
    for (int d = 0; d < 7; d++)
        for (int o = 0; o < 3; o++)
            for (int w = 0; w < M.nw; w++) {
                int k = T.key(d, o, w);
                T.canon[k] = canonical(T.dur[k], M, T);
                T.canon_ok[k] = T.canon[k] == k;
                pairs.push_back({T.dur[k], k});
            }
    std::stable_sort(pairs.begin(), pairs.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first < b.first; });
    for (auto& p : pairs) { T.dur_vals.push_back(p.first); T.dur_keys.push_back(p.second); }
    return T;
}

double hint_duration(const Cand& c) { return duration_of(c.hint_d, c.hint_dots, c.hwa, c.hwn); }

// ---------------------------------------------------------------- losses.py
double tick_twist(const std::vector<std::pair<double, double>>& p) {
    const double E = 1e-9;
    if (p.size() < 3) return 0.0;
    double x0 = p[0].first, t0 = p[0].second, x1 = p.back().first, t1 = p.back().second;
    if (x1 - x0 <= E) return 2.0;
    double s = (t1 - t0) / (x1 - x0);
    if (s <= E) return 2.0;
    double worst = 0.0;
    for (size_t i = 0; i + 1 < p.size(); i++) {
        double dx = p[i + 1].first - p[i].first, dt = p[i + 1].second - p[i].second;
        if (dt < -E) return 2.0;
        if (dx <= E) { worst = std::max(worst, 1.5); continue; }
        double local = dt / dx;
        double dev = std::fabs(local / s - 1.0);
        double pen = dev / (1.0 + dev);
        if (pen > worst) worst = pen;
    }
    return worst;
}

double tau_voice(const Voice& v, double length, double width) {
    PROF(0);
    if (v.size() < 3) return 0.0;
    std::vector<std::pair<double, double>> pts;
    for (auto& e : v) pts.push_back({e.x / width, e.onset / length});
    return tick_twist(pts);
}

double max_tau(const Voices& vs, double length, double width) {
    if (vs.empty()) return 0.0;
    double m = -1e300;
    for (auto& v : vs) m = std::max(m, tau_voice(v, length, width));
    return m;
}

double iso_penalty(const Voices& vs, double length, double width, double tol, bool use_max) {
    std::vector<const Ev*> evs;
    for (auto& v : vs) for (auto& e : v) evs.push_back(&e);
    size_t n = evs.size();
    if (n < 2) return 0.0;
    double tot = 0.0, worst = 0.0;
    long cnt = 0;
    for (size_t i = 0; i < n; i++) {
        double xi = evs[i]->x, ti = evs[i]->onset;
        for (size_t j = i + 1; j < n; j++) {
            double dx = std::fabs(xi - evs[j]->x) / width;
            if (dx >= tol) continue;
            double w = 1.0 - dx / tol;
            double d = w * (std::fabs(ti - evs[j]->onset) / length);
            tot += d;
            cnt++;
            if (d > worst) worst = d;
        }
    }
    if (!cnt) return 0.0;
    return use_max ? worst : tot / cnt;
}

// ---------------------------------------------------------------- search.py helpers
struct Ctx {
    const Model* M;
    const Params* P;
    Table T;
    std::vector<Cand> cands;                 // index == cid
    int n;
    double length;
    int n_beats;
    std::vector<int> hmap_d, hmap_o, hmap_w; // hint_map
    std::vector<int> allx;                   // cids sorted by x (stable)
    std::vector<int> allpos;                 // position of cid in allx
    int k_est;
};

double final_loss(const Ctx& C, const Voices& vs, double logp, int decisions) {
    PROF(1);
    const Params& P = *C.P;
    double tau = max_tau(vs, C.length, P.width);
    double kv = std::fabs((double)vs.size() - (double)C.k_est);
    double dev = -logp / std::max(1.0, (double)decisions);
    double iso = iso_penalty(vs, C.length, P.width, P.iso_tol, P.iso_max != 0);
    double hd = 0.0;
    if (P.hint_w != 0.0) {
        int n = 0, bad = 0;
        for (auto& v : vs) for (auto& e : v) {
            n++;
            if (e.d != C.hmap_d[e.cid] || e.o != C.hmap_o[e.cid] || e.w != C.hmap_w[e.cid]) bad++;
        }
        hd = n ? (double)bad / n : 0.0;
    }
    double sm = 0.0;
    if (P.stem_w != 0.0) {
        std::vector<std::vector<int>> staves;
        std::unordered_map<int, int> count;
        for (auto& v : vs) {
            std::vector<int> ss;
            for (auto& e : v) {
                int s = C.cands[e.cid].staff;
                if (std::find(ss.begin(), ss.end(), s) == ss.end()) ss.push_back(s);
            }
            for (int s : ss) count[s]++;
            staves.push_back(ss);
        }
        int bad = 0;
        for (size_t i = 0; i < vs.size(); i++) {
            bool shared = false;
            for (int s : staves[i]) if (count[s] > 1) shared = true;
            if (!shared) continue;
            std::vector<int> st;
            for (auto& e : vs[i]) {
                int s = C.cands[e.cid].stem;
                if (s != 0 && std::find(st.begin(), st.end(), s) == st.end()) st.push_back(s);
            }
            if (st.size() > 1) bad++;
        }
        sm = (double)bad;
    }
    double xs = 0.0;
    if (P.staff_w != 0.0) {
        int bad = 0;
        for (auto& v : vs) {
            std::vector<int> ss;
            for (auto& e : v) {
                int s = C.cands[e.cid].staff;
                if (std::find(ss.begin(), ss.end(), s) == ss.end()) ss.push_back(s);
            }
            if (ss.size() > 1) bad++;
        }
        xs = (double)bad;
    }
    double xm = 0.0;
    return P.tau_w * tau + P.voice_w * kv + P.dev_w * dev + P.iso_w * iso + P.hint_w * hd + 0.0 * xm +
           P.stem_w * sm + P.staff_w * xs;
}

// ---------------------------------------------------------------- pickers.py
struct PickOut {
    std::vector<double> succ;                // n+1 log-probs
    std::vector<double> D, O, W;             // n x 7, n x 3, n x nw log-probs
};

void cand_features(const Ctx& C, const std::vector<int>& rem, double acc, double last_x, std::vector<double>& F) {
    PROF(2);
    int n = (int)rem.size();
    double length = C.length;
    double rem_ticks = length - acc;
    F.assign((size_t)n * CAND_DIM, 0.0);
    // ranks by x: stable (numpy argsort is stable only up to 16 elements)
    std::vector<int> xs_sorted(n);
    for (int j = 0; j < n; j++) xs_sorted[j] = j;
    std::stable_sort(xs_sorted.begin(), xs_sorted.end(), [&](int a, int b) { return C.cands[rem[a]].x < C.cands[rem[b]].x; });
    std::vector<int> rank_of(n), order(n);
    for (int r = 0; r < n; r++) rank_of[xs_sorted[r]] = r;
    for (int j = 0; j < n; j++) order[j] = rank_of[j];
    double sum_hint = 0.0;
    for (int j = 0; j < n; j++) sum_hint += hint_duration(C.cands[rem[j]]);
    double misfit = (sum_hint - rem_ticks) / std::max(1.0, length);
    std::vector<double> hint_of(n);
    std::unordered_map<int, double> staff_sum;
    std::vector<int> staff_order;
    for (int j = 0; j < n; j++) {
        const Cand& c = C.cands[rem[j]];
        hint_of[j] = hint_duration(c);
        if (!staff_sum.count(c.staff)) staff_sum[c.staff] = 0.0;
        staff_sum[c.staff] += hint_of[j];
    }
    for (int i = 0; i < n; i++) {
        const Cand& c = C.cands[rem[i]];
        int hdd = std::min(std::max(c.hint_d, 0), 6);
        int hdots = std::min(std::max(c.hint_dots, 0), 2);
        std::vector<std::pair<int, double>> alts;
        for (int delta : {-1, 1}) {
            int dd = hdd + delta;
            if (dd >= 0 && dd <= 6) alts.push_back({delta, duration_of(dd, hdots, c.hwa, c.hwn)});
        }
        double hd = hint_duration(c);
        double dx = c.x - last_x;
        double frac = hd / std::max(rem_ticks, 1.0);
        int r = rank_of[i];
        const Cand* pv = r > 0 ? &C.cands[rem[xs_sorted[r - 1]]] : nullptr;
        const Cand* nx = r + 1 < n ? &C.cands[rem[xs_sorted[r + 1]]] : nullptr;
        // window over ALL candidates of the measure, x order, +-2
        int pos = C.allpos[c.cid];
        int N = (int)C.allx.size();
        int grp_n = 0, grp_w = 0, grp_s = 0;
        double grp_rsum = 0.0;
        for (int q = std::max(0, pos - 2); q < std::min(N, pos + 3); q++) {
            if (q == pos) continue;
            const Cand& z = C.cands[C.allx[q]];
            grp_n++;
            if (!(z.hwa == 1 && z.hwn == 1)) grp_w++;
            grp_rsum += (double)z.hwn / (double)std::max(1, z.hwa);
            if (z.staff == c.staff) grp_s++;
        }
        double grp_r = grp_n ? grp_rsum / grp_n : 0.0;
        double grp_sf = grp_n ? (double)grp_s / grp_n : 0.0;
        auto ratio = [](const Cand* z) { return z == nullptr ? 0.0 : (double)z->hwn / (double)std::max(1, z->hwa); };
        double base = std::fabs(sum_hint - rem_ticks);
        double best_gap = std::fabs(sum_hint - rem_ticks);
        int best_delta = 0, n_better = 0;
        for (auto& da : alts) {
            double gap = std::fabs(sum_hint - hd + da.second - rem_ticks);
            if (gap < best_gap - 1e-9) { best_gap = gap; best_delta = da.first; }
        }
        for (auto& da : alts)
            if (std::fabs(sum_hint - hd + da.second - rem_ticks) < base - 1e-9) n_better++;
        double ss = staff_sum[c.staff];
        double L1 = std::max(1.0, length);
        double* f = &F[(size_t)i * CAND_DIM];
        f[0] = c.x / 1000.0;
        f[1] = c.staff / 4.0;
        f[2] = c.is_rest ? 1.0 : 0.0;
        f[3] = c.hint_d / 6.0;
        f[4] = c.hint_dots / 2.0;
        f[5] = (double)c.hwn / (double)std::max(1, c.hwa);
        f[6] = dx / 1000.0;
        f[7] = std::fabs(dx) / 1000.0;
        f[8] = hd / length;
        f[9] = (rem_ticks - hd) / length;
        f[10] = n ? (double)order[i] / (double)std::max(1, n - 1) : 0.0;
        f[11] = sum_hint / L1;
        f[12] = misfit;
        f[13] = std::min(frac, 4.0) / 4.0;
        f[14] = pv ? (c.x - pv->x) / 1000.0 : 1.0;
        f[15] = nx ? (nx->x - c.x) / 1000.0 : 1.0;
        f[16] = (pv && pv->staff == c.staff) ? 1.0 : 0.0;
        f[17] = (nx && nx->staff == c.staff) ? 1.0 : 0.0;
        f[18] = ratio(pv);
        f[19] = ratio(nx);
        f[20] = grp_w / 4.0;
        f[21] = grp_r;
        f[22] = grp_sf;
        f[23] = (c.hwa == 1 && c.hwn == 1) ? 0.0 : 1.0;
        f[24] = (sum_hint - hd) / L1;
        f[25] = (sum_hint - hd - rem_ticks) / L1;
        f[26] = (ss - hd) / L1;
        f[27] = (ss - hd - rem_ticks) / L1;
        f[28] = best_gap / L1;
        f[29] = best_delta / 2.0;
        f[30] = n_better / 2.0;
    }
}

void rowls(double* x, int k) {
    double m = x[0];
    for (int j = 1; j < k; j++) m = std::max(m, x[j]);
    double z[16];
    for (int j = 0; j < k; j++) { x[j] = x[j] - m; z[j] = std::exp(x[j]); }
    double s = pairwise_sum(z, k);
    double l = std::log(s + 1e-300);
    for (int j = 0; j < k; j++) x[j] = x[j] - l;
}

// The picker MLP for several rows at once.  Row r uses features F[r], the state projection
// sw[sw_of_row[r]] and candidate cand_of_row[r] (hint anchoring).  Every output element is computed
// with the same operations in the same order as one row at a time (h2[r][j] accumulates over k
// ascending, skipping hk == 0), so the values are bitwise those of the row-by-row loop; but each
// W2 row (2 MB in all) is read once for all rows instead of once per row (the A53 L2 is 1 MB).
void forward_rows(const Ctx& C, const double* F, int nrows, const double* sw, const int* sw_of_row,
                  const int* cand_of_row, double* succ, double* D, double* O, double* W) {
    PROF(3);
    const Model& M = *C.M;
    int H = M.H, H2 = M.H2, nw = M.nw;
    std::vector<double> h1((size_t)nrows * H, 0.0), h2((size_t)nrows * H2, 0.0);
    for (int r = 0; r < nrows; r++) {
        const double* f = F + (size_t)r * CAND_DIM;
        double* a = &h1[(size_t)r * H];
        for (int k = 0; k < CAND_DIM; k++) {
            double fk = f[k];
            const double* w = &M.W1[(size_t)k * H];
            for (int j = 0; j < H; j++) a[j] += fk * w[j];
        }
        const double* s = sw + (size_t)sw_of_row[r] * H;
        for (int j = 0; j < H; j++) {
            double z = (a[j] + s[j]) + M.b1[j];
            a[j] = z > 0.0 ? z : 0.0;
        }
    }
    for (int k = 0; k < H; k++) {
        const double* w = &M.W2[(size_t)k * H2];
        for (int r = 0; r < nrows; r++) {
            double hk = h1[(size_t)r * H + k];
            if (hk == 0.0) continue;
            double* b = &h2[(size_t)r * H2];
            for (int j = 0; j < H2; j++) b[j] += hk * w[j];
        }
    }
    for (int r = 0; r < nrows; r++) {
        double* h = &h2[(size_t)r * H2];
        for (int j = 0; j < H2; j++) {
            double z = h[j] + M.b2[j];
            h[j] = z > 0.0 ? z : 0.0;
        }
        double s = 0.0, tz = 0.0;
        double* dl = D + (size_t)r * N_D;
        double* ol = O + (size_t)r * N_DOTS;
        double* wl = W + (size_t)r * nw;
        for (int c = 0; c < N_D; c++) dl[c] = 0.0;
        for (int c = 0; c < N_DOTS; c++) ol[c] = 0.0;
        for (int c = 0; c < nw; c++) wl[c] = 0.0;
        for (int j = 0; j < H2; j++) {
            double hj = h[j];
            s += hj * M.w2[j];
            tz += hj * M.wt[j];
            for (int c = 0; c < N_D; c++) dl[c] += hj * M.W3[(size_t)j * N_D + c];
            for (int c = 0; c < N_DOTS; c++) ol[c] += hj * M.W4[(size_t)j * N_DOTS + c];
            for (int c = 0; c < nw; c++) wl[c] += hj * M.W5[(size_t)j * nw + c];
        }
        for (int c = 0; c < N_D; c++) dl[c] += M.b3[c];
        for (int c = 0; c < N_DOTS; c++) ol[c] += M.b4[c];
        for (int c = 0; c < nw; c++) wl[c] += M.b5[c];
        if (succ) succ[r] = s;
        double z = tz + M.bt;
        double az = std::fabs(z), e = std::exp(-az);
        double t = z >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
        const Cand& c = C.cands[cand_of_row[r]];
        int di = std::min(std::max(c.hint_d, 0), N_D - 1);
        int oi = std::min(std::max(c.hint_dots, 0), N_DOTS - 1);
        int wi = c.hwi >= 0 ? c.hwi : 0;
        dl[di] += M.beta[0] * t;
        ol[oi] += M.beta[1] * t;
        wl[wi] += M.beta[2] * t;
        rowls(dl, N_D);
        rowls(ol, N_DOTS);
        rowls(wl, nw);
    }
}

// pickers state features and their projection through W1s (once per pick)
void state_projection(const Ctx& C, int n_rem, double acc, int n_closed, int cur_len, double* S, double* sw) {
    const Model& M = *C.M;
    double length = C.length;
    double s6[STATE_DIM] = {acc / length, (length - acc) / length, (double)n_rem / std::max(1, C.n),
                            n_closed / 4.0, cur_len / 8.0, 1.0};
    int H = M.H;
    for (int j = 0; j < H; j++) sw[j] = 0.0;
    for (int k = 0; k < STATE_DIM; k++) {
        S[k] = s6[k];
        const double* w = &M.W1s[(size_t)k * H];
        for (int j = 0; j < H; j++) sw[j] += s6[k] * w[j];
    }
}

// row0_only: attribute distributions of rem[0] only (two_stage.dp_refine reads row 0), no successor softmax
void pick(const Ctx& C, const std::vector<int>& rem, double acc, double last_x, int n_closed, int cur_len, PickOut& out,
          bool row0_only = false) {
    const Model& M = *C.M;
    int n = (int)rem.size();
    std::vector<double> F;
    cand_features(C, rem, acc, last_x, F);
    double S[STATE_DIM];
    std::vector<double> sw(M.H);
    state_projection(C, n, acc, n_closed, cur_len, S, sw.data());
    out.succ.assign(n + 1, 0.0);
    out.D.assign((size_t)n * N_D, 0.0);
    out.O.assign((size_t)n * N_DOTS, 0.0);
    out.W.assign((size_t)n * M.nw, 0.0);
    int nrow = row0_only ? std::min(n, 1) : n;
    std::vector<int> zero(nrow, 0);
    forward_rows(C, F.data(), nrow, sw.data(), zero.data(), rem.data(), out.succ.data(), out.D.data(), out.O.data(),
                 out.W.data());
    if (row0_only) return;
    double eos = 0.0;
    for (int k = 0; k < STATE_DIM; k++) eos += S[k] * M.we[k];
    eos = eos + M.be;
    out.succ[n] = eos;
    double m = out.succ[0];
    for (int j = 1; j <= n; j++) m = std::max(m, out.succ[j]);
    std::vector<double> z(n + 1);
    for (int j = 0; j <= n; j++) { out.succ[j] = out.succ[j] - m; z[j] = std::exp(out.succ[j]); }
    double l = std::log(pairwise_sum(z.data(), n + 1) + 1e-300);
    for (int j = 0; j <= n; j++) out.succ[j] = out.succ[j] - l;
}

// Row 0 of many independent picks at once (dp_refine and column joint: one pick per event).
// D/O/W rows (N_D, N_DOTS, nw doubles each) come back in request order.
struct Row0Req { std::vector<int> rem; double acc, last_x; int n_closed, cur_len; };
void attr_rows0(const Ctx& C, const std::vector<Row0Req>& req, std::vector<double>& D, std::vector<double>& O,
                std::vector<double>& W) {
    const Model& M = *C.M;
    int m = (int)req.size();
    std::vector<double> F((size_t)m * CAND_DIM), sw((size_t)m * M.H), feats, S(STATE_DIM);
    std::vector<int> sw_of(m), cand_of(m);
    for (int q = 0; q < m; q++) {
        const Row0Req& r = req[q];
        cand_features(C, r.rem, r.acc, r.last_x, feats);
        std::copy(feats.begin(), feats.begin() + CAND_DIM, F.begin() + (size_t)q * CAND_DIM);
        state_projection(C, (int)r.rem.size(), r.acc, r.n_closed, r.cur_len, S.data(), &sw[(size_t)q * M.H]);
        sw_of[q] = q;
        cand_of[q] = r.rem[0];
    }
    D.assign((size_t)m * N_D, 0.0);
    O.assign((size_t)m * N_DOTS, 0.0);
    W.assign((size_t)m * M.nw, 0.0);
    if (m) forward_rows(C, F.data(), m, sw.data(), sw_of.data(), cand_of.data(), nullptr, D.data(), O.data(), W.data());
}

// pair_succ.PairSuccModel.score_prob
void pair_scores(const Ctx& C, int last_cid, const std::vector<int>& rem, std::vector<double>& ps) {
    PROF(4);
    const Model& M = *C.M;
    ps.assign(rem.size(), 0.0);
    const Cand& a = C.cands[last_cid];
    for (size_t i = 0; i < rem.size(); i++) {
        const Cand& b = C.cands[rem[i]];
        double f[10] = {
            (b.x - a.x) / 100.0,
            (a.midi > 0 && b.midi > 0) ? (b.midi - a.midi) / 12.0 : 0.0,
            b.staff == a.staff ? 1.0 : 0.0,
            (double)a.stem,
            (double)b.stem,
            (double)a.stem * (double)b.stem,
            (a.stem != 0 && a.stem == b.stem) ? 1.0 : 0.0,
            (a.stem != 0 && b.stem != 0 && a.stem != b.stem) ? 1.0 : 0.0,
            b.x >= a.x ? 1.0 : 0.0,
            (a.is_rest || b.is_rest) ? 1.0 : 0.0,
        };
        double z = 0.0;
        for (int k = 0; k < 10; k++) z += ((f[k] - M.pmu[k]) / M.psd[k]) * M.pw[k];
        z = z + M.pb;
        z = std::min(30.0, std::max(-30.0, z));
        ps[i] = -std::log1p(std::exp(-z));
    }
}

struct Opt { int d, o, w; double lp; };

// search._attr_options (wide=False, warp_bias=None)
void attr_options(const Ctx& C, const double* d_lp, const double* o_lp, const double* w_lp, const Cand& c,
                  int k, double need, bool exact_only, std::vector<Opt>& out, bool use_fit = true) {
    PROF(5);
    const Model& M = *C.M;
    const Params& P = *C.P;
    const Table& T = C.T;
    int nd = N_D, no = N_DOTS, nw = M.nw;
    int dk = std::min(nd, P.d_top_n), ok = std::min(no, P.o_top_n), wk = std::min(nw, P.warp_top);
    auto top = [](const double* v, int len, int kk) {
        std::vector<int> idx(len);
        for (int i = 0; i < len; i++) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return -v[a] < -v[b]; });
        idx.resize(kk);
        return idx;
    };
    std::vector<int> d_top = top(d_lp, nd, dk), o_top = top(o_lp, no, ok), w_top = top(w_lp, nw, wk);
    std::unordered_map<int, double> lp_of;
    struct Sc { double lp; int d, o, w; };
    std::vector<Sc> scored;
    for (int d : d_top)
        for (int o : o_top) {
            double base = d_lp[d] + o_lp[o];
            for (int wi : w_top) {
                double lp = base + w_lp[wi];
                lp_of[T.key(d, o, wi)] = lp;
                scored.push_back({lp, d, o, wi});
            }
        }
    auto cpen = [&](int key) { return (P.canon_pen == 0.0 || T.canon_ok[key]) ? 0.0 : P.canon_pen; };
    auto lpget = [&](int key) { auto it = lp_of.find(key); return it == lp_of.end() ? -12.0 : it->second; };
    int fit = -1;
    if (use_fit && need > EPS) {
        auto& dv = T.dur_vals;
        int i = (int)(std::lower_bound(dv.begin(), dv.end(), need) - dv.begin());
        int j;
        if (i <= 0) j = 0;
        else if (i >= (int)dv.size()) j = (int)dv.size() - 1;
        else j = (dv[i] - need) < (need - dv[i - 1]) ? i : i - 1;
        double dur = dv[j];
        int key = T.dur_keys[j];
        if (std::fabs(dur - need) <= std::max(1.0, 0.02 * need)) fit = T.canon[key] >= 0 ? T.canon[key] : key;
    }
    out.clear();
    auto unpack = [&](int key, int& d, int& o, int& w) { w = key % nw; int t = key / nw; o = t % 3; d = t / 3; };
    if (exact_only) {
        if (fit < 0) return;
        int d, o, w;
        unpack(fit, d, o, w);
        out.push_back({d, o, w, lpget(fit) - 2.0 - cpen(fit)});
        return;
    }
    std::stable_sort(scored.begin(), scored.end(), [&](const Sc& a, const Sc& b) {
        return -(a.lp - cpen(T.key(a.d, a.o, a.w))) < -(b.lp - cpen(T.key(b.d, b.o, b.w)));
    });
    std::unordered_set<int> seen;
    for (auto& s : scored) {
        int key = T.key(s.d, s.o, s.w);
        if (seen.count(key)) continue;
        seen.insert(key);
        out.push_back({s.d, s.o, s.w, s.lp - cpen(key)});
        if ((int)out.size() >= k) break;
    }
    // hint: python checks the RAW hint tuple against `seen` (warp may be outside the vocab)
    bool hint_in_seen = c.hwi >= 0 && c.hint_d >= 0 && c.hint_d < 7 && c.hint_dots >= 0 && c.hint_dots < 3 &&
                        seen.count(T.key(c.hint_d, c.hint_dots, c.hwi));
    if (!hint_in_seen && c.hint_d >= 0 && c.hint_d < nd && c.hint_dots >= 0 && c.hint_dots < no) {
        int wi = c.hwi >= 0 ? c.hwi : 0;
        int hk = T.key(c.hint_d, c.hint_dots, wi);
        seen.insert(hk);
        out.push_back({c.hint_d, c.hint_dots, wi, lpget(hk) - 1.0 - cpen(hk)});
    }
    int flip = (c.hwa == 1 && c.hwn == 1) ? M.idx32 : M.idx11;   // (3,2) or (1,1), if in vocab
    if (flip >= 0 && c.hint_d >= 0 && c.hint_d < nd && c.hint_dots >= 0 && c.hint_dots < no) {
        int fk = T.key(c.hint_d, c.hint_dots, flip);
        if (!seen.count(fk)) {
            seen.insert(fk);
            out.push_back({c.hint_d, c.hint_dots, flip, lpget(fk) - 1.5 - cpen(fk)});
        }
    }
    if (fit >= 0 && !seen.count(fit)) {
        int d, o, w;
        unpack(fit, d, o, w);
        out.push_back({d, o, w, lpget(fit) - 2.0 - cpen(fit)});
    }
}

struct Node {
    uint64_t used = 0;
    Voice cur;
    double acc = 0.0;
    std::shared_ptr<const Voices> voices;
    double logp = 0.0;
    int decisions = 0;
    double tau = 0.0, priority = 0.0;
    int hdev = 0;
};

int popcount64(uint64_t x) { int c = 0; while (x) { x &= x - 1; c++; } return c; }

std::string node_key(const Node& nd) {
    PROF(6);
    std::string s((const char*)&nd.used, sizeof(nd.used));
    for (auto& e : nd.cur) {
        uint32_t v = (uint32_t)e.cid | ((uint32_t)e.d << 8) | ((uint32_t)e.o << 12) | ((uint32_t)e.w << 16);
        s.append((const char*)&v, 4);
    }
    return s;
}

// search.greedy (x order, hints only)
bool greedy(const Ctx& C, Voices& out) {
    const Model& M = *C.M;
    std::vector<int> order(C.n);
    for (int i = 0; i < C.n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return C.cands[a].x < C.cands[b].x; });
    Voices voices;
    Voice cur;
    double acc = 0.0;
    for (int cid : order) {
        const Cand& c = C.cands[cid];
        int d = c.hint_d, o = c.hint_dots, w = c.hwi;
        double dur;
        bool in_table = d >= 0 && d < 7 && o >= 0 && o < 3;
        if (in_table && w >= 0) dur = C.T.dur[C.T.key(d, o, w)];
        else {
            w = M.idx11;
            dur = in_table ? C.T.dur[C.T.key(d, o, w)] : 480.0;
        }
        if (acc + dur > C.length + EPS && !cur.empty()) { voices.push_back(cur); cur.clear(); acc = 0.0; }
        cur.push_back({cid, c.x, acc, dur, d, o, w});
        acc += dur;
        if (std::fabs(acc - C.length) <= EPS) { voices.push_back(cur); cur.clear(); acc = 0.0; }
    }
    bool truncated = !cur.empty();
    if (!cur.empty()) voices.push_back(cur);
    bool valid = !truncated && !voices.empty();
    if (valid)
        for (auto& v : voices) {
            double s = 0.0;
            for (auto& e : v) s += e.dur;
            if (std::fabs(s - C.length) > EPS) { valid = false; break; }
        }
    out = voices;
    return valid;
}

// search._greedy_complete (anytime fallback)
bool greedy_complete(const Ctx& C, const Node& node, Voices& out) {
    Voices voices = node.voices ? *node.voices : Voices();
    Voice cur = node.cur;
    double acc = node.acc;
    uint64_t used = node.used;
    PickOut po;
    std::vector<Opt> opts;
    for (int it = 0; it < C.n + 8; it++) {
        std::vector<int> rem;
        for (int i = 0; i < C.n; i++) if (!((used >> i) & 1)) rem.push_back(i);
        if (rem.empty()) {
            if (!cur.empty() && std::fabs(acc - C.length) <= EPS) voices.push_back(cur);
            bool ok = (!voices.empty() && cur.empty()) || std::fabs(acc - C.length) <= EPS;
            out = voices;
            return ok;
        }
        if (!cur.empty() && std::fabs(acc - C.length) <= EPS) { voices.push_back(cur); cur.clear(); acc = 0.0; continue; }
        double need = C.length - acc;
        pick(C, rem, acc, cur.empty() ? 0.0 : cur.back().x, (int)voices.size(), (int)cur.size(), po);
        std::vector<int> order(rem.size());
        for (size_t i = 0; i < rem.size(); i++) order[i] = (int)i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return -po.succ[a] < -po.succ[b]; });
        bool placed = false;
        for (int i : order) {
            const Cand& c = C.cands[rem[i]];
            // python's fallback passes no dur_vals, so its _attr_options never offers the gap-filling `fit`
            attr_options(C, &po.D[(size_t)i * N_D], &po.O[(size_t)i * N_DOTS], &po.W[(size_t)i * C.M->nw], c, 6, need,
                         rem.size() == 1, opts, false);
            int bd = -1, bo = 0, bw = 0;
            double bdur = 0.0;
            for (auto& op : opts) {
                double dur = C.T.dur[C.T.key(op.d, op.o, op.w)];
                if (dur > need + EPS) continue;
                if (bd < 0 || dur > bdur) { bdur = dur; bd = op.d; bo = op.o; bw = op.w; }
            }
            if (bd < 0) continue;
            cur.push_back({c.cid, c.x, acc, bdur, bd, bo, bw});
            used |= (uint64_t)1 << c.cid;
            acc += bdur;
            placed = true;
            break;
        }
        if (!placed) return false;
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------- C API
API void* dec_create(int H, int H2, int nw, const double* W1, const double* W1s, const double* b1, const double* W2,
                     const double* b2, const double* w2, const double* W3, const double* b3, const double* W4,
                     const double* b4, const double* W5, const double* b5, const double* we, double be,
                     const double* beta, const double* wt, double bt, const int* wv, const double* pmu,
                     const double* psd, const double* pw, double pb) {
    Model* M = new Model();
    M->H = H; M->H2 = H2; M->nw = nw;
    M->W1.assign(W1, W1 + (size_t)CAND_DIM * H);
    M->W1s.assign(W1s, W1s + (size_t)STATE_DIM * H);
    M->b1.assign(b1, b1 + H);
    M->W2.assign(W2, W2 + (size_t)H * H2);
    M->b2.assign(b2, b2 + H2);
    M->w2.assign(w2, w2 + H2);
    M->W3.assign(W3, W3 + (size_t)H2 * N_D);
    M->b3.assign(b3, b3 + N_D);
    M->W4.assign(W4, W4 + (size_t)H2 * N_DOTS);
    M->b4.assign(b4, b4 + N_DOTS);
    M->W5.assign(W5, W5 + (size_t)H2 * nw);
    M->b5.assign(b5, b5 + nw);
    M->we.assign(we, we + STATE_DIM);
    M->be = be;
    M->beta.assign(beta, beta + 3);
    M->wt.assign(wt, wt + H2);
    M->bt = bt;
    for (int i = 0; i < nw; i++) {
        M->wa.push_back(wv[2 * i]);
        M->wn.push_back(wv[2 * i + 1]);
        if (wv[2 * i] == 1 && wv[2 * i + 1] == 1) M->idx11 = i;
        if (wv[2 * i] == 3 && wv[2 * i + 1] == 2) M->idx32 = i;
    }
    std::memcpy(M->pmu, pmu, sizeof(M->pmu));
    std::memcpy(M->psd, psd, sizeof(M->psd));
    std::memcpy(M->pw, pw, sizeof(M->pw));
    M->pb = pb;
    return M;
}

API void dec_free(void* m) { delete (Model*)m; }

// ints: n x 10 [cid staff is_rest hint_d hint_dots hint_wa hint_wn hint_wi midi stem], xs: n
// params: see Params (ints first as doubles); out_ev: n x 7 [voice cid onset dur d o w]
// out_stats: [expanded complete loss depth n_voices]
namespace {
// parameters + per-measure context shared by dec_solve and dec_pick; returns 0 or an error code
int setup_ctx(const Model& M, int n, const int* ints, const double* xs, double length, int n_beats,
              const double* prm, Params& P, Ctx& C) {

    P.beam = (int)prm[0]; P.k_attr = (int)prm[1]; P.warp_top = (int)prm[2]; P.d_top_n = (int)prm[3];
    P.o_top_n = (int)prm[4]; P.max_expansions = (int)prm[5]; P.iso_max = (int)prm[6];
    P.pretense_tol = prm[7]; P.lam_struct = prm[8]; P.progress_w = prm[9]; P.voice_w = prm[10]; P.dev_w = prm[11];
    P.tau_w = prm[12]; P.canon_pen = prm[13]; P.iso_w = prm[14]; P.iso_tol = prm[15]; P.lam_iso = prm[16];
    P.hint_w = prm[17]; P.pair_w = prm[18]; P.close_cost = prm[19]; P.stem_w = prm[20]; P.staff_w = prm[21];
    P.width = prm[22]; P.lam_hint = prm[23];
    C.M = &M; C.P = &P; C.T = make_table(M); C.n = n; C.length = length; C.n_beats = n_beats;
    C.cands.resize(n);
    for (int i = 0; i < n; i++) {
        const int* r = ints + (size_t)i * 10;
        Cand c;
        c.cid = r[0]; c.staff = r[1]; c.is_rest = r[2]; c.hint_d = r[3]; c.hint_dots = r[4]; c.hwa = r[5];
        c.hwn = r[6]; c.hwi = r[7]; c.midi = r[8]; c.stem = r[9]; c.x = xs[i];
        if (c.cid != i) return -2;          // candidates must arrive in cid order 0..n-1
        C.cands[i] = c;
    }
    C.hmap_d.resize(n); C.hmap_o.resize(n); C.hmap_w.resize(n);
    double sum_hd = 0.0;
    for (int i = 0; i < n; i++) {
        const Cand& c = C.cands[i];
        C.hmap_d[i] = std::min(std::max(c.hint_d, 0), 6);
        C.hmap_o[i] = std::min(std::max(c.hint_dots, 0), 2);
        C.hmap_w[i] = c.hwi >= 0 ? c.hwi : 0;
        sum_hd += hint_duration(c);
    }
    C.k_est = std::max(1, (int)std::nearbyint(sum_hd / length));
    C.allx.resize(n);
    for (int i = 0; i < n; i++) C.allx[i] = i;
    std::stable_sort(C.allx.begin(), C.allx.end(), [&](int a, int b) { return C.cands[a].x < C.cands[b].x; });
    C.allpos.resize(n);
    for (int q = 0; q < n; q++) C.allpos[C.allx[q]] = q;

    return 0;
}
}  // namespace

API int dec_solve(void* model, int n, const int* ints, const double* xs, double length, int n_beats,
                  const double* prm, double* out_ev, double* out_stats) {
    const Model& M = *(const Model*)model;
    if (n <= 0) { out_stats[0] = 0; out_stats[1] = 1; out_stats[2] = 0; out_stats[3] = 0; out_stats[4] = 0; return 0; }
    if (n > 63) return -1;
    Params P;
    Ctx C;
    if (int err = setup_ctx(M, n, ints, xs, length, n_beats, prm, P, C)) return err;
    Node start;
    start.voices = std::make_shared<Voices>();
    std::vector<Node> frontier{start};
    std::unordered_map<std::string, double> seen;
    seen[node_key(start)] = 0.0;
    Node best_partial = start;
    long expanded = 0;
    int depth = 0;
    Voices best_complete;
    bool have_complete = false;
    double best_loss = 1e308 * 10;
    {
        Voices gv;
        if (greedy(C, gv) && !gv.empty()) {
            best_complete = gv;
            have_complete = true;
            best_loss = final_loss(C, gv, 0.0, 1);
        }
    }
    PickOut po;
    std::unordered_map<std::string, PickOut> pick_cache;
    std::vector<double> ps;
    std::vector<std::vector<Opt>> opts;
    const char* trace_path = std::getenv("DEC_TRACE");   // debug: append each depth's frontier to this file
    while (!frontier.empty() && depth < n + 6) {
        if (P.max_expansions && expanded >= P.max_expansions) break;
        depth++;
        std::vector<Node> children;
        for (const Node& node : frontier) {
            PROF(7);
            expanded++;
            std::vector<int> rem;
            for (int i = 0; i < n; i++) if (!((node.used >> i) & 1)) rem.push_back(i);
            // ---- EOS
            if (!node.cur.empty() && std::fabs(node.acc - length) <= EPS) {
                auto voices = std::make_shared<Voices>(*node.voices);
                voices->push_back(node.cur);
                if (rem.empty()) {
                    double loss = final_loss(C, *voices, node.logp, node.decisions);
                    if (loss < best_loss) { best_loss = loss; best_complete = *voices; have_complete = true; }
                } else {
                    Node ch;
                    ch.used = node.used;
                    ch.acc = 0.0;
                    ch.voices = voices;
                    ch.logp = node.logp - P.close_cost;
                    ch.decisions = node.decisions + 1;
                    ch.tau = max_tau(*voices, length, P.width);
                    ch.hdev = node.hdev;
                    ch.priority = node.logp - P.lam_struct * ch.tau -
                                  P.lam_hint * ((double)ch.hdev / std::max(1, ch.decisions)) +
                                  P.progress_w * ((double)popcount64(node.used) / n);
                    children.push_back(std::move(ch));
                }
            }
            if (rem.empty()) continue;
            {
                // pick() depends only on (remaining set, acc, last x, closed voices, current length): nodes that
                // reach the same state through other attribute choices reuse the result (a copy; the pair
                // scores are added to it below).  Same inputs, same outputs: bitwise identical.
                double last_x = node.cur.empty() ? 0.0 : node.cur.back().x;
                int n_closed = (int)node.voices->size(), cur_len = (int)node.cur.size();
                std::string key((const char*)&node.used, sizeof(node.used));
                key.append((const char*)&node.acc, sizeof(double));
                key.append((const char*)&last_x, sizeof(double));
                key.append((const char*)&n_closed, sizeof(int));
                key.append((const char*)&cur_len, sizeof(int));
                auto hit = pick_cache.find(key);
#ifdef DEC_PROF
                g_prof_n[9]++;
                if (hit != pick_cache.end()) g_prof_n[10]++;
#endif
                if (hit != pick_cache.end()) po = hit->second;
                else {
                    pick(C, rem, node.acc, last_x, n_closed, cur_len, po);
                    pick_cache.emplace(std::move(key), po);
                }
            }
            int nr = (int)rem.size();
            if (P.pair_w != 0.0 && !node.cur.empty()) {
                pair_scores(C, node.cur.back().cid, rem, ps);
                for (int i = 0; i < nr; i++) po.succ[i] = po.succ[i] + P.pair_w * ps[i];
            }
            double node_max_tau = max_tau(*node.voices, length, P.width);
            double need = length - node.acc;
            opts.assign(nr, {});
            for (int i = 0; i < nr; i++)
                attr_options(C, &po.D[(size_t)i * N_D], &po.O[(size_t)i * N_DOTS], &po.W[(size_t)i * M.nw],
                             C.cands[rem[i]], P.k_attr, need, nr == 1, opts[i]);
            for (int i = 0; i < nr; i++) {
                const Cand& c = C.cands[rem[i]];
                for (auto& op : opts[i]) {
                    double dur = C.T.dur[C.T.key(op.d, op.o, op.w)];
                    double new_acc = node.acc + dur;
                    if (new_acc > length + EPS) continue;
                    if (new_acc < length - EPS) {
                        if (nr == 1) continue;
                        if ((length - new_acc) < C.T.min_one - EPS) continue;
                    }
                    double logp = node.logp + po.succ[i] + op.lp;
                    if ((-logp) / std::max(1, node.decisions + 1) > P.pretense_tol) continue;
                    Node ch;
                    ch.used = node.used | ((uint64_t)1 << c.cid);
                    ch.cur = node.cur;
                    ch.cur.push_back({c.cid, c.x, node.acc, dur, op.d, op.o, op.w});
                    ch.acc = new_acc;
                    ch.voices = node.voices;
                    ch.logp = logp;
                    ch.decisions = node.decisions + 1;
                    double ctau = tau_voice(ch.cur, length, P.width);
                    if (ctau >= 2.0) continue;
                    ch.tau = std::max(ctau, node_max_tau);
                    bool same = C.hmap_d[c.cid] == op.d && C.hmap_o[c.cid] == op.o && C.hmap_w[c.cid] == op.w;
                    ch.hdev = node.hdev + (same ? 0 : 1);
                    double piso = 0.0;          // lam_iso is 0 in production; the term is skipped
                    ch.priority = logp - P.lam_struct * ch.tau - P.lam_iso * piso -
                                  P.lam_hint * ((double)ch.hdev / std::max(1, ch.decisions)) +
                                  P.progress_w * ((double)popcount64(ch.used) / n);
                    children.push_back(std::move(ch));
                }
            }
        }
        if (children.empty()) break;
        {
        PROF(8);
        std::stable_sort(children.begin(), children.end(), [](const Node& a, const Node& b) { return -a.priority < -b.priority; });
        std::vector<Node> nxt;
        for (auto& ch : children) {
            std::string k = node_key(ch);
            if (seen.count(k)) continue;
            seen[k] = ch.priority;
            nxt.push_back(std::move(ch));
            if ((int)nxt.size() >= P.beam) break;
        }
        frontier = std::move(nxt);
        }
        if (trace_path) {
            const char* tp = trace_path;
            if (FILE* f = std::fopen(tp, "a")) {
                for (auto& ch : frontier) {
                    std::fprintf(f, "%d %llu |", depth, (unsigned long long)ch.used);
                    for (auto& e : ch.cur) std::fprintf(f, " %d,%d,%d,%d", e.cid, e.d, e.o, e.w);
                    std::fprintf(f, " | %.17g %.17g\n", ch.priority, ch.logp);
                }
                std::fclose(f);
            }
        }
        for (auto& ch : frontier)
            if (popcount64(ch.used) > popcount64(best_partial.used)) best_partial = ch;
    }
    bool complete;
    Voices out;
    double loss = 0.0;
    if (!have_complete) {
        Voices comp;
        if (greedy_complete(C, best_partial, comp)) {
            best_complete = comp;
            have_complete = true;
            // python's fallback scores without the hint/stem/staff terms
            Params P2 = P;
            P2.hint_w = 0.0; P2.stem_w = 0.0; P2.staff_w = 0.0;
            Ctx C2 = C;
            C2.P = &P2;
            best_loss = final_loss(C2, best_complete, best_partial.logp, std::max(1, best_partial.decisions));
        }
    }
    if (have_complete) { out = best_complete; complete = true; loss = best_loss; }
    else {
        out = *best_partial.voices;
        if (!best_partial.cur.empty()) out.push_back(best_partial.cur);
        complete = false;
    }
    int k = 0;
    for (size_t v = 0; v < out.size(); v++)
        for (auto& e : out[v]) {
            double* r = out_ev + (size_t)k * 7;
            r[0] = (double)v; r[1] = e.cid; r[2] = e.onset; r[3] = e.dur; r[4] = e.d; r[5] = e.o; r[6] = e.w;
            k++;
        }
    out_stats[0] = (double)expanded; out_stats[1] = complete ? 1.0 : 0.0; out_stats[2] = loss;
    out_stats[3] = depth; out_stats[4] = (double)out.size();
    return k;
}

// debug: the picker's outputs for one search state (rem = cids not in `used`), n_rem x (1 | 7 | 3 | nw)
API int dec_pick(void* model, int n, const int* ints, const double* xs, double length, int n_beats,
                 const double* prm, unsigned long long used, double acc, double last_x, int n_closed, int cur_len,
                 double* out_succ, double* out_D, double* out_O, double* out_W) {
    const Model& M = *(const Model*)model;
    if (n <= 0 || n > 63) return -1;
    Params P;
    Ctx C;
    if (int err = setup_ctx(M, n, ints, xs, length, n_beats, prm, P, C)) return err;
    std::vector<int> rem;
    for (int i = 0; i < n; i++) if (!((used >> i) & 1)) rem.push_back(i);
    PickOut po;
    pick(C, rem, acc, last_x, n_closed, cur_len, po);
    std::memcpy(out_succ, po.succ.data(), sizeof(double) * po.succ.size());
    std::memcpy(out_D, po.D.data(), sizeof(double) * po.D.size());
    std::memcpy(out_O, po.O.data(), sizeof(double) * po.O.size());
    std::memcpy(out_W, po.W.data(), sizeof(double) * po.W.size());
    return (int)rem.size();
}

// ---------------------------------------------------------------- two_stage.dp_refine (attr_dp)
namespace {
struct DOpt { double dur; int d, o, w; double cost; };

// attr_dp.build_options (canon_pen 0, dur_bias 0, cx_bonus 0): all (d, o, w), cheapest k_attr, stable
void build_options(const Ctx& C, const double* D, const double* O, const double* W, const Cand& c, int k_attr,
                   double hint_w, std::vector<DOpt>& out) {
    const Model& M = *C.M;
    const Table& T = C.T;
    int hd = std::min(std::max(c.hint_d, 0), 6), ho = std::min(std::max(c.hint_dots, 0), 2);
    out.clear();
    for (int d = 0; d < N_D; d++)
        for (int o = 0; o < N_DOTS; o++)
            for (int wi = 0; wi < M.nw; wi++) {
                int key = T.key(d, o, wi);
                double cost = -((D[d] + O[o]) + W[wi]);
                if (!T.canon_ok[key]) cost += 0.0;           // python adds canon_pen (0.0): -0.0 -> 0.0
                if (hint_w != 0.0 && !(d == hd && o == ho && M.wa[wi] == c.hwa && M.wn[wi] == c.hwn)) cost += hint_w;
                out.push_back({T.dur[key], d, o, wi, cost});
            }
    std::stable_sort(out.begin(), out.end(), [](const DOpt& a, const DOpt& b) { return a.cost < b.cost; });
    out.resize(std::min(out.size(), (size_t)std::max(1, k_attr)));
}

// attr_dp.voice_dp without state penalties; false = the voice cannot be tiled
bool voice_dp(double length, const std::vector<std::vector<DOpt>>& opts, std::vector<Ev>& out,
              const std::vector<int>& cids, const std::vector<double>& xs) {
    const double INF = std::numeric_limits<double>::infinity();
    int L = (int)std::nearbyint(length * SCALE);
    int n = (int)opts.size();
    std::vector<double> dp(L + 1, INF), nxt(L + 1);
    dp[0] = 0.0;
    std::vector<std::vector<int>> bpa(n), bpo(n);
    std::vector<int> live;                   // finite states: inf + c never beats anything, so the rest is skipped
    for (int i = 0; i < n; i++) {
        std::fill(nxt.begin(), nxt.end(), INF);
        bpa[i].assign(L + 1, -1);
        bpo[i].assign(L + 1, -1);
        live.clear();
        for (int s = 0; s <= L; s++) if (dp[s] != INF) live.push_back(s);
        for (size_t j = 0; j < opts[i].size(); j++) {
            int sh = (int)std::nearbyint(opts[i][j].dur * SCALE);
            if (sh <= 0 || sh > L) continue;
            double c = opts[i][j].cost;
            for (int s : live) {
                if (s + sh > L) break;
                double cand = dp[s] + c;
                if (cand < nxt[s + sh]) { nxt[s + sh] = cand; bpa[i][s + sh] = s; bpo[i][s + sh] = (int)j; }
            }
        }
        dp.swap(nxt);
        bool any = false;
        for (double v : dp) if (std::isfinite(v)) { any = true; break; }
        if (!any) return false;
    }
    if (!std::isfinite(dp[L])) return false;
    out.assign(n, Ev());
    int acc = L;
    for (int i = n - 1; i >= 0; i--) {
        int j = bpo[i][acc];
        if (j < 0) return false;
        int pa = bpa[i][acc];
        const DOpt& op = opts[i][j];
        out[i] = {cids[i], xs[i], pa / (double)SCALE, op.dur, op.d, op.o, op.w};
        acc = pa;
    }
    return true;
}
}  // namespace

// two_stage.dp_refine(cm, got, pk, wv, canon_ok, iters, k_attr, iso_w=0, exact_x=0) with PROD hint_w.
// in_ev: one row per event (cid onset dur d o w), voices concatenated, vlen[v] events each.
// Returns 1 and out_ev (same layout) on success, 0 where python returns `got` unchanged.
API int dec_refine(void* model, int n, const int* ints, const double* xs, double length, int n_beats,
                   const double* prm, int n_voices, const int* vlen, const double* in_ev, int iters, int k_attr,
                   double hint_w, double* out_ev) {
    const Model& M = *(const Model*)model;
    if (n <= 0 || n > 63 || n_voices <= 0) return 0;
    Params P;
    Ctx C;
    if (setup_ctx(M, n, ints, xs, length, n_beats, prm, P, C)) return -1;
    Voices cur(n_voices);
    for (int v = 0, k = 0; v < n_voices; v++)
        for (int i = 0; i < vlen[v]; i++, k++) {
            const double* r = in_ev + (size_t)k * 6;
            int cid = (int)r[0];
            cur[v].push_back({cid, C.cands[cid].x, r[1], r[2], (int)r[3], (int)r[4], (int)r[5]});
        }
    PickOut po;
    std::vector<std::vector<DOpt>> opts;
    for (int it = 0; it < std::max(1, iters); it++) {
        Voices nv;
        std::vector<Row0Req> req;
        std::vector<std::vector<int>> seqs;
        std::vector<std::vector<double>> xss;
        std::vector<double> bD, bO, bW;
        for (const Voice& v : cur) {
            int m = (int)v.size();
            std::vector<int> seq(m);
            std::vector<double> sx(m), accs(m);
            double a = 0.0;
            for (int i = 0; i < m; i++) { seq[i] = v[i].cid; sx[i] = v[i].x; accs[i] = a; a += v[i].dur; }
            for (int i = 0; i < m; i++)
                req.push_back({std::vector<int>(seq.begin() + i, seq.end()), accs[i], i ? v[i - 1].x : 0.0, 0, i});
            seqs.push_back(seq);
            xss.push_back(sx);
        }
        // every event's options depend only on `cur`: one batched MLP pass for the whole iteration
        attr_rows0(C, req, bD, bO, bW);
        for (size_t vi = 0, q = 0; vi < cur.size(); vi++) {
            int m = (int)seqs[vi].size();
            opts.assign(m, {});
            for (int i = 0; i < m; i++, q++)
                build_options(C, &bD[q * N_D], &bO[q * N_DOTS], &bW[q * M.nw], C.cands[seqs[vi][i]], k_attr, hint_w,
                              opts[i]);
            Voice out;
            if (!voice_dp(length, opts, out, seqs[vi], xss[vi])) return 0;
            nv.push_back(out);
        }
        cur.swap(nv);
    }
    for (int v = 0, k = 0; v < n_voices; v++)
        for (const Ev& e : cur[v]) {
            double* r = out_ev + (size_t)k * 6;
            r[0] = e.cid; r[1] = e.onset; r[2] = e.dur; r[3] = e.d; r[4] = e.o; r[5] = e.w;
            k++;
        }
    return 1;
}

// ---------------------------------------------------------------- column_joint.column_joint_refine
namespace {
struct CjState {
    int t0;
    std::vector<int> accs, placed;
    double cost;
    int parent;                              // index into the previous layer (as stored, after pruning)
    int vi, i, du, d, o, w;                  // action; vi < 0 = column entry (no action)
};

std::string cj_key(int t0, const std::vector<int>& accs, const std::vector<int>& placed) {
    std::string k((const char*)&t0, sizeof(int));
    k.append((const char*)accs.data(), accs.size() * sizeof(int));
    k.append((const char*)placed.data(), placed.size() * sizeof(int));
    return k;
}

struct CjLayer {
    std::vector<CjState> st;
    std::unordered_map<std::string, int> at;
    // python: `if key not in nxt or cost < nxt[key][0]: nxt[key] = ...` (insertion order kept)
    void offer(CjState&& s) {
        std::string k = cj_key(s.t0, s.accs, s.placed);
        auto it = at.find(k);
        if (it == at.end()) { at.emplace(k, (int)st.size()); st.push_back(std::move(s)); }
        else if (s.cost < st[it->second].cost) st[it->second] = std::move(s);
    }
    // python: dict(sorted(items, key=cost)[:max_states]) -- stable, new insertion order = sorted order
    void prune(int max_states) {
        if ((int)st.size() <= max_states) return;
        std::vector<int> idx(st.size());
        for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return st[a].cost < st[b].cost; });
        std::vector<CjState> keep;
        for (int q = 0; q < max_states; q++) keep.push_back(st[idx[q]]);
        st.swap(keep);
        at.clear();
        for (size_t i = 0; i < st.size(); i++) at.emplace(cj_key(st[i].t0, st[i].accs, st[i].placed), (int)i);
    }
};
}  // namespace

// column_joint_refine for one measure.  The python wrapper passes the voices already sorted by
// (x, onset) (`vs`), their durations, and the column index of every candidate (from _xkey).
// out_ev per event in vs order: onset, dur, d, o, w.  Returns 1, or 0 where python returns None.
API int dec_column_joint(void* model, int n, const int* ints, const double* xs, double length, int n_beats,
                         const double* prm, int n_voices, const int* vlen, const int* vcid, const double* vdur,
                         const int* col_of, int k_attr, double hint_w, int max_states, double* out_ev) {
    const Model& M = *(const Model*)model;
    if (n <= 0 || n > 63 || n_voices <= 0) return 0;
    Params P;
    Ctx C;
    if (setup_ctx(M, n, ints, xs, length, n_beats, prm, P, C)) return -1;
    int L = (int)std::nearbyint(length * SCALE);
    int nv = n_voices;
    std::vector<std::vector<int>> seqs(nv);
    std::vector<std::vector<double>> durs(nv);
    for (int v = 0, k = 0; v < nv; v++)
        for (int i = 0; i < vlen[v]; i++, k++) { seqs[v].push_back(vcid[k]); durs[v].push_back(vdur[k]); }
    // per-event options (dp_refine's), durations in cells
    struct Op { int du, d, o, w; double cost; };
    std::vector<std::vector<std::vector<Op>>> opts(nv);
    std::vector<DOpt> bo;
    std::vector<Row0Req> req;
    for (int v = 0; v < nv; v++) {
        int m = (int)seqs[v].size();
        double a = 0.0;
        std::vector<double> accs(m);
        for (int i = 0; i < m; i++) { accs[i] = a; a += durs[v][i]; }
        opts[v].resize(m);
        for (int i = 0; i < m; i++)
            req.push_back({std::vector<int>(seqs[v].begin() + i, seqs[v].end()), accs[i],
                           i ? C.cands[seqs[v][i - 1]].x : 0.0, 0, i});
    }
    std::vector<double> bD, bO, bW;
    attr_rows0(C, req, bD, bO, bW);                 // one batched MLP pass for every event
    for (int v = 0, q = 0; v < nv; v++)
        for (int i = 0; i < (int)seqs[v].size(); i++, q++) {
            build_options(C, &bD[(size_t)q * N_D], &bO[(size_t)q * N_DOTS], &bW[(size_t)q * M.nw],
                          C.cands[seqs[v][i]], k_attr, hint_w, bo);
            for (auto& o : bo) opts[v][i].push_back({(int)std::nearbyint(o.dur * SCALE), o.d, o.o, o.w, o.cost});
        }
    // reach[v][j][t]: the voice's last j events can fill exactly t cells
    std::vector<std::vector<std::vector<char>>> reach(nv);
    for (int v = 0; v < nv; v++) {
        int m = (int)seqs[v].size();
        reach[v].assign(m + 1, std::vector<char>(L + 1, 0));
        reach[v][0][0] = 1;
        for (int j = 1; j <= m; j++) {
            const std::vector<char>& prev = reach[v][j - 1];
            std::vector<char>& cur = reach[v][j];
            for (auto& o : opts[v][m - j]) {
                int du = o.du;
                if (du < 0 || du > L) continue;
                for (int t = du; t <= L; t++) if (prev[t - du]) cur[t] = 1;
            }
        }
    }
    int ncol = 0;
    for (int q = 0; q < n; q++) ncol = std::max(ncol, col_of[q] + 1);
    std::vector<std::vector<std::pair<int, int>>> members(ncol);
    for (int v = 0; v < nv; v++)
        for (int i = 0; i < (int)seqs[v].size(); i++) members[col_of[seqs[v][i]]].push_back({v, i});
    std::vector<CjLayer> layers;
    CjLayer frontier;
    frontier.offer({-1, std::vector<int>(nv, 0), std::vector<int>(nv, 0), 0.0, -1, -1, 0, 0, 0, 0, 0});
    for (int ci = 0; ci < ncol; ci++) {
        const auto& pres = members[ci];
        if (pres.empty()) return 0;
        std::vector<char> here(nv, 0);
        for (auto& p : pres) here[p.first] = 1;
        CjLayer nxt;
        for (int s = 0; s < (int)frontier.st.size(); s++) {
            const CjState& S = frontier.st[s];
            int t0 = S.accs[pres[0].first];
            bool bad = t0 <= S.t0;
            for (int v = 0; v < nv && !bad; v++) if (here[v] && S.accs[v] != t0) bad = true;
            for (int v = 0; v < nv && !bad; v++)
                if (!here[v] && S.placed[v] < (int)seqs[v].size() && S.accs[v] <= t0) bad = true;
            if (bad) continue;
            nxt.offer({t0, S.accs, S.placed, S.cost, s, -1, 0, 0, 0, 0, 0});
        }
        if (nxt.st.empty()) return 0;
        layers.push_back(nxt);
        frontier = layers.back();
        for (auto& p : pres) {
            int vi = p.first, i = p.second;
            int rem_after = (int)seqs[vi].size() - i - 1;
            CjLayer nx2;
            for (int s = 0; s < (int)frontier.st.size(); s++) {
                const CjState& S = frontier.st[s];
                for (auto& o : opts[vi][i]) {
                    int a2 = S.accs[vi] + o.du;
                    if (o.du <= 0 || a2 > L || !reach[vi][rem_after][L - a2]) continue;
                    CjState T{S.t0, S.accs, S.placed, S.cost + o.cost, s, vi, i, o.du, o.d, o.o, o.w};
                    T.accs[vi] = a2;
                    T.placed[vi] += 1;
                    nx2.offer(std::move(T));
                }
            }
            if (nx2.st.empty()) return 0;
            nx2.prune(max_states);
            layers.push_back(nx2);
            frontier = layers.back();
        }
    }
    // python: key = min((cost, key) for complete states)  -- ties by (t0, accs, placed)
    int best = -1;
    for (int s = 0; s < (int)frontier.st.size(); s++) {
        const CjState& S = frontier.st[s];
        bool done = true;
        for (int v = 0; v < nv; v++) if (S.accs[v] != L) { done = false; break; }
        if (!done) continue;
        if (best < 0) { best = s; continue; }
        const CjState& B = frontier.st[best];
        if (S.cost < B.cost) best = s;
        else if (!(B.cost < S.cost)) {
            if (std::tie(S.t0, S.accs, S.placed) < std::tie(B.t0, B.accs, B.placed)) best = s;
        }
    }
    if (best < 0) return 0;
    std::vector<int> offs(nv + 1, 0);
    for (int v = 0; v < nv; v++) offs[v + 1] = offs[v] + (int)seqs[v].size();
    std::vector<char> got((size_t)offs[nv], 0);
    int s = best;
    for (int li = (int)layers.size() - 1; li >= 0; li--) {
        const CjState& S = layers[li].st[s];
        if (S.vi >= 0) {
            int parent_acc = layers[li - 1].st[S.parent].accs[S.vi];
            double* r = out_ev + (size_t)(offs[S.vi] + S.i) * 5;
            r[0] = parent_acc / (double)SCALE; r[1] = S.du / (double)SCALE; r[2] = S.d; r[3] = S.o; r[4] = S.w;
            got[offs[S.vi] + S.i] = 1;
        }
        s = S.parent;
    }
    for (char g : got) if (!g) return -3;
    return 1;
}

// profiling build: out[0..15] seconds, out[16..31] calls; reset after reading
API void dec_prof(double* out) {
#ifdef DEC_PROF
    for (int i = 0; i < 16; i++) { out[i] = g_prof_t[i]; out[16 + i] = (double)g_prof_n[i]; g_prof_t[i] = 0; g_prof_n[i] = 0; }
#else
    for (int i = 0; i < 32; i++) out[i] = -1;
#endif
}
