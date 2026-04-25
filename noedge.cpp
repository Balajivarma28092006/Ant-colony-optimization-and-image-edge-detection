/*
 * ACO Edge Detection — using TIGR for I/O and display
 *
 * Variant: Smooth pheromone-to-color gradient (no hard edge threshold).
 *   Instead of a binary threshold, every pixel's normalized pheromone
 *   intensity is passed through a smooth HSV-like mapping so that even
 *   sub-threshold edges are visualized with a color that reflects their
 *   strength.  The background is a dim grayscale; as pheromone rises the
 *   hue sweeps blue → cyan → green → yellow → red, giving a perceptually
 *   intuitive "heat" reading of edge confidence.
 *
 * Usage:
 *   ./aco_edge_colormap <input_image>
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#endif
#include <omp.h>
#include <vector>
#include <string>
#include "tigr.h"

/* ── ACO parameters ──────────────────────────────────────────── */
static const int   NUM_ANTS      = 1000;
static const int   NUM_ITERATIONS = 20;
static const float ALPHA         = 1.8f;
static const float BETA          = 1.2f;
static const float RHO           = 0.1f;
static const float Q             = 1.0f;
static const float TAU_MIN       = 0.01f;
static const float TAU_MAX       = 7.0f;
static const float TAU0          = 0.1f;
static const int   ANT_STEPS     = 30;
static const float MIN_ETA       = 0.05f;

/* NOTE: EDGE_THRESHOLD is intentionally removed in this variant.
   Every pixel receives a color; the mapping itself encodes strength. */

#define max(a,b) (((a) > (b)) ? (a) : (b))

static const int DX[8] = {-1, -1, -1,  0,  0,  1,  1,  1};
static const int DY[8] = {-1,  0,  1, -1,  1, -1,  0,  1};

static inline int clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ── Image scaling ───────────────────────────────────────────── */
static Tigr* resize_image(Tigr* src, int max_dim) {
    if (src->w <= max_dim && src->h <= max_dim) return src;
    float scale = (float)max_dim / max(src->w, src->h);
    int nw = (int)(src->w * scale);
    int nh = (int)(src->h * scale);
    Tigr* scaled = tigrBitmap(nw, nh);
    for (int y = 0; y < nh; y++) {
        for (int x = 0; x < nw; x++) {
            int sx = clamp((int)(x / scale), 0, src->w - 1);
            int sy = clamp((int)(y / scale), 0, src->h - 1);
            scaled->pix[y * nw + x] = src->pix[sy * src->w + sx];
        }
    }
    return scaled;
}

/* ── Smooth pheromone → color mapping ───────────────────────────
 *
 *  t  = normalized pheromone in [0, 1]  (tau / TAU_MAX)
 *
 *  The mapping uses a piece-wise linear hue ramp over five colour stops:
 *    0.00 → dim blue-black  (almost invisible background)
 *    0.20 → pure blue        (faint edge traces)
 *    0.45 → cyan             (moderate edges)
 *    0.70 → green-yellow     (strong edges)
 *    1.00 → red              (maximum pheromone / definite edge)
 *
 *  Brightness also scales with t so near-zero regions stay dark.
 *
 *  All arithmetic is on [0,1] floats; final cast to unsigned char.
 */
static TPixel pheromone_to_color(float t) {
    /* brightness: gentle power curve so faint traces are visible but dark */
    float brightness = powf(t, 0.55f);

    /* hue ramp: five stops, each covering 0.25 of the [0,1] range */
    float r, g, b;

        /* black → blue */
        float s = t / 0.20f;
        r = 0.0f;
        g = 0.0f;
        b = s;

    r *= brightness*100;
    g *= brightness*100;
    b *= brightness*100;

    return tigrRGB(
        (unsigned char)(r * 255.f + 0.5f),
        (unsigned char)(g * 255.f + 0.5f),
        (unsigned char)(b * 255.f + 0.5f)
    );
}

/* ── ACO helpers ─────────────────────────────────────────────── */
static void to_gray(Tigr* src, std::vector<float>& gray) {
    int N = src->w * src->h;
    gray.resize(N);
    for (int i = 0; i < N; i++) {
        TPixel p = src->pix[i];
        float r = p.r / 255.f, g = p.g / 255.f, b = p.b / 255.f;
        gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

static void build_heuristic(const std::vector<float>& gray, int w, int h,
                             std::vector<float>& eta) {
    int N = w * h;
    eta.resize(N);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int im1 = clamp(i - 1, 0, h - 1), ip1 = clamp(i + 1, 0, h - 1);
            int jm1 = clamp(j - 1, 0, w - 1), jp1 = clamp(j + 1, 0, w - 1);
            float gx =
                (gray[im1 * w + jp1] + 2.f * gray[i * w + jp1] + gray[ip1 * w + jp1]) -
                (gray[im1 * w + jm1] + 2.f * gray[i * w + jm1] + gray[ip1 * w + jm1]);
            float gy =
                (gray[ip1 * w + jm1] + 2.f * gray[ip1 * w + j] + gray[ip1 * w + jp1]) -
                (gray[im1 * w + jm1] + 2.f * gray[im1 * w + j] + gray[im1 * w + jp1]);
            eta[i * w + j] = sqrtf(gx * gx + gy * gy);
        }
    }
    float mx = *std::max_element(eta.begin(), eta.end());
    if (mx > 0.f)
        for (int k = 0; k < N; k++)
            eta[k] /= mx;
}

static void build_spawn_cdf(const std::vector<float>& eta, std::vector<float>& cdf) {
    int N = (int)eta.size();
    cdf.resize(N);
    double sum = 0.0;
    for (int i = 0; i < N; i++) { sum += eta[i]; cdf[i] = (float)sum; }
    if (sum > 0.0)
        for (int i = 0; i < N; i++) cdf[i] /= (float)sum;
}

static int sample_from_cdf(const std::vector<float>& cdf, float u) {
    int idx = (int)(std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
    if (idx >= (int)cdf.size()) idx = (int)cdf.size() - 1;
    return idx;
}

/* ── main ────────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_image>\n", argv[0]);
        return 1;
    }

    const char* in_path = argv[1];
    Tigr* orig = tigrLoadImage(in_path);
    if (!orig) { fprintf(stderr, "Cannot load %s\n", in_path); return 1; }

    Tigr* src = resize_image(orig, 800);
    int w = src->w, h = src->h, N = w * h;
    printf("Loaded and constrained %s to [%dx%d]\n", in_path, w, h);

    std::vector<float> gray, eta, tau(N, TAU0);
    to_gray(src, gray);
    build_heuristic(gray, w, h, eta);

    std::vector<float> spawn_cdf;
    build_spawn_cdf(eta, spawn_cdf);

    int nthreads = omp_get_max_threads();
    printf("OpenMP threads: %d\n", nthreads);
    std::vector<std::vector<float>> delta(nthreads, std::vector<float>(N, 0.f));

    Tigr* screen = tigrWindow(w * 2, h, "ACO Edge Detection — Colour Gradient", TIGR_FIXED);

    /* Initial draw: left = grayscale, right = black */
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            unsigned char gv = (unsigned char)(gray[i * w + j] * 255.f);
            tigrPlot(screen, j,     i, tigrRGB(gv, gv, gv));
            tigrPlot(screen, j + w, i, tigrRGB(0,  0,  0));
        }
    }
    tigrUpdate(screen);
#ifdef _WIN32
    Sleep(500);
#else
    usleep(500000);
#endif

    /* ── ACO iterations ──────────────────────────────────────── */
    for (int iter = 0; iter < NUM_ITERATIONS && !tigrClosed(screen); iter++) {
        for (int t = 0; t < nthreads; t++)
            std::fill(delta[t].begin(), delta[t].end(), 0.f);

#pragma omp parallel for schedule(dynamic, 64)
        for (int ant = 0; ant < NUM_ANTS; ant++) {
            int tid = omp_get_thread_num();
            std::mt19937 rng((unsigned)(time(nullptr) ^ ((ant + 1) * 2654435761u) ^
                                        ((tid  + 1) * 2246822519u) ^
                                        ((unsigned)iter * 3266489917u)));
            std::uniform_real_distribution<float> udist(0.0f, 1.0f);

            int sidx = sample_from_cdf(spawn_cdf, udist(rng));
            int ci = sidx / w, cj = sidx % w;
            int last_ni = -1, last_nj = -1;

            bool alive = true;
            for (int step = 0; step < ANT_STEPS && alive; step++) {
                float probs[8], total = 0.f;
                int valid_moves = 0;

                for (int d = 0; d < 8; d++) {
                    int ni = ci + DX[d], nj = cj + DY[d];
                    if (ni < 0 || ni >= h || nj < 0 || nj >= w) { probs[d] = 0.f; continue; }
                    if (ni == last_ni && nj == last_nj)           { probs[d] = 0.f; continue; }
                    float p = powf(tau[ni * w + nj] + 1e-9f, ALPHA) *
                              powf(eta[ni * w + nj] + 1e-9f, BETA);
                    probs[d] = p;
                    total   += p;
                    valid_moves++;
                }

                int chosen = 0;
                if (total <= 1e-9f || valid_moves == 0) {
                    chosen = rng() % 8;
                } else {
                    float r = udist(rng) * total, cum = 0.f;
                    for (int d = 0; d < 8; d++) {
                        cum += probs[d];
                        if (r <= cum) { chosen = d; break; }
                    }
                }

                int ni   = clamp(ci + DX[chosen], 0, h - 1);
                int nj   = clamp(cj + DY[chosen], 0, w - 1);
                int nidx = ni * w + nj;

                if (eta[nidx] < MIN_ETA) {
                    alive = false;
                } else {
                    delta[tid][nidx] += Q * eta[nidx];
                }

                last_ni = ci; last_nj = cj;
                ci = ni;      cj = nj;
            }
        }

        /* pheromone evaporation + deposit + clamp */
        for (int k = 0; k < N; k++) {
            tau[k] *= (1.f - RHO);
            for (int t = 0; t < nthreads; t++) tau[k] += delta[t][k];
            if (tau[k] > TAU_MAX) tau[k] = TAU_MAX;
            if (tau[k] < TAU_MIN) tau[k] = TAU_MIN;
        }

        printf("Iter %2d/%d\n", iter + 1, NUM_ITERATIONS);

        /* ── Rendering: smooth color gradient, no hard threshold ── */
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int   k        = i * w + j;
                float norm_tau = tau[k] / TAU_MAX;   /* always in [TAU_MIN/TAU_MAX, 1] */

                /* Every pixel gets a color — strength encodes confidence */
                TPixel col = pheromone_to_color(norm_tau);
                tigrPlot(screen, j + w, i, col);
            }
        }
        tigrUpdate(screen);
    }

    /* Save result */
    std::string out_path = std::string(in_path) + "_colormap_edges.png";
    Tigr* out = tigrBitmap(w * 2, h);
    tigrBlit(out, screen, 0, 0, 0, 0, w * 2, h);
    tigrSaveImage(out_path.c_str(), out);
    printf("Saved %s\n", out_path.c_str());
    tigrFree(out);

    while (!tigrClosed(screen) && !tigrKeyDown(screen, TK_ESCAPE))
        tigrUpdate(screen);

    if (src != orig) tigrFree(src);
    tigrFree(orig);
    tigrFree(screen);
    return 0;
}