/*
 * ACO Edge Detection — using TIGR for I/O and display
 *
 * Usage:
 *   ./aco_edge <input_image>
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <omp.h>
#include <vector>
#include <string>
#include "tigr.h"

/* ── ACO parameters ──────────────────────────────────────────── */
static const int NUM_ANTS = 8000;
static const int NUM_ITERATIONS = 30;
static const float ALPHA = 1.8f;
static const float BETA = 1.2f;   // Increased to emphasize visual gradient heavily 
static const float RHO = 0.1f;
static const float Q = 1.0f;
static const float TAU_MIN = 0.01f;
static const float TAU_MAX = 9.0f; // Max-Min limit to prevent pheromone spikes
static const float TAU0 = 0.1f;
static const int ANT_STEPS = 50;
static const float EDGE_THRESHOLD = 0.35f; // Threshold is relative to TAU_MAX

static const int DX[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const int DY[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

static inline int clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ── Image scaling ───────────────────────────────────────────── */
static Tigr* resize_image(Tigr* src, int max_dim) {
    if (src->w <= max_dim && src->h <= max_dim) return src;
    float scale = (float)max_dim / std::max(src->w, src->h);
    int nw = (int)(src->w * scale);
    int nh = (int)(src->h * scale);
    Tigr* scaled = tigrBitmap(nw, nh);
    
    // Bilinear or nearest neighbour resize
    for (int y = 0; y < nh; y++) {
        for (int x = 0; x < nw; x++) {
            int sx = (int)(x / scale);
            int sy = (int)(y / scale);
            if (sx >= src->w) sx = src->w - 1;
            if (sy >= src->h) sy = src->h - 1;
            scaled->pix[y * nw + x] = src->pix[sy * src->w + sx];
        }
    }
    return scaled;
}

/* ── Color mapping ────────────────────────────────────────────── */
static TPixel get_edge_color(float intensity, float threshold) {
    float t = (intensity - threshold) / (1.0f - threshold);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    unsigned char r, g, b;

    if (t < 0.33f) {
        // Deep Blue → Purple
        float f = t / 0.33f;
        r = (unsigned char)(f * 120.f);
        g = 0;
        b = (unsigned char)(180.f + f * 75.f);
    }
    else if (t < 0.66f) {
        // Purple → Neon Pink
        float f = (t - 0.33f) / 0.33f;
        r = (unsigned char)(120.f + f * 135.f);
        g = (unsigned char)(f * 40.f);
        b = (unsigned char)(255.f - f * 80.f);
    }
    else {
        // Neon Pink → Cyan
        float f = (t - 0.66f) / 0.34f;
        r = (unsigned char)(255.f - f * 155.f);
        g = (unsigned char)(40.f + f * 215.f);
        b = (unsigned char)(175.f + f * 80.f);
    }

    return tigrRGB(r, g, b);
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

static void build_heuristic(const std::vector<float>& gray, int w, int h, std::vector<float>& eta) {
    int N = w * h;
    eta.resize(N);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int im1 = clamp(i - 1, 0, h - 1), ip1 = clamp(i + 1, 0, h - 1);
            int jm1 = clamp(j - 1, 0, w - 1), jp1 = clamp(j + 1, 0, w - 1);
            float gx = (gray[im1 * w + jp1] + 2 * gray[i * w + jp1] + gray[ip1 * w + jp1]) -
                       (gray[im1 * w + jm1] + 2 * gray[i * w + jm1] + gray[ip1 * w + jm1]);
            float gy = (gray[ip1 * w + jm1] + 2 * gray[ip1 * w + j] + gray[ip1 * w + jp1]) -
                       (gray[im1 * w + jm1] + 2 * gray[im1 * w + j] + gray[im1 * w + jp1]);
            eta[i * w + j] = sqrtf(gx * gx + gy * gy);
        }
    }
    float mx = *std::max_element(eta.begin(), eta.end());
    if (mx > 0)
        for (int k = 0; k < N; k++)
            eta[k] /= mx;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_image>\n", argv[0]);
        return 1;
    }

    const char* in_path = argv[1];
    Tigr* orig = tigrLoadImage(in_path);
    if (!orig) {
        fprintf(stderr, "Cannot load %s\n", in_path);
        return 1;
    }

    // Downscale huge images to a visually comfortable size
    // max dimension 800 allows reasonable calculation speeds and fits the window on standard screens.
    Tigr* src = resize_image(orig, 800);
    
    int w = src->w, h = src->h, N = w * h;
    printf("Loaded and constrained %s to [%dx%d]\n", in_path, w, h);

    std::vector<float> gray, eta, tau(N, TAU0);
    to_gray(src, gray);
    build_heuristic(gray, w, h, eta);

    int nthreads = omp_get_max_threads();
    printf("OpenMP threads: %d\n", nthreads);
    std::vector<std::vector<float>> delta(nthreads, std::vector<float>(N, 0.f));

    // A single unified window containing Side-by-side: original vs processed
    Tigr* screen = tigrWindow(w * 2, h, "ACO Edge Detection", TIGR_FIXED);

    // Initial draw
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            unsigned char gv = (unsigned char)(gray[i * w + j] * 255.f);
            tigrPlot(screen, j, i, tigrRGB(gv, gv, gv));
            tigrPlot(screen, j + w, i, tigrRGB(0, 0, 0));
        }
    }
    tigrUpdate(screen);

    /* ── ACO iterations ─────────────────────────────────────── */
    for (int iter = 0; iter < NUM_ITERATIONS && !tigrClosed(screen); iter++) {
        for (int t = 0; t < nthreads; t++)
            std::fill(delta[t].begin(), delta[t].end(), 0.f);

#pragma omp parallel for schedule(dynamic, 64)
        for (int ant = 0; ant < NUM_ANTS; ant++) {
            int tid = omp_get_thread_num();
            std::mt19937 rng((unsigned)(time(nullptr) ^ ((ant + 1) * 2654435761u) ^
                                       ((tid + 1) * 2246822519u) ^ (iter * 3266489917u)));
            std::uniform_int_distribution<int> hdist(0, h - 1);
            std::uniform_int_distribution<int> wdist(0, w - 1);
            std::uniform_real_distribution<float> udist(0.0f, 1.0f);
            
            int ci = hdist(rng), cj = wdist(rng);
            int last_ni = -1, last_nj = -1; // To prevent immediate back-tracking

            for (int step = 0; step < ANT_STEPS; step++) {
                float probs[8], total = 0.f;
                int valid_moves = 0;
                
                for (int d = 0; d < 8; d++) {
                    int ni = ci + DX[d], nj = cj + DY[d];
                    // Bounds check
                    if (ni < 0 || ni >= h || nj < 0 || nj >= w) {
                        probs[d] = 0.f;
                        continue;
                    }
                    // Prevent 180-degree backtracking
                    if (ni == last_ni && nj == last_nj) {
                        probs[d] = 0.f;
                        continue;
                    }
                    float p = powf(tau[ni * w + nj] + 1e-9f, ALPHA) *
                              powf(eta[ni * w + nj] + 1e-9f, BETA);
                    probs[d] = p;
                    total += p;
                    valid_moves++;
                }

                int chosen = 0;
                if (total <= 1e-9f || valid_moves == 0) {
                    // Wandering in completely flat area or stuck -> pick a random valid direction
                    chosen = rng() % 8;
                } else {
                    float r = udist(rng) * total;
                    float cum = 0.f;
                    for (int d = 0; d < 8; d++) {
                        cum += probs[d];
                        if (r <= cum) {
                            chosen = d;
                            break;
                        }
                    }
                }
                
                int ni = clamp(ci + DX[chosen], 0, h - 1);
                int nj = clamp(cj + DY[chosen], 0, w - 1);
                int nidx = ni * w + nj;
                
                // Deposit pheromones proportional to the gradient found!
                delta[tid][nidx] += Q * eta[nidx];
                
                last_ni = ci; 
                last_nj = cj;
                ci = ni;
                cj = nj;
            }
        }
        
        for (int k = 0; k < N; k++) {
            tau[k] *= (1.f - RHO);
            for (int t = 0; t < nthreads; t++) {
                tau[k] += delta[t][k];
            }
            // Clamp pheromone using Max-Min ant system logic
            if (tau[k] > TAU_MAX) tau[k] = TAU_MAX;
            if (tau[k] < TAU_MIN) tau[k] = TAU_MIN;
        }
        
        printf("Iter %2d/%d\n", iter + 1, NUM_ITERATIONS);

        // Display progress visually
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int k = i * w + j;
                // Threshold against fixed TAU_MAX instead of a single spiked pixel max
                float norm_tau = tau[k] / TAU_MAX;
                unsigned char gv = (unsigned char)(gray[k] * 255.f);
                
                if (norm_tau >= EDGE_THRESHOLD) {
                    // Fetch dynamic gradient color based on edge strength!
                    TPixel edge_color = get_edge_color(norm_tau, EDGE_THRESHOLD);
                    tigrFillCircle(screen, j + w, i, 1, edge_color);
                } else {
                    // Dimmed background
                    unsigned char bg = (unsigned char)(gv * 0.3f);
                    tigrPlot(screen, j + w, i, tigrRGB(bg, bg, bg));
                }
            }
        }
        tigrUpdate(screen);
    }

    std::string out_path = std::string(in_path) + "_edges.png";
    Tigr* out = tigrBitmap(w * 2, h);
    tigrBlit(out, screen, 0, 0, 0, 0, w * 2, h);
    tigrSaveImage(out_path.c_str(), out);
    printf("Saved %s\n", out_path.c_str());
    tigrFree(out);

    while (!tigrClosed(screen) && !tigrKeyDown(screen, TK_ESCAPE)) {
        tigrUpdate(screen);
    }
    
    // Clean up memory
    if (src != orig) {
        tigrFree(src);
    }
    tigrFree(orig);
    tigrFree(screen);

    return 0;
}
