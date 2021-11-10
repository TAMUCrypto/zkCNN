//
// Created by 69029 on 3/9/2021.
//

#include <cmath>
#include <iostream>
#include <hyrax-bls12-381/polyCommit.hpp>
#include "utils.hpp"

using std::cerr;
using std::endl;
using std::string;
using std::cin;

int ceilPow2BitLengthSigned(double n) {
    return (i8) ceil(log2(n));
}

int floorPow2BitLengthSigned(double n) {
    return (i8) floor(log2(n));
}

i8 ceilPow2BitLength(u32 n) {
    return n < 1e-9 ? -1 : (i8) ceil(log(n) / log(2.));
}

i8 floorPow2BitLength(u32 n) {
//    cerr << n << ' ' << log(n) / log(2.)<<endl;
    return n < 1e-9 ? -1 : (i8) floor(log(n) / log(2.));
}

void initHalfTable(vector<F> &beta_f, vector<F> &beta_s, const vector<F>::const_iterator &r, const F &init, u32 first_half, u32 second_half) {
    beta_f.at(0) = init;
    beta_s.at(0) = F_ONE;

    for (u32 i = 0; i < first_half; ++i) {
        for (u32 j = 0; j < (1ULL << i); ++j) {
            auto tmp = beta_f.at(j) * r[i];
            beta_f.at(j | (1ULL << i)) = tmp;
            beta_f.at(j) = beta_f[j] - tmp;
        }
    }

    for (u32 i = 0; i < second_half; ++i) {
        for (u32 j = 0; j < (1ULL << i); ++j) {
            auto tmp = beta_s[j] * r[(i + first_half)];
            beta_s[j | (1ULL << i)] = tmp;
            beta_s[j] = beta_s[j] - tmp;
        }
    }
}

void phiPowInit(vector<F> &phi_mul, int n, bool isIFFT) {
    u32 N = 1ULL << n;
    F phi = getRootOfUnit(n);
    if (isIFFT) F::inv(phi, phi);
    phi_mul[0] = F_ONE;
    for (u32 i = 1; i < N; ++i) phi_mul[i] = phi_mul[i - 1] * phi;
}

void phiGInit(vector<F> &phi_g, const vector<F>::const_iterator &rx, const F &scale, int n, bool isIFFT) {
    vector<F> phi_mul(1 << n);
    phiPowInit(phi_mul, n, isIFFT);

    if (isIFFT) {
//        cerr << "==" << endl;
//        cerr << "gLength: " << n << endl;
//        for (int i = 0; i < n - 1; ++i) {
//            cerr << rx[i];
//            cerr << endl;
//        }
        phi_g[0] = phi_g[1] = scale;
        for (int i = 2; i <= n; ++i)
            for (u32 b = 0; b < (1ULL << (i - 1)); ++b) {
                u32 l = b, r = b ^ (1ULL << (i - 1));
                int m = n - i;
                F tmp1 = F_ONE - rx[m], tmp2 = rx[m] * phi_mul[b << m];
                phi_g[r] = phi_g[l] * (tmp1 - tmp2);
                phi_g[l] = phi_g[l] * (tmp1 + tmp2);
            }
    } else {
//        cerr << "==" << endl;
//        cerr << "gLength: " << n << endl;
//        for (int i = 0; i < n; ++i) {
//            cerr << rx[i];
//            cerr << endl;
//        }
        phi_g[0] = scale;
        for (int i = 1; i < n; ++i)
            for (u32 b = 0; b < (1ULL << (i - 1)); ++b) {
                u32 l = b, r = b ^ (1ULL << (i - 1));
                int m = n - i;
                F tmp1 = F_ONE - rx[m], tmp2 = rx[m] * phi_mul[b << m];
                phi_g[r] = phi_g[l] * (tmp1 - tmp2);
                phi_g[l] = phi_g[l] * (tmp1 + tmp2);
            }
        for (u32 b = 0; b < (1ULL << (n - 1)); ++b) {
            u32 l = b;
            F tmp1 = F_ONE - rx[0], tmp2 = rx[0] * phi_mul[b];
            phi_g[l] = phi_g[l] * (tmp1 + tmp2);
        }
    }
}

void fft(vector<F> &arr, int logn, bool flag) {
//    cerr << "fft: " << endl;
//    for (auto x: arr) cerr << x << ' ';
//    cerr << endl;
    static std::vector<u32> rev;
    static std::vector<F> w;

    u32 len = 1ULL << logn;
    assert(arr.size() == len);

    rev.resize(len);
    w.resize(len);

    rev[0] = 0;
    for (u32 i = 1; i < len; ++i)
        rev[i] = rev[i >> 1] >> 1 | (i & 1) << (logn - 1);

    w[0] = F_ONE;
    w[1] = getRootOfUnit(logn);
    if (flag) F::inv(w[1], w[1]);
    for (u32 i = 2; i < len; ++i) w[i] = w[i - 1] * w[1];

    for (u32 i = 0; i < len; ++i)
        if (rev[i] < i) std::swap(arr[i], arr[rev[i]]);

    for (u32 i = 2; i <= len; i <<= 1)
        for (u32 j = 0; j < len; j += i)
            for (u32 k = 0; k < (i >> 1); ++k) {
                auto u = arr[j + k];
                auto v = arr[j + k + (i >> 1)] * w[len / i * k];
                arr[j + k] = u + v;
                arr[j + k + (i >> 1)] = u - v;
            }

    if (flag) {
        F ilen;
        F::inv(ilen, len);
        for (u32 i = 0; i < len; ++i)
            arr[i] = arr[i] * ilen;
    }
}

void
initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1,
              const F &alpha, const F &beta) {
    u8 first_half = gLength >> 1, second_half = gLength - first_half;
    u32 mask_fhalf = (1ULL << first_half) - 1;

    vector<F> beta_f(1ULL << first_half), beta_s(1ULL << second_half);
    if (!beta.isZero()) {
        initHalfTable(beta_f, beta_s, r_1, beta, first_half, second_half);
        for (u32 i = 0; i < (1ULL << gLength); ++i)
            beta_g[i] = beta_f[i & mask_fhalf] * beta_s[i >> first_half];
    } else for (u32 i = 0; i < (1ULL << gLength); ++i)
        beta_g[i].clear();

    if (alpha.isZero()) return;
    initHalfTable(beta_f, beta_s, r_0, alpha, first_half, second_half);
    for (u32 i = 0; i < (1ULL << gLength); ++i)
        beta_g[i] = beta_g[i] + beta_f[i & mask_fhalf] * beta_s[i >> first_half];
}


void initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r, const F &init) {
    if (gLength == -1) return;
    int first_half = gLength >> 1, second_half = gLength - first_half;
    u32 mask_fhalf = (1ULL << first_half) - 1;
    vector<F> beta_f(1ULL << first_half), beta_s(1ULL << second_half);

    if (!init.isZero()) {
        initHalfTable(beta_f, beta_s, r, init, first_half, second_half);
        for (u32 i = 0; i < (1ULL << gLength); ++i)
            beta_g[i] = beta_f[i & mask_fhalf] * beta_s[i >> first_half];
    } else for (u32 i = 0; i < (1ULL << gLength); ++i)
        beta_g[i].clear();
}

bool check(long x, long y, long nx, long ny) {
    return 0 <= x && x < nx && 0 <= y && y < ny;
}
//
//F getData(u8 scale_bl) {
//    double x;
//    in >> x;
//    long y = round(x * (1L << scale_bl));
//    return F(y);
//}

void initLayer(layer &circuit, long size, layerType ty) {
    circuit.size = circuit.zero_start_id = size;
    circuit.bit_length = ceilPow2BitLength(size);
    circuit.ty = ty;
}

long sqr(long x) {
    return x * x;
}

double byte2KB(size_t x) { return x / 1024.0; }

double byte2MB(size_t x) { return x / 1024.0 / 1024.0; }

double byte2GB(size_t x) { return x / 1024.0 / 1024.0 / 1024.0; }

long matIdx(long x, long y, long n) {
    assert(y < n);
    return x * n + y;
}

long cubIdx(long x, long y, long z, long n, long m) {
    assert(y < n && z < m);
    return matIdx(matIdx(x, y, n), z, m);
}

long tesIdx(long w, long x, long y, long z, long n, long m, long l) {
    assert(x < n && y < m && z < l);
    return matIdx(cubIdx(w, x, y, n, m), z, l);
}

F getRootOfUnit(int n) {
    F res = -F_ONE;
    if (!n) return F_ONE;
    while (--n) {
        bool b = F::squareRoot(res, res);
        assert(b);
    }
    return res;
}


