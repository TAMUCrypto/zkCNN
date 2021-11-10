//
// Created by 69029 on 3/9/2021.
//

#include "prover.hpp"
#include <iostream>
#include <utils.hpp>

static vector<F> beta_gs, beta_u;

using std::unique_ptr;

linear_poly interpolate(const F &zero_v, const F &one_v) {
    return {one_v - zero_v, zero_v};
}

void prover::init() {
    proof_size = 0;
    r_u.resize(C.size + 1);
    r_v.resize(C.size + 1);
}

/**
 * This is to initialize all process.
 *
 * @param the random point to be evaluated at the output layer
 */
void prover::sumcheckInitAll(const vector<F>::const_iterator &r_0_from_v) {
    sumcheck_id = C.size;
    i8 last_bl = C.circuit[sumcheck_id - 1].bit_length;
    r_u[sumcheck_id].resize(last_bl);

    prove_timer.start();
    for (int i = 0; i < last_bl; ++i) r_u[sumcheck_id][i] = r_0_from_v[i];
    prove_timer.stop();
}

/**
 * This is to initialize before the process of a single layer.
 *
 * @param the random combination coefficiants for multiple reduction points
 */
void prover::sumcheckInit(const F &alpha_0, const F &beta_0) {
    prove_timer.start();
    auto &cur = C.circuit[sumcheck_id];
    alpha = alpha_0;
    beta = beta_0;
    r_0 = r_u[sumcheck_id].begin();
    r_1 = r_v[sumcheck_id].begin();
    --sumcheck_id;
    prove_timer.stop();
}

/**
 * This is to initialize before the phase 1 of a single inner production layer.
 */
void prover::sumcheckDotProdInitPhase1() {
    fprintf(stderr, "sumcheck level %d, phase1 init start\n", sumcheck_id);

    auto &cur = C.circuit[sumcheck_id];
    i8 fft_bl = cur.fft_bit_length;
    i8 cnt_bl = cur.bit_length - fft_bl;
    total[0] = 1ULL << fft_bl;
    total[1] = 1ULL << cur.bit_length_u[1];
    total_size[1] = cur.size_u[1];
    u32 fft_len = total[0];

    r_u[sumcheck_id].resize(cur.max_bl_u);
    V_mult[0].resize(total[1]);
    V_mult[1].resize(total[1]);
    mult_array[1].resize(total[0]);
    beta_gs.resize(1ULL << fft_bl);

    prove_timer.start();

    initBetaTable(beta_gs, fft_bl, r_0, F_ONE);

    for (u32 t = 0; t < fft_len; ++t)
        mult_array[1][t] = beta_gs[t];
    for (u32 u = 0; u < total[1]; ++u) {
        V_mult[0][u].clear();
        if (u >= cur.size_u[1]) V_mult[1][u].clear();
        else V_mult[1][u] = val[sumcheck_id - 1][u];
    }

    for (auto &gate: cur.bin_gates)
        for (u32 t = 0; t < fft_len; ++t) {
            u32 idx_u = gate.u << fft_bl | t;
            u32 idx_v = gate.v << fft_bl | t;
            V_mult[0][idx_u] = V_mult[0][idx_u] + beta_g[gate.g] * val[sumcheck_id - 1][idx_v];
        }

    round = 0;
    prove_timer.stop();
}

/**
 * This is the one-step reduction within a sumcheck process of a single inner production layer.
 *
 * @param the random point of the reduction of the previous step
 * @return the reducted cubic degree polynomial of the current variable from prover to verifier
 */
cubic_poly prover::sumcheckDotProdUpdate1(const F &previous_random) {
    prove_timer.start();

    if (round) r_u[sumcheck_id].at(round - 1) = previous_random;
    ++round;

    auto &tmp_mult = mult_array[1];
    auto &tmp_v0 = V_mult[0], &tmp_v1 = V_mult[1];

    if (total[0] == 1)
        tmp_mult[0] = tmp_mult[0].eval(previous_random);
    else for (u32 i = 0; i < (total[0] >> 1); ++i) {
        u32 g0 = i << 1, g1 = i << 1 | 1;
        tmp_mult[i] = interpolate(tmp_mult[g0].eval(previous_random), tmp_mult[g1].eval(previous_random));
    }
    total[0] >>= 1;

    cubic_poly ret;
    for (u32 i = 0; i < (total[1] >> 1); ++i) {
        u32 g0 = i << 1, g1 = i << 1 | 1;
        if (g0 >= total_size[1]) {
            tmp_v0[i].clear();
            tmp_v1[i].clear();
            continue;
        }
        if (g1 >= total_size[1]) {
            tmp_v0[g1].clear();
            tmp_v1[g1].clear();
        }
        tmp_v0[i] = interpolate(tmp_v0[g0].eval(previous_random), tmp_v0[g1].eval(previous_random));
        tmp_v1[i] = interpolate(tmp_v1[g0].eval(previous_random), tmp_v1[g1].eval(previous_random));
        if (total[0]) ret = ret + tmp_mult[i & total[0] - 1] * tmp_v1[i] * tmp_v0[i];
        else ret = ret + tmp_mult[0] * tmp_v1[i] * tmp_v0[i];
    }
    proof_size += F_BYTE_SIZE * (3 + (!ret.a.isZero()));

    total[1] >>= 1;
    total_size[1] = (total_size[1] + 1) >> 1;

    prove_timer.stop();
    return ret;
}

void prover::sumcheckDotProdFinalize1(const F &previous_random, F &claim_1) {
    prove_timer.start();
    r_u[sumcheck_id].at(round - 1) = previous_random;
    claim_1 = V_mult[1][0].eval(previous_random);
    V_u1 = V_mult[1][0].eval(previous_random) * mult_array[1][0].eval(previous_random);
    prove_timer.stop();
    proof_size += F_BYTE_SIZE * 1;
}

void prover::sumcheckInitPhase1(const F &relu_rou_0) {
    fprintf(stderr, "sumcheck level %d, phase1 init start\n", sumcheck_id);

    auto &cur = C.circuit[sumcheck_id];
    total[0] = ~cur.bit_length_u[0] ? 1ULL << cur.bit_length_u[0] : 0;
    total_size[0] = cur.size_u[0];
    total[1] = ~cur.bit_length_u[1] ? 1ULL << cur.bit_length_u[1] : 0;
    total_size[1] = cur.size_u[1];

    r_u[sumcheck_id].resize(cur.max_bl_u);
    V_mult[0].resize(total[0]);
    V_mult[1].resize(total[1]);
    mult_array[0].resize(total[0]);
    mult_array[1].resize(total[1]);
    beta_g.resize(1ULL << cur.bit_length);
    if (cur.ty == layerType::PADDING) beta_gs.resize(1ULL << cur.fft_bit_length);
    if (cur.ty == layerType::FFT || cur.ty == layerType::IFFT)
        beta_gs.resize(total[1]);

    prove_timer.start();

    relu_rou = relu_rou_0;
    add_term.clear();
    for (int b = 0; b < 2; ++b)
        for (u32 u = 0; u < total[b]; ++u)
            mult_array[b][u].clear();

    if (cur.ty == layerType::FFT || cur.ty == layerType::IFFT) {
        i8 fft_bl = cur.fft_bit_length;
        i8 fft_blh = cur.fft_bit_length - 1;
        i8 cnt_bl = cur.ty == layerType::FFT ? cur.bit_length - fft_bl : cur.bit_length - fft_blh;
        u32 cnt_len = cur.size >> (cur.ty == layerType::FFT ? fft_bl : fft_blh);
        if (cur.ty == layerType::FFT)
            initBetaTable(beta_g, cnt_bl, r_0 + fft_bl, r_1, alpha, beta);
        else initBetaTable(beta_g, cnt_bl, r_0 + fft_blh, alpha);
        for (u32 u = 0, l = sumcheck_id - 1; u < total[1]; ++u) {
            V_mult[1][u].clear();
            if (u >= cur.size_u[1]) continue;
            for (u32 g = 0; g < cnt_len; ++g) {
                u32 idx = g << cur.max_bl_u | u;
                V_mult[1][u] = V_mult[1][u] + val[l][idx] * beta_g[g];
            }
        }

        beta_gs.resize(total[1]);
        phiGInit(beta_gs, r_0, cur.scale, fft_bl, cur.ty == layerType::IFFT);
        for (u32 u = 0; u < total[1] ; ++u) {
            mult_array[1][u] = beta_gs[u];
        }
    } else {
        for (int b = 0; b < 2; ++b) {
            auto dep = !b ? 0 : sumcheck_id - 1;
            for (u32 u = 0; u < total[b]; ++u) {
                if (u >= cur.size_u[b])
                    V_mult[b][u].clear();
                else V_mult[b][u] = getCirValue(dep, cur.ori_id_u, u);
            }
        }

        if (cur.ty == layerType::PADDING) {
            i8 fft_blh = cur.fft_bit_length - 1;
            u32 fft_lenh = 1ULL << fft_blh;
            initBetaTable(beta_gs, fft_blh, r_0, F_ONE);
            for (long g = (1L << cur.bit_length) - 1; g >= 0; --g)
                beta_g[g] = beta_g[g >> fft_blh] * beta_gs[g & fft_lenh - 1];
        } else initBetaTable(beta_g, cur.bit_length, r_0, r_1, alpha * cur.scale, beta * cur.scale);
        if (cur.zero_start_id < cur.size)
            for (u32 g = cur.zero_start_id; g < 1ULL << cur.bit_length; ++g) beta_g[g] = beta_g[g] * relu_rou;

        for (auto &gate: cur.uni_gates) {
            bool idx = gate.lu != 0;
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + beta_g[gate.g] * C.two_mul[gate.sc];
        }

        for (auto &gate: cur.bin_gates) {
            bool idx = gate.getLayerIdU(sumcheck_id) != 0;
            auto val_lv = getCirValue(gate.getLayerIdV(sumcheck_id), cur.ori_id_v, gate.v);
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + val_lv * beta_g[gate.g] * C.two_mul[gate.sc];
        }
    }

    round = 0;
    prove_timer.stop();
    fprintf(stderr, "sumcheck level %d, phase1 init finished\n", sumcheck_id);
}

void prover::sumcheckInitPhase2() {
    fprintf(stderr, "sumcheck level %d, phase2 init start\n", sumcheck_id);

    auto &cur = C.circuit[sumcheck_id];
    total[0] = ~cur.bit_length_v[0] ? 1ULL << cur.bit_length_v[0] : 0;
    total_size[0] = cur.size_v[0];
    total[1] = ~cur.bit_length_v[1] ? 1ULL << cur.bit_length_v[1] : 0;
    total_size[1] = cur.size_v[1];
    i8 fft_bl = cur.fft_bit_length;
    i8 cnt_bl = cur.max_bl_v;

    r_v[sumcheck_id].resize(cur.max_bl_v);

    V_mult[0].resize(total[0]);
    V_mult[1].resize(total[1]);
    mult_array[0].resize(total[0]);
    mult_array[1].resize(total[1]);

    if (cur.ty == layerType::DOT_PROD) {
        beta_u.resize(1ULL << cnt_bl);
        beta_gs.resize(1ULL << fft_bl);
    } else beta_u.resize(1ULL << cur.max_bl_u);

    prove_timer.start();

    add_term.clear();
    for (int b = 0; b < 2; ++b) {
        for (u32 v = 0; v < total[b]; ++v)
            mult_array[b][v].clear();
    }

    if (cur.ty == layerType::DOT_PROD) {
        u32 fft_len = 1ULL << cur.fft_bit_length;
        initBetaTable(beta_u, cnt_bl, r_u[sumcheck_id].begin() + fft_bl, F_ONE);
        initBetaTable(beta_gs, fft_bl, r_u[sumcheck_id].begin(), F_ONE);

        for (u32 v = 0; v < total[1]; ++v) {
            V_mult[1][v].clear();
            if (v >= cur.size_v[1]) continue;
            for (u32 t = 0; t < fft_len; ++t) {
                u32 idx_v = (v << fft_bl) | t;
                V_mult[1][v] = V_mult[1][v] + val[sumcheck_id - 1][idx_v] * beta_gs[t];
            }
        }

        for (auto &gate: cur.bin_gates)
            mult_array[1][gate.v] =
                    mult_array[1][gate.v] + beta_g[gate.g] * beta_u[gate.u] * V_u1;
    } else {
        initBetaTable(beta_u, cur.max_bl_u, r_u[sumcheck_id].begin(), F_ONE);
        for (int b = 0; b < 2; ++b) {
            auto dep = !b ? 0 : sumcheck_id - 1;
            for (u32 v = 0; v < total[b]; ++v) {
                V_mult[b][v] = v >= cur.size_v[b] ? F_ZERO : getCirValue(dep, cur.ori_id_v, v);
            }
        }
        for (auto &gate: cur.uni_gates) {
            auto V_u = !gate.lu ? V_u0 : V_u1;
            add_term = add_term + beta_g[gate.g] * beta_u[gate.u] * V_u * C.two_mul[gate.sc];
        }
        for (auto &gate: cur.bin_gates) {
            bool idx = gate.getLayerIdV(sumcheck_id);
            auto V_u = !gate.getLayerIdU(sumcheck_id) ? V_u0 : V_u1;
            mult_array[idx][gate.v] = mult_array[idx][gate.v] + beta_g[gate.g] * beta_u[gate.u] * V_u * C.two_mul[gate.sc];
        }
    }

    round = 0;
    prove_timer.stop();
}

void prover::sumcheckLiuInit(const vector<F> &s_u, const vector<F> &s_v) {
    sumcheck_id = 0;
    total[1] = (1ULL << C.circuit[sumcheck_id].bit_length);
    total_size[1] = C.circuit[sumcheck_id].size;

    r_u[0].resize(C.circuit[0].bit_length);
    mult_array[1].resize(total[1]);
    V_mult[1].resize(total[1]);

    i8 max_bl = 0;
    for (int i = sumcheck_id + 1; i < C.size; ++i)
        max_bl = max(max_bl, max(C.circuit[i].bit_length_u[0], C.circuit[i].bit_length_v[0]));
    beta_g.resize(1ULL << max_bl);

    prove_timer.start();
    add_term.clear();

    for (u32 g = 0; g < total[1]; ++g) {
        mult_array[1][g].clear();
        V_mult[1][g] = (g < total_size[1]) ? val[0][g] : F_ZERO;
    }

    for (u8 i = sumcheck_id + 1; i < C.size; ++i) {
        i8 bit_length_i = C.circuit[i].bit_length_u[0];
        u32 size_i = C.circuit[i].size_u[0];
        if (~bit_length_i) {
            initBetaTable(beta_g, bit_length_i, r_u[i].begin(), s_u[i - 1]);
            for (u32 hu = 0; hu < size_i; ++hu) {
                u32 u = C.circuit[i].ori_id_u[hu];
                mult_array[1][u] = mult_array[1][u] + beta_g[hu];
            }
        }

        bit_length_i = C.circuit[i].bit_length_v[0];
        size_i = C.circuit[i].size_v[0];
        if (~bit_length_i) {
            initBetaTable(beta_g, bit_length_i, r_v[i].begin(), s_v[i - 1]);
            for (u32 hv = 0; hv < size_i; ++hv) {
                u32 v = C.circuit[i].ori_id_v[hv];
                mult_array[1][v] = mult_array[1][v] + beta_g[hv];
            }
        }
    }

    round = 0;
    prove_timer.stop();
}

quadratic_poly prover::sumcheckUpdate1(const F &previous_random) {
    return sumcheckUpdate(previous_random, r_u[sumcheck_id]);
}

quadratic_poly prover::sumcheckUpdate2(const F &previous_random) {
    return sumcheckUpdate(previous_random, r_v[sumcheck_id]);
}

quadratic_poly prover::sumcheckUpdate(const F &previous_random, vector<F> &r_arr) {
    prove_timer.start();

    if (round) r_arr.at(round - 1) = previous_random;
    ++round;
    quadratic_poly ret;

    add_term = add_term * (F_ONE - previous_random);
    for (int b = 0; b < 2; ++b)
        ret = ret + sumcheckUpdateEach(previous_random, b);
    ret = ret + quadratic_poly(F_ZERO, -add_term, add_term);

    prove_timer.stop();
    proof_size += F_BYTE_SIZE * 3;
    return ret;
}

quadratic_poly prover::sumcheckLiuUpdate(const F &previous_random) {
    prove_timer.start();
    ++round;

    auto ret = sumcheckUpdateEach(previous_random, true);

    prove_timer.stop();
    proof_size += F_BYTE_SIZE * 3;
    return ret;
}

quadratic_poly prover::sumcheckUpdateEach(const F &previous_random, bool idx) {
    auto &tmp_mult = mult_array[idx];
    auto &tmp_v = V_mult[idx];

    if (total[idx] == 1) {
        tmp_v[0] = tmp_v[0].eval(previous_random);
        tmp_mult[0] = tmp_mult[0].eval(previous_random);
        add_term = add_term + tmp_v[0].b * tmp_mult[0].b;
    }

    quadratic_poly ret;
    for (u32 i = 0; i < (total[idx] >> 1); ++i) {
        u32 g0 = i << 1, g1 = i << 1 | 1;
        if (g0 >= total_size[idx]) {
            tmp_v[i].clear();
            tmp_mult[i].clear();
            continue;
        }
        if (g1 >= total_size[idx]) {
            tmp_v[g1].clear();
            tmp_mult[g1].clear();
        }
        tmp_v[i] = interpolate(tmp_v[g0].eval(previous_random), tmp_v[g1].eval(previous_random));
        tmp_mult[i] = interpolate(tmp_mult[g0].eval(previous_random), tmp_mult[g1].eval(previous_random));
        ret = ret + tmp_mult[i] * tmp_v[i];
    }
    total[idx] >>= 1;
    total_size[idx] = (total_size[idx] + 1) >> 1;

    return ret;
}

/**
 * This is to evaluate a multi-linear extension at a random point.
 *
 * @param the value of the array & random point & the size of the array & the size of the random point
 * @return sum of `values`, or 0.0 if `values` is empty.
 */
F prover::Vres(const vector<F>::const_iterator &r, u32 output_size, u8 r_size) {
    prove_timer.start();

    vector<F> output(output_size);
    for (u32 i = 0; i < output_size; ++i)
        output[i] = val[C.size - 1][i];
    u32 whole = 1ULL << r_size;
    for (u8 i = 0; i < r_size; ++i) {
        for (u32 j = 0; j < (whole >> 1); ++j) {
            if (j > 0)
                output[j].clear();
            if ((j << 1) < output_size)
                output[j] = output[j << 1] * (F_ONE - r[i]);
            if ((j << 1 | 1) < output_size)
                output[j] = output[j] + output[j << 1 | 1] * (r[i]);
        }
        whole >>= 1;
    }
    F res = output[0];

    prove_timer.stop();
    proof_size += F_BYTE_SIZE;
    return res;
}

void prover::sumcheckFinalize1(const F &previous_random, F &claim_0, F &claim_1) {
    prove_timer.start();
    r_u[sumcheck_id].at(round - 1) = previous_random;
    V_u0 = claim_0 = total[0] ? V_mult[0][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_u[0]) ? V_mult[0][0].b : F_ZERO;
    V_u1 = claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_u[1]) ? V_mult[1][0].b : F_ZERO;
    prove_timer.stop();

    mult_array[0].clear();
    mult_array[1].clear();
    V_mult[0].clear();
    V_mult[1].clear();
    proof_size += F_BYTE_SIZE * 2;
}

void prover::sumcheckFinalize2(const F &previous_random, F &claim_0, F &claim_1) {
    prove_timer.start();
    r_v[sumcheck_id].at(round - 1) = previous_random;
    claim_0 = total[0] ? V_mult[0][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_v[0]) ? V_mult[0][0].b : F_ZERO;
    claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_v[1]) ? V_mult[1][0].b : F_ZERO;
    prove_timer.stop();

    mult_array[0].clear();
    mult_array[1].clear();
    V_mult[0].clear();
    V_mult[1].clear();
    proof_size += F_BYTE_SIZE * 2;
}

void prover::sumcheckLiuFinalize(const F &previous_random, F &claim_1) {
    prove_timer.start();
    r_u[sumcheck_id].at(round - 1) = previous_random;
    claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : V_mult[1][0].b;
    prove_timer.stop();
    proof_size += F_BYTE_SIZE;

    mult_array[1].clear();
    V_mult[1].clear();
    beta_g.clear();
}

F prover::getCirValue(u8 layer_id, const vector<u32> &ori, u32 u) {
    return !layer_id ? val[0][ori[u]] : val[layer_id][u];
}

hyrax_bls12_381::polyProver &prover::commitInput(const vector<G> &gens) {
    if (C.circuit[0].size != (1ULL << C.circuit[0].bit_length)) {
        val[0].resize(1ULL << C.circuit[0].bit_length);
        for (int i = C.circuit[0].size; i < val[0].size(); ++i)
            val[0][i].clear();
    }
    poly_p = std::make_unique<hyrax_bls12_381::polyProver>(val[0], gens);
    return *poly_p;
}