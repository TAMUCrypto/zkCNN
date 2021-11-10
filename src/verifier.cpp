//
// Created by 69029 on 3/9/2021.
//

#include "verifier.hpp"
#include "global_var.hpp"
#include <utils.hpp>
#include <circuit.h>
#include <iostream>

vector<F> beta_v;
static vector<F> beta_u, beta_gs;

verifier::verifier(prover *pr, const layeredCircuit &cir):
    p(pr), C(cir) {
    final_claim_u0.resize(C.size + 2);
    final_claim_v0.resize(C.size + 2);

    r_u.resize(C.size + 2);
    r_v.resize(C.size + 2);
    // make the prover ready
    p->init();
}

F verifier::getFinalValue(const F &claim_u0, const F &claim_u1, const F &claim_v0, const F &claim_v1) {

    auto test_value = bin_value[0] * (claim_u0 * claim_v0)
                      + bin_value[1] * (claim_u1 * claim_v1)
                      + bin_value[2] * (claim_u1 * claim_v0)
                      + uni_value[0] * claim_u0
                      + uni_value[1] * claim_u1;

    return test_value;
}

void verifier::betaInitPhase1(u8 depth, const F &alpha, const F &beta, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1, const F &relu_rou) {
    i8 bl = C.circuit[depth].bit_length;
    i8 fft_bl = C.circuit[depth].fft_bit_length;
    i8 fft_blh = C.circuit[depth].fft_bit_length - 1;
    i8 cnt_bl = bl - fft_bl, cnt_bl2 = C.circuit[depth].max_bl_u - fft_bl;

    switch (C.circuit[depth].ty) {
        case layerType::FFT:
        case layerType::IFFT:
            beta_gs.resize(1ULL << fft_bl);
            phiGInit(beta_gs, r_0, C.circuit[depth].scale, fft_bl, C.circuit[depth].ty == layerType::IFFT);
            beta_u.resize(1ULL << C.circuit[depth].max_bl_u);
            initBetaTable(beta_u, C.circuit[depth].max_bl_u, r_u[depth].begin(), F_ONE);
            break;
        case layerType::PADDING:
            beta_g.resize(1ULL << bl);
            beta_gs.resize(1ULL << fft_blh);
            initBetaTable(beta_g, bl - fft_blh, r_u[depth + 2].begin() + fft_bl, r_v[depth + 2].begin(), alpha, beta);
            initBetaTable(beta_gs, fft_blh, r_0, F_ONE);
            for (u32 g = (1ULL << bl) - 1; g < (1ULL << bl); --g)
                beta_g[g] = beta_g[g >> fft_blh] *
                            beta_gs[g & (1ULL << fft_blh) - 1];
            beta_u.resize(1ULL << C.circuit[depth].max_bl_u);
            initBetaTable(beta_u, C.circuit[depth].max_bl_u, r_u[depth].begin(), F_ONE);
            break;
        case layerType::DOT_PROD:
            beta_g.resize(1ULL << cnt_bl);
            initBetaTable(beta_g, cnt_bl, r_u[depth + 2].begin() + fft_bl - 1, alpha);

            beta_u.resize(1ULL << cnt_bl2);
            initBetaTable(beta_u, cnt_bl2, r_u[depth].begin() + fft_bl, F_ONE);
            for (u32 i = 0; i < 1ULL << cnt_bl2; ++i)
                for (u32 j = 0; j < fft_bl; ++j)
                    beta_u[i] = beta_u[i] * ((r_0[j] * r_u[depth][j]) + (F_ONE - r_0[j]) * (F_ONE - r_u[depth][j]));
            break;

        default:
            beta_g.resize(1ULL << bl);
            initBetaTable(beta_g, C.circuit[depth].bit_length, r_0, r_1, alpha * C.circuit[depth].scale,
                          beta * C.circuit[depth].scale);
            if (C.circuit[depth].zero_start_id < C.circuit[depth].size)
                for (u32 g = C.circuit[depth].zero_start_id; g < 1ULL << C.circuit[depth].bit_length; ++g)
                    beta_g[g] = beta_g[g] * relu_rou;
            beta_u.resize(1ULL << C.circuit[depth].max_bl_u);
            initBetaTable(beta_u, C.circuit[depth].max_bl_u, r_u[depth].begin(), F_ONE);
    }
}

void verifier::betaInitPhase2(u8 depth) {
    beta_v.resize(1ULL << C.circuit[depth].max_bl_v);
    initBetaTable(beta_v, C.circuit[depth].max_bl_v, r_v[depth].begin(), F_ONE);
}

void verifier::predicatePhase1(u8 layer_id) {
    auto &cur_layer = C.circuit[layer_id];

    uni_value[0].clear();
    uni_value[1].clear();
    if (cur_layer.ty == layerType::FFT || cur_layer.ty == layerType::IFFT)
        for (u32 u = 0; u < 1ULL << cur_layer.max_bl_u; ++u)
            uni_value[1] = uni_value[1] + beta_gs[u] * beta_u[u];
    else for (auto &gate: cur_layer.uni_gates) {
            bool idx = gate.lu;
            uni_value[idx] = uni_value[idx] + beta_g[gate.g] * beta_u[gate.u] * C.two_mul[gate.sc];
        }
    bin_value[0] = bin_value[1] = bin_value[2] = F_ZERO;
}

void verifier::predicatePhase2(u8 layer_id) {
    uni_value[0] = uni_value[0] * beta_v[0];
    uni_value[1] = uni_value[1] * beta_v[0];

    auto &cur_layer = C.circuit[layer_id];
    if (C.circuit[layer_id].ty == layerType::DOT_PROD) {
        for (auto &gate: cur_layer.bin_gates)
            bin_value[gate.l] =
                    bin_value[gate.l] +
                    beta_g[gate.g] * beta_u[gate.u] * beta_v[gate.v];
    } else for (auto &gate: cur_layer.bin_gates)
        bin_value[gate.l] = bin_value[gate.l] + beta_g[gate.g] * beta_u[gate.u] * beta_v[gate.v] * C.two_mul[gate.sc];
}

bool verifier::verify() {
    u8 logn = C.circuit[0].bit_length;
    u64 n_sqrt = 1ULL << (logn - (logn >> 1));
    vector<G> gens(n_sqrt);
    for (auto &x: gens) {
        Fr tmp;
        tmp.setByCSPRNG();
        x = mcl::bn::getG1basePoint() * tmp;
    }

    poly_v = std::make_unique<hyrax_bls12_381::polyVerifier>(p -> commitInput(gens), gens);
    return verifyInnerLayers() && verifyFirstLayer() && verifyInput();
}

bool verifier::verifyInnerLayers() {
    total_timer.start();
    total_slow_timer.start();

    F alpha = F_ONE, beta = F_ZERO, relu_rou, final_claim_u1, final_claim_v1;
    r_u[C.size].resize(C.circuit[C.size - 1].bit_length);
    for (i8 i = 0; i < C.circuit[C.size - 1].bit_length; ++i)
        r_u[C.size][i].setByCSPRNG();
    vector<F>::const_iterator r_0 = r_u[C.size].begin();
    vector<F>::const_iterator r_1;

    total_timer.stop();
    total_slow_timer.stop();

    auto previousSum = p->Vres(r_0, C.circuit[C.size - 1].size, C.circuit[C.size - 1].bit_length);
    p -> sumcheckInitAll(r_0);

    for (u8 i = C.size - 1; i; --i) {
        auto &cur = C.circuit[i];
        p->sumcheckInit(alpha, beta);
        total_timer.start();
        total_slow_timer.start();

        // phase 1
        r_u[i].resize(cur.max_bl_u);
        for (int j = 0; j < cur.max_bl_u; ++j) r_u[i][j].setByCSPRNG();
        if (cur.zero_start_id < cur.size)
            relu_rou.setByCSPRNG();
        else relu_rou = F_ONE;

        total_timer.stop();
        total_slow_timer.stop();
        if (cur.ty == layerType::DOT_PROD)
            p->sumcheckDotProdInitPhase1();
        else p->sumcheckInitPhase1(relu_rou);

        F previousRandom = F_ZERO;
        for (i8 j = 0; j < cur.max_bl_u; ++j) {
            F cur_claim, nxt_claim;
            if (cur.ty == layerType::DOT_PROD) {
                cubic_poly poly = p->sumcheckDotProdUpdate1(previousRandom);
                total_timer.start();
                total_slow_timer.start();
                cur_claim = poly.eval(F_ZERO) + poly.eval(F_ONE);
                nxt_claim = poly.eval(r_u[i][j]);
            } else {
                quadratic_poly poly = p->sumcheckUpdate1(previousRandom);
                total_timer.start();
                total_slow_timer.start();
                cur_claim = poly.eval(F_ZERO) + poly.eval(F_ONE);
                nxt_claim = poly.eval(r_u[i][j]);
            }

            if (cur_claim != previousSum) {
                cerr << cur_claim << ' ' << previousSum << endl;
                fprintf(stderr, "Verification fail, phase1, circuit %d, current bit %d\n", i, j);
                return false;
            }
            previousRandom = r_u[i][j];
            previousSum = nxt_claim;
            total_timer.stop();
            total_slow_timer.stop();
        }

        if (cur.ty == layerType::DOT_PROD)
            p->sumcheckDotProdFinalize1(previousRandom, final_claim_u1);
        else p->sumcheckFinalize1(previousRandom, final_claim_u0[i], final_claim_u1);

        total_slow_timer.start();
        betaInitPhase1(i, alpha, beta, r_0, r_1, relu_rou);
        predicatePhase1(i);

        total_timer.start();
        if (cur.need_phase2) {
            r_v[i].resize(cur.max_bl_v);
            for (int j = 0; j < cur.max_bl_v; ++j) r_v[i][j].setByCSPRNG();

            total_timer.stop();
            total_slow_timer.stop();

            p->sumcheckInitPhase2();
            previousRandom = F_ZERO;
            for (u32 j = 0; j < cur.max_bl_v; ++j) {
                quadratic_poly poly = p->sumcheckUpdate2(previousRandom);

                total_timer.start();
                total_slow_timer.start();
                if (poly.eval(F_ZERO) + poly.eval(F_ONE) != previousSum) {
                    fprintf(stderr, "Verification fail, phase2, circuit level %d, current bit %d, total is %d\n", i, j,
                            cur.max_bl_v);
                    return false;
                }

                previousRandom = r_v[i][j];
                previousSum = poly.eval(previousRandom);
                total_timer.stop();
                total_slow_timer.stop();
            }
            p->sumcheckFinalize2(previousRandom, final_claim_v0[i], final_claim_v1);

            total_slow_timer.start();
            betaInitPhase2(i);
            predicatePhase2(i);
            total_timer.start();
        }
        F test_value = getFinalValue(final_claim_u0[i], final_claim_u1, final_claim_v0[i], final_claim_v1);

        if (previousSum != test_value) {
            std::cerr << test_value << ' ' << previousSum << std::endl;
            fprintf(stderr, "Verification fail, semi final, circuit level %d\n", i);
            return false;
        } else fprintf(stderr, "Verification Pass, semi final, circuit level %d\n", i);

        if (cur.ty == layerType::FFT || cur.ty == layerType::IFFT)
            previousSum = final_claim_u1;
        else {
            if (~cur.bit_length_u[1])
                alpha.setByCSPRNG();
            else alpha.clear();
            if ((~cur.bit_length_v[1]) || cur.ty == layerType::FFT)
                beta.setByCSPRNG();
            else beta.clear();
            previousSum = alpha * final_claim_u1 + beta * final_claim_v1;
        }

        r_0 = r_u[i].begin();
        r_1 = r_v[i].begin();

        total_timer.stop();
        total_slow_timer.stop();
        beta_u.clear();
        beta_v.clear();
    }
    return true;
}

bool verifier::verifyFirstLayer() {
    total_slow_timer.start();
    total_timer.start();

    auto &cur = C.circuit[0];

    vector<F> sig_u(C.size - 1);
    for (int i = 0; i < C.size - 1; ++i) sig_u[i].setByCSPRNG();
    vector<F> sig_v(C.size - 1);
    for (int i = 0; i < C.size - 1; ++i) sig_v[i].setByCSPRNG();
    r_u[0].resize(cur.bit_length);
    for (int i = 0; i < cur.bit_length; ++i) r_u[0][i].setByCSPRNG();
    auto r_0 = r_u[0].begin();

    F previousSum = F_ZERO;
    for (int i = 1; i < C.size; ++i) {
        if (~C.circuit[i].bit_length_u[0])
            previousSum = previousSum + sig_u[i - 1] * final_claim_u0[i];
        if (~C.circuit[i].bit_length_v[0])
            previousSum = previousSum + sig_v[i - 1] * final_claim_v0[i];
    }
    total_timer.stop();
    total_slow_timer.stop();

    p->sumcheckLiuInit(sig_u, sig_v);
    F previousRandom = F_ZERO;
    for (int j = 0; j < cur.bit_length; ++j) {
        auto poly = p -> sumcheckLiuUpdate(previousRandom);
        if (poly.eval(F_ZERO) + poly.eval(F_ONE) != previousSum) {
            fprintf(stderr, "Liu fail, circuit 0, current bit %d\n", j);
            return false;
        }
        previousRandom = r_0[j];
        previousSum = poly.eval(previousRandom);
    }

    F gr = F_ZERO;
    p->sumcheckLiuFinalize(previousRandom, eval_in);

    beta_g.resize(1ULL << cur.bit_length);

    total_slow_timer.start();
    initBetaTable(beta_g, cur.bit_length, r_0, F_ONE);
    for (int i = 1; i < C.size; ++i) {
        if (~C.circuit[i].bit_length_u[0]) {
            beta_u.resize(1ULL << C.circuit[i].bit_length_u[0]);
            initBetaTable(beta_u, C.circuit[i].bit_length_u[0], r_u[i].begin(), sig_u[i - 1]);
            for (u32 j = 0; j < C.circuit[i].size_u[0]; ++j)
                gr = gr + beta_g[C.circuit[i].ori_id_u[j]] * beta_u[j];
        }

        if (~C.circuit[i].bit_length_v[0]) {
            beta_v.resize(1ULL << C.circuit[i].bit_length_v[0]);
            initBetaTable(beta_v, C.circuit[i].bit_length_v[0], r_v[i].begin(), sig_v[i - 1]);
            for (u32 j = 0; j < C.circuit[i].size_v[0]; ++j)
                gr = gr + beta_g[C.circuit[i].ori_id_v[j]] * beta_v[j];
        }
    }

    beta_u.clear();
    beta_v.clear();

    total_timer.start();
    if (eval_in * gr != previousSum) {
        fprintf(stderr, "Liu fail, semi final, circuit 0.\n");
        return false;
    }

    total_timer.stop();
    total_slow_timer.stop();
    output_tb[PT_OUT_ID] = to_string_wp(p->proveTime());
    output_tb[VT_OUT_ID] = to_string_wp(verifierTime());
    output_tb[PS_OUT_ID] = to_string_wp(p -> proofSize());

    fprintf(stderr, "Verification pass\n");
    fprintf(stderr, "Prove Time %lf\n", p->proveTime());
    fprintf(stderr, "verify time %lf = %lf + %lf(slow)\n", verifierSlowTime(), verifierTime(), verifierSlowTime() - verifierTime());
    fprintf(stderr, "proof size = %lf kb\n", p -> proofSize());

    beta_g.clear();
    beta_gs.clear();
    beta_u.clear();
    beta_v.clear();
    r_u.resize(1);
    r_v.clear();

    sig_u.clear();
    sig_v.clear();
    return true;
}

bool verifier::verifyInput() {
    if (!poly_v -> verify(r_u[0], eval_in)) {
        fprintf(stderr, "Verification fail, final input check fail.\n");
        return false;
    }

    fprintf(stderr, "poly pt = %.5f, vt = %.5f, ps = %.5f\n", p -> polyProverTime(), poly_v -> getVT(), p -> polyProofSize());
    output_tb[POLY_PT_OUT_ID] = to_string_wp(p -> polyProverTime());
    output_tb[POLY_VT_OUT_ID] = to_string_wp(poly_v -> getVT());
    output_tb[POLY_PS_OUT_ID] = to_string_wp(p -> polyProofSize());
    output_tb[TOT_PT_OUT_ID] = to_string_wp(p -> polyProverTime() + p->proveTime());
    output_tb[TOT_VT_OUT_ID] = to_string_wp(poly_v -> getVT() + verifierTime());
    output_tb[TOT_PS_OUT_ID] = to_string_wp(p -> polyProofSize() + p -> proofSize());
    return true;
}
