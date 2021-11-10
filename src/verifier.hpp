//
// Created by 69029 on 3/9/2021.
//

#ifndef ZKCNN_CONVVERIFIER_HPP
#define ZKCNN_CONVVERIFIER_HPP

#include "prover.hpp"

using std::unique_ptr;
class verifier {
public:
    prover *p;
    const layeredCircuit &C;

    verifier(prover *pr, const layeredCircuit &cir);

    bool verify();

    timer total_timer, total_slow_timer;
    double verifierTime() const { return total_timer.elapse_sec(); }
    double verifierSlowTime() const { return total_slow_timer.elapse_sec(); }

private:
    vector<vector<F>> r_u, r_v;
    vector<F> final_claim_u0, final_claim_v0;
    bool verifyInnerLayers();
    bool verifyFirstLayer();
    bool verifyInput();

    vector<F> beta_g;
    void betaInitPhase1(u8 depth, const F &alpha, const F &beta, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1, const F &relu_rou);
    void betaInitPhase2(u8 depth);

    F uni_value[2];
    F bin_value[3];
    void predicatePhase1(u8 layer_id);
    void predicatePhase2(u8 layer_id);

    F getFinalValue(const F &claim_u0, const F &claim_u1, const F &claim_v0, const F &claim_v1);

    F eval_in;
    unique_ptr<hyrax_bls12_381::polyVerifier> poly_v;
};


#endif //ZKCNN_CONVVERIFIER_HPP
