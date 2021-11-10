//
// Created by 69029 on 3/9/2021.
//

#ifndef ZKCNN_PROVER_HPP
#define ZKCNN_PROVER_HPP

#include "global_var.hpp"
#include "circuit.h"
#include "polynomial.h"

using std::unique_ptr;

class neuralNetwork;
class singleConv;
class prover {
public:
    void init();

    void sumcheckInitAll(const vector<F>::const_iterator &r_0_from_v);
    void sumcheckInit(const F &alpha_0, const F &beta_0);
    void sumcheckDotProdInitPhase1();
    void sumcheckInitPhase1(const F &relu_rou_0);
    void sumcheckInitPhase2();

    cubic_poly sumcheckDotProdUpdate1(const F &previous_random);
    quadratic_poly sumcheckUpdate1(const F &previous_random);
    quadratic_poly sumcheckUpdate2(const F &previous_random);

    F Vres(const vector<F>::const_iterator &r, u32 output_size, u8 r_size);

    void sumcheckDotProdFinalize1(const F &previous_random, F &claim_1);
    void sumcheckFinalize1(const F &previous_random, F &claim_0, F &claim_1);
    void sumcheckFinalize2(const F &previous_random, F &claim_0, F &claim_1);
    void sumcheckLiuFinalize(const F &previous_random, F &claim_1);

    void sumcheckLiuInit(const vector<F> &s_u, const vector<F> &s_v);
    quadratic_poly sumcheckLiuUpdate(const F &previous_random);

    hyrax_bls12_381::polyProver &commitInput(const vector<G> &gens);

    timer prove_timer;
    double proveTime() const { return prove_timer.elapse_sec(); }
    double proofSize() const { return (double) proof_size / 1024.0; }
    double polyProverTime() const { return poly_p -> getPT(); }
    double polyProofSize() const { return poly_p -> getPS(); }

    layeredCircuit C;
    vector<vector<F>> val;        // the output of each gate
private:
    quadratic_poly sumcheckUpdateEach(const F &previous_random, bool idx);
    quadratic_poly sumcheckUpdate(const F &previous_random, vector<F> &r_arr);
    F getCirValue(u8 layer_id, const vector<u32> &ori, u32 u);

    vector<F>::iterator r_0, r_1;         // current positions
    vector<vector<F>> r_u, r_v;             // next positions

    vector<F> beta_g;

    F add_term;
    vector<linear_poly> mult_array[2];
    vector<linear_poly> V_mult[2];

    F V_u0, V_u1;

    F alpha, beta, relu_rou;

    u64 proof_size;

    u32 total[2], total_size[2];
    u8 round;          // step within a sumcheck
    u8 sumcheck_id;    // the level

    unique_ptr<hyrax_bls12_381::polyProver> poly_p;

    friend neuralNetwork;
    friend singleConv;
};


#endif //ZKCNN_PROVER_HPP
