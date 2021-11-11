//
// Created by 69029 on 3/16/2021.
//

#ifndef ZKCNN_NEURALNETWORK_HPP
#define ZKCNN_NEURALNETWORK_HPP

#include <vector>
#include <fstream>
#include "circuit.h"
#include "prover.hpp"

using std::vector;
using std::tuple;
using std::pair;

enum convType {
    FFT, NAIVE, NAIVE_FAST
};

struct convKernel {
    convType ty;
    i64 channel_out, channel_in, size, stride_bl, padding, weight_start_id, bias_start_id;
    convKernel(convType _ty, i64 _channel_out, i64 _channel_in, i64 _size, i64 _log_stride, i64 _padding) :
            ty(_ty), channel_out(_channel_out), channel_in(_channel_in), size(_size), stride_bl(_log_stride), padding(_padding) {
    }

    convKernel(convType _ty, i64 _channel_out, i64 _channel_in, i64 _size) :
            convKernel(_ty, _channel_out, _channel_in, _size, 0, _size >> 1) {
    }
};

struct fconKernel {
    i64 channel_out, channel_in, weight_start_id, bias_start_id;
    fconKernel(i64 _channel_out, i64 _channel_in):
        channel_out(_channel_out), channel_in(_channel_in) {}
};

enum poolType {
    AVG, MAX, NONE
};

enum actType {
    RELU_ACT
};

struct poolKernel {
    poolType ty;
    i64 size, stride_bl, dcmp_start_id, max_start_id, max_dcmp_start_id;
    poolKernel(poolType _ty, i64 _size, i64 _log_stride):
            ty(_ty), size(_size), stride_bl(_log_stride) {}
};


class neuralNetwork {
public:
    explicit neuralNetwork(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, const string &i_filename,
                           const string &c_filename, const string &o_filename);

    neuralNetwork(i64 psize, i64 pchannel, i64 pparallel, i64 kernel_size, i64 sec_size, i64 fc_size,
                  i64 start_channel, poolType pool_ty);

    void create(prover &pr, bool only_compute);

protected:

    void initParam();

    int getNextBit(int layer_id);

    void refreshConvParam(i64 new_nx, i64 new_ny, const convKernel &conv);

    void calcSizeAfterPool(const poolKernel &p);

    void refreshFCParam(const fconKernel &fc);

    [[nodiscard]] i64 getFFTLen() const;

    [[nodiscard]] i8 getFFTBitLen() const;

    [[nodiscard]] i64 getPoolDecmpSize() const;

    void prepareDecmpBit(i64 layer_id, i64 idx, i64 dcmp_id, i64 bit_shift);

    void prepareFieldBit(const F &data, i64 dcmp_id, i64 bit_shift);

    void prepareSignBit(i64 layer_id, i64 idx, i64 dcmp_id);

    void prepareMax(i64 layer_id, i64 idx, i64 max_id);

    void calcInputLayer(layer &circuit);

    void calcNormalLayer(const layer &circuit, i64 layer_id);

    void calcDotProdLayer(const layer &circuit, i64 layer_id);

    void calcFFTLayer(const layer &circuit, i64 layer_id);

    vector<vector<convKernel>> conv_section;
    vector<poolKernel> pool;
    poolType pool_ty;
    i64 pool_bl, pool_sz;
    i64 pool_stride_bl, pool_stride;
    i64 pool_layer_cnt, act_layer_cnt, conv_layer_cnt;
    actType act_ty;

    vector<fconKernel> full_conn;

    i64 pic_size_x, pic_size_y, pic_channel, pic_parallel;
    i64 SIZE;
    const i64 NCONV_FAST_SIZE, NCONV_SIZE, FFT_SIZE, AVE_POOL_SIZE, FC_SIZE, RELU_SIZE;
    i64 T;
    const i64 Q = 9;
    i64 Q_MAX;
    const i64 Q_BIT_SIZE = 220;

    i64 nx_in, nx_out, ny_in, ny_out, m, channel_in, channel_out, log_stride, padding;
    i64 new_nx_in, new_ny_in;
    i64 nx_padded_in, ny_padded_in;
    i64 total_in_size, total_para_size, total_relu_in_size, total_ave_in_size, total_max_in_size;
    int x_bit, w_bit, x_next_bit;

    vector<vector<F>>::iterator val;
    vector<F>::iterator two_mul;

    void inputLayer(layer &circuit);

    void paddingLayer(layer &circuit, i64 &layer_id, i64 first_conv_id);

    void fftLayer(layer &circuit, i64 &layer_id);

    void dotProdLayer(layer &circuit, i64 &layer_id);

    void ifftLayer(layer &circuit, i64 &layer_id);

    void addBiasLayer(layer &circuit, i64 &layer_id, i64 first_bias_id);

    void naiveConvLayerFast(layer &circuit, i64 &layer_id, i64 first_conv_id, i64 first_bias_id);

    void naiveConvLayerMul(layer &circuit, i64 &layer_id, i64 first_conv_id);

    void naiveConvLayerAdd(layer &circuit, i64 &layer_id, i64 first_bias_id);

    void reluActConvLayer(layer &circuit, i64 &layer_id);

    void reluActFconLayer(layer &circuit, i64 &layer_id);

    void avgPoolingLayer(layer &circuit, i64 &layer_id);

    void
    maxPoolingLayer(layeredCircuit &C, i64 &layer_id, i64 first_dcmp_id, i64 first_max_id, i64 first_max_dcmp_id);

    void fullyConnLayer(layer &circuit, i64 &layer_id, i64 first_fc_id, i64 first_bias_id);

    static void printLayerInfo(const layer &circuit, i64 layer_id);

    void readBias(i64 first_bias_id);

    void readConvWeight(i64 first_conv_id);

    void readFconWeight(i64 first_fc_id);

    void printWitnessInfo(const layer &circuit) const;

    void printLayerValues(prover &pr);

    void printInfer(prover &pr);
};


#endif //ZKCNN_NEURALNETWORK_HPP
