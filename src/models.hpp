//
// Created by 69029 on 3/16/2021.
//

#ifndef ZKCNN_VGG_HPP
#define ZKCNN_VGG_HPP

#include "neuralNetwork.hpp"

class vgg: public neuralNetwork {

public:
    explicit vgg(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, const std::string &i_filename, const string &c_filename, const std::string &o_filename, const std::string &n_filename);

};

class vgg16: public neuralNetwork {

public:
    explicit vgg16(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, poolType pool_ty,
                   const std::string &i_filename,
                   const string &c_filename, const std::string &o_filename);

};

class vgg11: public neuralNetwork {

public:
    explicit vgg11(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, poolType pool_ty,
                   const std::string &i_filename,
                   const string &c_filename, const std::string &o_filename);

};

class lenet: public neuralNetwork {
public:
    explicit lenet(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, poolType pool_ty,
                   const std::string &i_filename,
                   const string &c_filename, const std::string &o_filename);
};

class lenetCifar: public neuralNetwork {
public:
    explicit lenetCifar(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, poolType pool_ty,
                        const std::string &i_filename, const string &c_filename, const std::string &o_filename);
};

class ccnn: public neuralNetwork {
public:
    explicit ccnn(i64 psize_x, i64 psize_y, i64 pparallel, i64 pchannel, poolType pool_ty);
};

class singleConv: public neuralNetwork {
public:
    explicit singleConv(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, i64 kernel_size, i64 channel_out,
                        i64 log_stride, i64 padding, convType conv_ty);

    void createConv(prover &p);

    void initParamConv();

    vector<F> getFFTAns(const vector<F> &output);

    double calcRawFFT();

    double calcRawNaive();
};

#endif //ZKCNN_VGG_HPP
