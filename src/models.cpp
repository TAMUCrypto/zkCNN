//
// Created by 69029 on 3/16/2021.
//

#include <tuple>
#include <iostream>
#include "models.hpp"
#include "utils.hpp"
#undef USE_VIRGO


vgg::vgg(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, const string &i_filename,
         const string &c_filename, const std::string &o_filename, const std::string &n_filename):
        neuralNetwork(psize_x, psize_y, pchannel, pparallel, i_filename, c_filename, o_filename)  {
    assert(psize_x == psize_y);
    conv_section.resize(5);

    ifstream config_in(n_filename);
    string con;
    i64 kernel_size = 3, ch_in = pic_channel, ch_out, new_nx = pic_size_x, new_ny = pic_size_y;
    convType conv_ty = kernel_size > 3 || pparallel > 1 ? FFT : NAIVE_FAST;

    int idx = 0;
    while (config_in >> con) {
        if (con[0] != 'M' && con[0] != 'A') {
            ch_out = stoi(con, nullptr, 10);
            conv_section[idx].emplace_back(conv_ty, ch_out, ch_in, kernel_size);
            ch_in = ch_out;
        } else {
            ++idx;
            pool.emplace_back(con[0] == 'M' ? MAX : AVG, 2, 1);
            new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
            new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;
        }
    }

    assert(pic_size_x == 32);
    full_conn.emplace_back(512, new_nx * new_ny * ch_in);
    full_conn.emplace_back(512, 512);
    full_conn.emplace_back(10, 512);
}

vgg16::vgg16(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, poolType pool_ty, const std::string &i_filename,
             const string &c_filename, const std::string &o_filename)
        : neuralNetwork(psize_x, psize_y, pchannel, pparallel, i_filename, c_filename, o_filename) {
    assert(psize_x == psize_y);
    conv_section.resize(5);

    i64 start = 64, kernel_size = 3, new_nx = pic_size_x, new_ny = pic_size_y;
    convType conv_ty = kernel_size > 3 || pparallel > 1 ? FFT : NAIVE_FAST;

    conv_section[0].emplace_back(conv_ty, start,  pic_channel, kernel_size);
    conv_section[0].emplace_back(conv_ty, start, start, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    conv_section[1].emplace_back(conv_ty, start << 1,  start, kernel_size);
    conv_section[1].emplace_back(conv_ty, start << 1, start << 1, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    conv_section[2].emplace_back(conv_ty, start << 2, start << 1, kernel_size);
    conv_section[2].emplace_back(conv_ty, start << 2, start << 2, kernel_size);
    conv_section[2].emplace_back(conv_ty, start << 2, start << 2, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    conv_section[3].emplace_back(conv_ty, start << 3, start << 2, 3);
    conv_section[3].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[3].emplace_back(conv_ty, start << 3, start << 3, 3);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);

    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    if (pic_size_x == 224) {
        full_conn.emplace_back(4096, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(4096, 4096);
        full_conn.emplace_back(1000, 4096);
    } else {
        assert(pic_size_x == 32);
        full_conn.emplace_back(512, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(512, 512);
        full_conn.emplace_back(10, 512);
    }
}

vgg11::vgg11(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, poolType pool_ty, const std::string &i_filename,
             const string &c_filename, const std::string &o_filename)
        : neuralNetwork(psize_x, psize_y, pchannel, pparallel, i_filename, c_filename, o_filename) {
    assert(psize_x == psize_y);
    conv_section.resize(5);

    i64 start = 64, kernel_size = 3, new_nx = pic_size_x, new_ny = pic_size_y;
    convType conv_ty = kernel_size > 3 || pparallel > 1 ? FFT : NAIVE_FAST;

    conv_section[0].emplace_back(conv_ty, start,  pic_channel, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    conv_section[1].emplace_back(conv_ty, start << 1,  start, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    conv_section[2].emplace_back(conv_ty, start << 2, start << 1, kernel_size);
    conv_section[2].emplace_back(conv_ty, start << 2, start << 2, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    conv_section[3].emplace_back(conv_ty, start << 3, start << 2, 3);
    conv_section[3].emplace_back(conv_ty, start << 3, start << 3, 3);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);

    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    if (pic_size_x == 224) {
        full_conn.emplace_back(4096, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(4096, 4096);
        full_conn.emplace_back(1000, 4096);
    } else {
        assert(pic_size_x == 32);
        full_conn.emplace_back(512, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(512, 512);
        full_conn.emplace_back(10, 512);
    }
}

ccnn::ccnn(i64 psize_x, i64 psize_y, i64 pparallel, i64 pchannel, poolType pool_ty) :
    neuralNetwork(psize_x, psize_y, pchannel, pparallel, "", "", "") {
    conv_section.resize(1);

    i64 kernel_size = 2;
    convType conv_ty = kernel_size > 3 || pparallel > 1 ? FFT : NAIVE_FAST;
    conv_section[0].emplace_back(conv_ty, 2,  pchannel, kernel_size, 0, 0);
    pool.emplace_back(pool_ty, 2, 1);

//    conv_section[1].emplace_back(FFT, 64, 4, 3);
//    conv_section[1].emplace_back(NAIVE, 64,  64, 3);
//    pool.emplace_back(pool_ty, 2, 1);

//    conv_section[0].emplace_back(FFT, 2, pic_channel, 3);
//    conv_section[1].emplace_back(NAIVE, 1,  2, 3);
//    pool.emplace_back(pool_ty, 2, 1);
}

lenet::lenet(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, poolType pool_ty, const std::string &i_filename,
             const string &c_filename, const std::string &o_filename)
        : neuralNetwork(psize_x, psize_y, pchannel, pparallel, i_filename, c_filename, o_filename) {
    conv_section.emplace_back();

    i64 kernel_size = 5;
    convType conv_ty = kernel_size > 3 || pparallel > 1 ? FFT : NAIVE_FAST;

    if (psize_x == 28 && psize_y == 28)
        conv_section[0].emplace_back(conv_ty, 6,  pchannel, kernel_size, 0, 2);
    else conv_section[0].emplace_back(conv_ty, 6,  pchannel, kernel_size, 0, 0);
    pool.emplace_back(pool_ty, 2, 1);

    conv_section.emplace_back();
    conv_section[1].emplace_back(conv_ty, 16, 6, kernel_size, 0, 0);
    pool.emplace_back(pool_ty, 2, 1);

    full_conn.emplace_back(120, 400);
    full_conn.emplace_back(84, 120);
    full_conn.emplace_back(10, 84);
}

lenetCifar::lenetCifar(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, poolType pool_ty,
                       const std::string &i_filename, const string &c_filename, const std::string &o_filename)
        : neuralNetwork(psize_x, psize_y, pchannel, pparallel, i_filename, c_filename, o_filename) {
    conv_section.resize(3);

    i64 kernel_size = 5;
    convType conv_ty = kernel_size > 3 || pparallel > 1 ? FFT : NAIVE_FAST;

    conv_section[0].emplace_back(conv_ty, 6, pchannel, kernel_size, 0, 0);
    pool.emplace_back(pool_ty, 2, 1);

    conv_section[1].emplace_back(conv_ty, 16, 6, kernel_size, 0, 0);
    pool.emplace_back(pool_ty, 2, 1);

    conv_section[2].emplace_back(conv_ty, 120, 16, kernel_size, 0, 0);

    full_conn.emplace_back(84, 120);
    full_conn.emplace_back(10, 84);
}

void singleConv::createConv(prover &p) {
    initParamConv();
    p.C.init(Q_BIT_SIZE, SIZE);

    p.val.resize(SIZE);
    val = p.val.begin();
    two_mul = p.C.two_mul.begin();

    i64 layer_id = 0;
    inputLayer(p.C.circuit[layer_id++]);

    new_nx_in = pic_size_x;
    new_ny_in = pic_size_y;
    pool_ty = NONE;
    for (i64 i = 0; i < conv_section.size(); ++i) {
        auto &sec = conv_section[i];
        for (i64 j = 0; j < sec.size(); ++j) {
            auto &conv = sec[j];
            refreshConvParam(new_nx_in, new_ny_in, conv);

            switch (conv.ty) {
                case FFT:
                    paddingLayer(p.C.circuit[layer_id], layer_id, conv.weight_start_id);
                    fftLayer(p.C.circuit[layer_id], layer_id);
                    dotProdLayer(p.C.circuit[layer_id], layer_id);
                    ifftLayer(p.C.circuit[layer_id], layer_id);
                    break;
                case NAIVE_FAST:
                    naiveConvLayerFast(p.C.circuit[layer_id], layer_id, conv.weight_start_id, conv.bias_start_id);
                    break;
                default:
                    naiveConvLayerMul(p.C.circuit[layer_id], layer_id, conv.weight_start_id);
                    naiveConvLayerAdd(p.C.circuit[layer_id], layer_id, conv.bias_start_id);
            }
        }
    }
    p.C.initSubset();
//    for (i64 i = 0; i < SIZE; ++i) {
//        cerr << i << "(" << p.C.circuit[i].zero_start_id << ", " << p.C.circuit[i].size << "):\t";
//        for (i64 j = 0; j < std::min(100u, p.C.circuit[i].size); ++j)
//            cerr << p.val[i][j] << ' ';
//        cerr << endl;
//        bool flag = false;
//        for (i64 j = 0; j < p.C.circuit[i].size; ++j)
//            if (p.val[i][j] != F_ZERO) flag = true;
//        if (flag) cerr << "not all zero: " << i << endl;
//        for (i64 j = p.C.circuit[i].zero_start_id; j < p.C.circuit[i].size; ++j)
//            if (p.val[i][j] != F_ZERO) { cerr << "WRONG! " << i << ' ' << j << ' ' << p.val[i][j] << endl; exit(EXIT_FAILURE); }
//    }
    cerr << "finish creating circuit." << endl;
}

void singleConv::initParamConv() {
    i64 conv_layer_cnt = 0;
    total_in_size = 0;
    total_para_size = total_relu_in_size = total_ave_in_size = total_max_in_size = 0;

    // data
    i64 pos = pic_size_x * pic_size_y * pic_channel * pic_parallel;

    new_nx_in = pic_size_x;
    new_ny_in = pic_size_y;
    for (i64 i = 0; i < conv_section.size(); ++i) {
        auto &sec = conv_section[i];
        for (i64 j = 0; j < sec.size(); ++j) {
            refreshConvParam(new_nx_in, new_ny_in, sec[j]);
            conv_layer_cnt += sec[j].ty == FFT ? FFT_SIZE - 1 : sec[j].ty == NAIVE ? NCONV_SIZE : NCONV_FAST_SIZE;
            // conv_kernel
            sec[j].weight_start_id = pos;
            pos += sqr(m) * channel_in * channel_out;
            total_para_size += sqr(m) * channel_in * channel_out;
            sec[j].bias_start_id = -1;
        }
    }
    total_in_size = pos;

    SIZE = 1 + conv_layer_cnt;
    cerr << "SIZE: " << SIZE << endl;
}

vector<F> singleConv::getFFTAns(const vector<F> &output) {
    vector<F> res;
    res.resize(nx_out * ny_out * channel_out * pic_channel * pic_parallel);

    i64 lst_fft_lenh = getFFTLen() >> 1;
    i64 L = -padding, Rx = nx_in + padding, Ry = ny_in + padding;
    for (i64 p = 0; p < pic_parallel; ++p)
    for (i64 co = 0; co < channel_out; ++co)
        for (i64 x = L; x + m <= Rx; x += (1 << log_stride))
            for (i64 y = L; y + m <= Ry; y += (1 << log_stride)) {
                i64 idx = tesIdx(p, co, ((x - L) >> log_stride), ((y - L) >> log_stride), channel_out, nx_out, ny_out);
                i64 i = cubIdx(p, co, matIdx(Rx - x - 1, Ry - y - 1, ny_padded_in), channel_out, lst_fft_lenh);
                res[idx] = output[i];
            }
    return res;
}

double singleConv::calcRawFFT() {
   auto in = val[0].begin();
   auto conv = val[0].begin() + conv_section[0][0].weight_start_id;
   auto bias = val[0].begin() + conv_section[0][0].bias_start_id;

   timer tm;
   int logn = ceilPow2BitLength(nx_padded_in * ny_padded_in) + 1;
   vector<F> res(nx_out * ny_out * channel_out * pic_parallel, F_ZERO);
   vector<F> arr1(1 << logn, F_ZERO);
   vector<F> arr2(arr1.size(), F_ZERO);

   assert(pic_parallel == 1 && pic_channel == 1);
   // data matrix
   i64 L = -padding;
   i64 Rx = nx_in + padding, Ry = ny_in + padding;

   tm.start();
   for (i64 x = L; x < Rx; ++x)
       for (i64 y = L; y < Ry; ++y)
           if (check(x, y, nx_in, ny_in)) {
               i64 g = matIdx(Rx - x - 1, Ry - y - 1, ny_padded_in);
               i64 u = matIdx(x, y, ny_in);
               arr1[g] = in[u];
           }

   // kernel matrix
   for (i64 x = 0; x < nx_padded_in; ++x)
       for (i64 y = 0; y < ny_padded_in; ++y)
           if (check(x, y, m, m)) {
               i64 g = matIdx(x, y, ny_padded_in);
               i64 u = matIdx(x, y, m);
               arr2[g] = conv[u];
           }

   fft(arr1, logn, false);
   fft(arr2, logn, false);
   for (i64 i = 0; i < arr1.size(); ++i)
       arr1[i] = arr1[i] * arr2[i];
   fft(arr1, logn, true);
   reverse(arr1.begin(), arr1.end());

   tm.stop();
   return tm.elapse_sec();
}

double singleConv::calcRawNaive() {
   auto in = val[0].begin();
   auto conv = val[0].begin() + conv_section[0][0].weight_start_id;
   auto bias = val[0].begin() + conv_section[0][0].bias_start_id;

   timer tm;
   vector<F> res(nx_out * ny_out * channel_out * pic_parallel, F_ZERO);
   tm.start();
   for (i64 p = 0; p < pic_parallel; ++p)
       for (i64 i = 0; i < channel_out; ++i)
           for (i64 j = 0; j < channel_in; ++j)
               for (i64 x = -padding; x + m <= nx_in + padding; x += (1 << log_stride))
                   for (i64 y = -padding; y + m <= ny_in + padding; y += (1 << log_stride)) {
                       i64 idx = tesIdx(p, i, (x + padding) >> log_stride, (y + padding) >> log_stride, channel_out, nx_out, ny_out);
                       if (j == 0) res[idx] = res[idx] + bias[i];
                       for (i64 tx = x; tx < x + m; ++tx)
                           for (i64 ty = y; ty < y + m; ++ty)
                               if (check(tx, ty, nx_in, ny_in)) {
                                   i64 u = tesIdx(p, j, tx, ty, channel_in, nx_in, ny_in);
                                   i64 v = tesIdx(i, j, tx - x, ty - y, channel_in, m, m);
                                   res.at(idx) = res.at(idx) + in[u] * conv[v];
                               }
                   }
   tm.stop();
   return tm.elapse_sec();
}
