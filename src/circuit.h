#pragma once

#include <vector>
#include <unordered_map>
#include <utility>
#include <hyrax-bls12-381/polyCommit.hpp>
#include <unordered_set>
#include <iostream>
#include "global_var.hpp"

using std::cerr;
using std::endl;
using std::vector;

struct uniGate {
    u32 g, u;
    u8 lu, sc;
    uniGate(u32 _g, u32 _u, u8 _lu, u8 _sc) :
        g(_g), u(_u), lu(_lu), sc(_sc) {
//        cerr << "uni: " << g << ' ' << u << ' ' << lu <<' ' << sc.real << endl;
    }
};

struct binGate {
    u32 g, u, v;
    u8 sc, l;
    binGate(u32 _g, u32 _u, u32 _v, u8 _sc, u8 _l):
        g(_g), u(_u), v(_v), sc(_sc), l(_l) {
//        cerr << "bin: " << g << ' ' << u << ' ' << lu << ' ' << v << ' ' << lu << ' ' << sc.real << endl;
    }
    [[nodiscard]] u8 getLayerIdU(u8 layer_id) const { return !l ? 0 : layer_id - 1; }
    [[nodiscard]] u8 getLayerIdV(u8 layer_id) const { return !(l & 1) ? 0 : layer_id - 1; }
};

enum class layerType {
    INPUT, FFT, IFFT, ADD_BIAS, RELU, Sqr, OPT_AVG_POOL, MAX_POOL, AVG_POOL, DOT_PROD, PADDING, FCONN, NCONV, NCONV_MUL, NCONV_ADD
};

class layer {
public:
    layerType ty;
	u32 size{}, size_u[2]{}, size_v[2]{};
	i8 bit_length_u[2]{}, bit_length_v[2]{}, bit_length{};
    i8 max_bl_u{}, max_bl_v{};

    bool need_phase2;

    // bit decomp related
    u32 zero_start_id;

    std::vector<uniGate> uni_gates;
	std::vector<binGate> bin_gates;

	vector<u32> ori_id_u, ori_id_v;
    i8 fft_bit_length;

    // iFFT or avg pooling.
    F scale;

	layer() {
        bit_length_u[0] = bit_length_v[0] = -1;
        size_u[0] = size_v[0] = 0;
        bit_length_u[1] = bit_length_v[1] = -1;
        size_u[1] = size_v[1] = 0;
        need_phase2 = false;
        zero_start_id = 0;
        fft_bit_length = -1;
        scale = F_ONE;
	}

	void updateSize() {
	    max_bl_u = std::max(bit_length_u[0], bit_length_u[1]);
	    max_bl_v = 0;
	    if (!need_phase2) return;

        max_bl_v = std::max(bit_length_v[0], bit_length_v[1]);
	}
};

class layeredCircuit {
public:
	vector<layer> circuit;
    u8 size;
    vector<F> two_mul;

    void init(u8 q_bit_size, u8 _layer_sz);
	void initSubset();
};

