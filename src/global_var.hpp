//
// Created by 69029 on 5/4/2021.
//

#include <bits/stdc++.h>
#include <hyrax-bls12-381/polyCommit.hpp>

#ifndef ZKCNN_GLOBAL_VAR_HPP
#define ZKCNN_GLOBAL_VAR_HPP

// the output format
#define MO_INFO_OUT_ID 0
#define PSIZE_OUT_ID 1
#define KSIZE_OUT_ID 2
#define PCNT_OUT_ID 3
#define CONV_TY_OUT_ID 4
#define QS_OUT_ID 5
#define WS_OUT_ID 6
#define PT_OUT_ID 7
#define VT_OUT_ID 8
#define PS_OUT_ID 9
#define POLY_PT_OUT_ID 10
#define POLY_VT_OUT_ID 11
#define POLY_PS_OUT_ID 12
#define TOT_PT_OUT_ID 13
#define TOT_VT_OUT_ID 14
#define TOT_PS_OUT_ID 15

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::max;
using std::min;
using std::ifstream;
using std::ofstream;
using std::ostream;
using std::pair;
using std::make_pair;

extern vector<string> output_tb;

#define F Fr
#define G G1
#define F_ONE (Fr::one())
#define F_ZERO (Fr(0))

#define F_BYTE_SIZE (Fr::getByteSize())

template <typename T>
string to_string_wp(const T a_value, const int n = 4) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

#endif //ZKCNN_GLOBAL_VAR_HPP
