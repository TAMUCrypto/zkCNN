//
// Created by 69029 on 4/12/2021.
//

#include "circuit.h"
#include "neuralNetwork.hpp"
#include "verifier.hpp"
#include "models.hpp"
#include "global_var.hpp"

// the arguments' format
#define INPUT_FILE_ID 1     // the input filename
#define CONFIG_FILE_ID 2    // the config filename
#define OUTPUT_FILE_ID 3    // the input filename
#define NETWORK_FILE_ID 4   // the configuration of vgg
#define PIC_CNT 5           // the number of picture paralleled

vector<std::string> output_tb(16, "");

int main(int argc, char **argv) {
    initPairing(mcl::BLS12_381);

    char i_filename[500], c_filename[500], o_filename[500], n_filename[500];

    strcpy(i_filename, argv[INPUT_FILE_ID]);
    strcpy(c_filename, argv[CONFIG_FILE_ID]);
    strcpy(o_filename, argv[OUTPUT_FILE_ID]);
    strcpy(n_filename, argv[NETWORK_FILE_ID]);

    int pic_cnt = atoi(argv[PIC_CNT]);

    output_tb[MO_INFO_OUT_ID] ="vgg (relu)";
    output_tb[PCNT_OUT_ID] = std::to_string(pic_cnt);

    prover p;
    vgg nn(32, 32, 3, pic_cnt, i_filename, c_filename, o_filename, n_filename);
    nn.create(p, false);
    verifier v(&p, p.C);
    v.verify();

    for (auto &s: output_tb) printf("%s, ", s.c_str());
    puts("");
}