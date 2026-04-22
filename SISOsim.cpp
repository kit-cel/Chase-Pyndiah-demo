#include "simulation.h"
int main(int argc, char** argv) {
    int n = 255;
    int t = 2;
    bool extend = true;
    int shorten = 0;
    bool even = false;

    //(decodeMode == 0 ? "iBDD" : decodeMode == 1 ? "original Chase-Pyndiah" : decodeMode == 2 ? "modified Chase-Pyndiah" : "unknown")
    int decodeMode =2;

    int codeType = 0;

    simulation sim(n, t, even, extend, shorten, 0, 2);

    // GPC parameters
    sim.GPCparameters(100, 1, 0, 0);
    sim.EbNo_dB_start = 3.825;
    sim.decIter =  4;
    sim.p_chase =  5;
    sim.num_frame_error = 100;
    sim.stop_ber = 1e-7;
    sim.displayResult = true;
    sim.debug = false;
    sim.zeroCw = false;
    sim.decodeMode = decodeMode;
    sim.ChaseII = true;
    sim.top2 = false;
    sim.NN4MDOnly = true;
    sim.inner_loop_size = omp_get_max_threads();
    sim.num_max_frame = sim.inner_loop_size * 100000;

    // omp_set_num_threads(1);

    if (codeType > 0) {
        sim.decIter = 7;
        sim.windowLen = 8;
    }

    // Set Eb/No range
    // sim.EbNo_dB_start =std::stod(argv[11]);

    sim.EbNo_dB_end = 5.2;
    sim.step = 0.025;


    //limit the max number of minute for simulating a single data point
    sim.max_min = 60;
    sim.saveSP = false;

    //need to change accord to RecordedParameters
    sim.alpha = {0.265, 0.34, 0.35, 0.355, 0.36, 0.365, 0.465, 0.47};

    sim.beta = {0.61, 0.665, 0.67, 0.68, 0.685, 0.69, 0.72, 0.725};
    //
    sim.top2Threshold = {
        4.1, 2.2, 2.1, 1.7, 1, 1, 0.7, 0.5
    };
    sim.MDscale = {0.16, 0.17, 0.18, 0.21, 0.22, 0.23, 0.26, 0.4};


    if (GPC::genie_aided) {
        sim.top2 = false;
        sim.NN4MDOnly = false;
        sim.alpha = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        sim.beta = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        sim.MDscale = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        sim.zeroCw = true;
    }
    // Run simulation
    sim.printParameters();
    auto results = sim.simBerCurve();
}
