#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "simulation.h"




double simOnePointSISO(
    double EbNo_dB, int n, int t, bool even, bool extend, int shorten,
    int codeType, int decodeMode, int decIter, int p_chase, bool useChaseII, bool top2, bool NN4MDonly,
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    const std::vector<double>& MDscale,
    const std::vector<double>& top2Threshold,
    const std::vector<double>& NotMDscale){
    //    std::cout<<"Simulate BER at one SNR point\n";
    simulation sim(n, t, even, extend, shorten, codeType, decodeMode);
    sim.displayResult = false;
    // GPC parameters
    sim.GPCparameters(100, 8, 0, 0);
    if (top2)
        std::cout<<"use top2\n";
    if (NN4MDonly)
        std::cout<<"NN4MDonly\n";

    // Optional: set decoder params
    sim.decIter = decIter;
    sim.p_chase = p_chase;
    sim.ChaseII = useChaseII;
    sim.top2 = top2;
    sim.NN4MDOnly = NN4MDonly;
    if (!top2)
        sim.NN4sclae = !NN4MDonly;
    sim.zeroCw = false;

    // Set Eb/No range
    sim.EbNo_dB_start = EbNo_dB;
    sim.step = 0.1;
    sim.EbNo_dB_end = EbNo_dB + sim.step/2;
    sim.num_frame_error = 100;
    sim.saveSP = false;
    sim.alpha = alpha;
    sim.beta  = beta;
    sim.MDscale  = MDscale;
    sim.top2Threshold = top2Threshold;
    sim.NotMDscale = NotMDscale;
    sim.num_max_frame = 196*100;
    // sim.printParameters();

    // Run simulation
    auto results = sim.simBerCurve();
    return results[0][9];
}

double simBERcurveSISO(
    double EbNo_dB, double EbNo_dB_end, int n, int t, bool even, bool extend, int shorten,
    int codeType, int decodeMode, int decIter, int p_chase,bool useChaseII, bool top2, bool NN4MDonly,
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    const std::vector<double>& MDscale,
    const std::vector<double>& top2Threshold,
    const std::vector<double>& NotMDscale){
    //    std::cout<<"Simulate BER at one SNR point\n";
    simulation sim(n, t, even, extend, shorten, codeType, decodeMode);
    sim.displayResult = true;
    sim.num_max_frame = 196*100;

    // GPC parameters
    sim.GPCparameters(100, 8, 0, 0);


    // Optional: set decoder params
    sim.decIter = decIter;
    sim.p_chase = p_chase;

    sim.ChaseII = useChaseII;
    sim.top2 = top2;
    sim.NN4MDOnly = NN4MDonly;
    if (!top2)
        sim.NN4sclae = !NN4MDonly;

    // Set Eb/No range
    sim.EbNo_dB_start = EbNo_dB;
    sim.step = 0.05;
    sim.EbNo_dB_end = EbNo_dB_end;
    sim.num_frame_error = 100;
    sim.saveSP = false;
    sim.alpha = alpha;
    sim.beta  = beta;
    sim.MDscale  = MDscale;
    sim.top2Threshold = top2Threshold;
    sim.NotMDscale = NotMDscale;
    sim.zeroCw = false;
    sim.stop_ber = 1e-5;
    sim.printParameters();

    // Run simulation
    auto results = sim.simBerCurve();
    return results[0][9];
}

PYBIND11_MODULE(GPC_simulator_python, handle){
    handle.doc() = "python wrapper";
    handle.def("simOnePointSISO", &simOnePointSISO, "Simulate BER at one SNR point for SISO decoding");
    handle.def("simBERcurveSISO", &simBERcurveSISO, "Simulate BER curve for SISO decoding");
}