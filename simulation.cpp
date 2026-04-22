#include "simulation.h"

simulation::simulation(int n_, int t_, bool even_, bool extend_, int shorten_, int codeType_, int decodeMode_)
: n(n_), t(t_), even(even_), extend(extend_), shorten(shorten_), codeType(codeType_), decodeMode(decodeMode_) {
    readG(G,n,t,even, extend);
    mBCH = std::make_shared<BCH>(n,t,even, extend, G, shorten);
    debug = false;


}

#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<std::vector<double>> simulation::simBerCurve() {


    result.clear();

    if (displayResult) {
        auto output = displaySelected(std::vector<double>{}, printIndices, true);
        std::cout << output.str() << "\\\\\n";
    }

    using clock = std::chrono::steady_clock;
    int innerloopsize = inner_loop_size;
    if (innerloopsize <= 0) innerloopsize = 1;

    if (num_max_frame % inner_loop_size != 0) {
        throw std::runtime_error("total frame number is not a multiple of the inner loop size\n");
    }

    for (double EbNo_dB = EbNo_dB_start; EbNo_dB < EbNo_dB_end; EbNo_dB += step) {
            double error_count = 0.0;
            double simulatedFrame = 0.0;
            double frame_error = 0.0;

            auto ChannelParameters = calculate_parameters(EbNo_dB, rate, 0);
            double stddev = std::sqrt(ChannelParameters[3]);
            ChannelParameters.push_back(stddev);

            std::set<double> SPsizes; // still unused (kept for behavior parity)
            miscorrections = 0;       // shared member: will be read safely later

            // simResults layout is unchanged
            std::vector<double> simResults(9, 0.0);
            simResults[4] = 1e8;

            std::chrono::minutes max_run_min(max_min);
            const auto duration =
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(max_run_min);
            using clock = std::chrono::steady_clock;
            const auto start = clock::now();



            for (int outerloop = 0; outerloop < num_max_frame / innerloopsize; outerloop += 1) {
                double error_delta = 0.0;
                long long frame_delta = 0;
                long long fe_delta = 0;

#pragma omp parallel for reduction(+:error_delta, frame_delta, fe_delta)
                for (long long rep = 0; rep < innerloopsize; ++rep) {
                    double last_result = simOnePoint(ChannelParameters);
                    error_delta += last_result;
                    frame_delta += 1;
                    fe_delta += (last_result > 1e-4);
                }

                // exactly once per outerloop, single-thread
                error_count    += error_delta;
                simulatedFrame += (double)frame_delta;
                frame_error    += (double)fe_delta;

                bool hasFE = (fe_delta > 0);


                double ber = 0.0;
                if (error_count == 0.0) {
                    ber = 1.0 / simulatedFrame / FrameSize;
                } else {
                    ber = error_count / simulatedFrame / FrameSize;
                }

                simResults[0] = simulatedFrame;
                simResults[1] = frame_error;
                if (frame_error > 0){
                    simResults[2] = frame_error / simulatedFrame; // note: if frame_error==0 this becomes 0 (same as your update block)
                    simResults[3] = ber;
                }else{
                    simResults[2] = 1.0 / simulatedFrame;
                    simResults[3] = 1.0 / (simulatedFrame * FrameSize);
                }


                // CI uses simResults[2] (FER estimate) exactly like you did
                const double p = simResults[2];
                const double confidenceInterval = 3.0 * std::sqrt(p * (1.0 - p) / simulatedFrame);
                simResults[4] = confidenceInterval;

                simResults[5] = (frame_error > 1e-3) ? (error_count / frame_error) :  (1.0 / frame_error);
                simResults[6] = error_count;

                std::chrono::duration<double> elapsed_seconds = clock::now() - start;
                simResults[7] = (elapsed_seconds.count() > 0.0)
                                ? (simulatedFrame * FrameSize) / elapsed_seconds.count()
                                : 0.0;

                // miscorrections is a member; this read is safe here (critical)
                simResults[8] = miscorrections / (simulatedFrame * FrameSize);

                if (displayResult)
                if ((simResults[2] >1e-5&& outerloop > 0 && outerloop%20==0) ||  (simResults[2] <1e-5&& hasFE)  ||  (simResults[2] <1e-4&& outerloop%100==0)){
                    auto a = ChannelParameters;
                    a.insert(a.end(), simResults.begin(), simResults.end());
                    std::cout << "% ";
                    auto output = displaySelected(a, printIndices);
                    output << "\\\\ % ";
                    auto now = std::chrono::system_clock::now();
                    std::time_t t = std::chrono::system_clock::to_time_t(now);
                    output << std::ctime(&t);
                    std::cout << output.str();
                }
                // Global stop conditions (now synchronized)
                if (frame_error >= num_frame_error)
                    break;
                if (simulatedFrame >= num_max_frame)
                    break;
                auto end = clock::now();
                if (end - start >= duration) break;
            }

            auto a = ChannelParameters;
            a.insert(a.end(), simResults.begin(), simResults.end());
            result.push_back(a);

            if (displayResult) {
                auto output = displaySelected(a, printIndices);
                output << "\\\\ % ";
                auto now = std::chrono::system_clock::now();
                std::time_t t = std::chrono::system_clock::to_time_t(now);
                output << std::ctime(&t);
                std::cout << output.str();
            }


            if (simResults[3] < stop_ber) {
                break;
            }
        }
    return result;
}



double simulation::simOnePoint(const std::vector<double> &ChannelParamter) {
   {
        productCode code(mBCH);
        code.decIter = decIter;
        code.mAllzeroCw = zeroCw;
        code.encode();
        code.ChaseII = ChaseII;
        code.top2 = top2;
        code.NN4MDOnly = NN4MDOnly;
        if (decodeMode == 0) {
            code.simulate_transmission_BSC(ChannelParamter[2]);
            return code.iBDD_block();
        } else if (decodeMode == 1) {
            code.p_chase = p_chase;
            code.alpha = alpha;
            code.beta = beta;
            code.MDsclae = MDscale;
            code.top2Threshold = top2Threshold;
            code.simulate_transmission_BIAWGN_SISO(ChannelParamter[5]);
            double resultBE = code.original_CP_block();
            miscorrections += code.miscorrectionBit;
            return resultBE;
        }
        else if (decodeMode == 2) {
            code.p_chase = p_chase;
            code.alpha = alpha;
            code.beta = beta;
            code.MDsclae = MDscale;
            code.top2Threshold = top2Threshold;
            code.NotMDscale = NotMDscale;
            code.simulate_transmission_BIAWGN_SISO(ChannelParamter[5]);
            double resultBE = code.SISO_block_x();
            miscorrections += code.miscorrectionBit;
            return resultBE;
        }
        else {
            throw std::runtime_error("invalid decoder type!");
        }

    }





}



