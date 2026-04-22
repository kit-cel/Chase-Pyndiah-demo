#ifndef GPC_SIMULATION_H
#define GPC_SIMULATION_H

#include "lib/productCode.h"
#include "lib/staircaseCode.h"


#include <fstream>
//#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <thread>
//namespace fs = std::filesystem;


class simulation {
public:
    //regarding component code
    int n;
    int t;
    bool even;
    bool extend;
    int shorten;
    std::vector<std::vector<uint8_t>> G;
    std::shared_ptr<BCH> mBCH;

    //regarind simulation setup
    double EbNo_dB_start;
    double EbNo_dB_end;
    double step = 0.1;
    long int num_max_frame = 1e6;
    double stop_ber = 2e-10;
    int num_frame_error = 100;
    bool displayResult;
    int max_min = 15;
    int inner_loop_size;

    //for the GPD
    bool zeroCw = true;
    double rate;
    int num_blocks=50;
    int smallBlockNumPerRow;
    int skip_start;
    int skip_end;
    int FrameSize;
    int codeType=0;
    int decodeMode=2;

    //decoder setup
    int decIter;
    int windowLen;


    //for Chase-Pyndian decoder================
    int p_chase;
    std::vector<double> alpha;
    std::vector<double> beta;
    std::vector<double> MDscale;
    std::vector<double> top2Threshold;
    std::vector<double> NotMDscale;

    bool ChaseII = true;
    bool top2 = false;
    bool NN4MDOnly = true;


    //for Chase-Pyndian decoder================

    bool debug = false;
    bool saveSP;
    std::vector<std::vector<double>> result;
    std::vector<std::vector<std::vector<double>>> result_with_PP;
    std::vector<bool> reached_stop_ber;

    double miscorrections;


    // Index mapping for result vector:
    //  0: EbNo          - Energy per bit to noise power spectral density ratio (dB)
    //  1: EsNo          - Energy per symbol to noise power spectral density ratio (dB)
    //  2: ErrorProb     - Channel error probability
    //  3: NoiseVar      - Channel noise variance
    //  4: ErasureProb   - Channel erasure probability
    //  5: stddev
    //  6: DecodedFrame  - Number of decoded frames
    //  7: FrameErr      - Number of frames with errors
    //  8: FER           - Frame Error Rate
    //  9: BER           - Bit Error Rate
    // 10: Confidence Iterval for 3p_e
    // 11: average number of bit errors in an errorenous frame
    // 12: number of bit errors
    std::vector<std::string> labels = {
            "EbNo", "EsNo", "delta", "NoiseVar", "ErasureProb", "stddev",
            "totalFrame", "FE", "FER", "BER", "CIfer", "SPsize", "BE","throughput","mis rate"
    };
    std::vector<int> printIndices = {0,1,2,4,6,7,8,9, 11, 12, 13, 14};


    std::vector<std::string> colors = {
            "red", "blue", "green", "cyan", "magenta",
            "black", "gray", "brown", "violet", "teal", "olive", "yellow"
    };




    simulation(int n_, int t_, bool even_, bool extend_, int shorten_, int codeType_, int decodeMode_);
    std::vector<std::vector<double>> simBerCurve();
    double simOnePoint(const std::vector<double>& ChannelParamter);


    void GPCparameters(int num_blocks_val, int smallBlockNumPerRow_val, int skip_start_val, int skip_end_val) {
        num_blocks = num_blocks_val;
        smallBlockNumPerRow = smallBlockNumPerRow_val;
        skip_start = skip_start_val;
        skip_end = skip_end_val;

        if (codeType == 0){
            productCode PC(mBCH);
            rate = PC.mRate;
            FrameSize = PC.BufferSize;

        }else if(codeType == 1){
            staircaseCode SCC(mBCH, num_blocks);
            rate = SCC.mRate;
            FrameSize = SCC.BufferSize;

        }
        else{
            throw std::runtime_error("invalid code type!");
        }

    }



    std::ostringstream displaySelected(const std::vector<double>& result, const std::vector<int>& indices, bool print_labels=false) {

        std::ostringstream output;

        if (print_labels) {
            for (int idx: indices) {
                if (idx == 0 || idx==7) {
                    output << std::setw(6) << labels[idx] << ";";
                }
                else if (idx > 0 && idx < labels.size()) {
                    output << std::setw(12) << labels[idx] << ";";
                } else {
                    std::cerr << "Index out of range: " << idx << '\n';
                }
            }
        }else{
            for (int idx : indices) {
                if (idx == 0 || idx==7) {
                    output << std::setw(6) << result[idx] <<";";
                }
                else if (idx > 0 && idx < result.size()) {
                    output << std::setw(12) << result[idx] << ";";
                } else {
                    output << "Index out of range: " << idx << '\n';
                }
            }
        }

        return output;
    }
    void printParameters() const {
        // Component code
        std::cout << "\%Component Code: n=" << n << ", t=" << t
                  << ", even=" << std::boolalpha << even
                  << ", extend=" << extend << ", shorten=" << shorten << std::endl;

        // Simulation setup
        std::cout << "\%Simulation: EbNo_dB_start=" << EbNo_dB_start
                  << ", EbNo_dB_end=" << EbNo_dB_end << ", step=" << step
                  << ", max_frame=" << num_max_frame << ", stop_ber=" << stop_ber
                  << ", frame_error=" << num_frame_error
                  << ", displayResult=" << displayResult << std::endl;
        // Code type & mode
        std::cout << "\%CodeType = product code"
                  << ", DecodeMode="
                  << (decodeMode == 0 ? "iBDD" : decodeMode == 1 ? "original Chase-Pyndiah" : decodeMode == 2 ? "modified Chase-Pyndiah" : "unknown")
                  << std::endl;
        if (decodeMode==2) {
            if (top2 && (NN4MDOnly)) {
                throw std::runtime_error("invalid decode mode!");
            }
            if (top2)
                std::cout << "\%use top2 MD\n";
            if (NN4MDOnly) {
                std::cout << "\%use NN-assisted MD\n";
            }
            if (GPC::genie_aided) {
                std::cout << "\%use GPC genie_aided\n";
            }
        }
        // GPC
        std::cout << "\%GPD: zeroCw=" << zeroCw << ", rate=" << rate<< ", FrameSize=" << FrameSize ;
        if (codeType ==0 || codeType == 3){
            std::cout << std::endl;

        }else{
            std::cout << ", blocks=" << num_blocks
                      << ", perRow=" << smallBlockNumPerRow
                      << ", skip_start=" << skip_start
                      << ", skip_end=" << skip_end
                      << std::endl;
        };

        if (decodeMode>0) {
            if (ChaseII) {
                std::cout << "\% use Chase II patterns\n";
            }else {
                std::cout << "\% use Chase I patterns\n";
            }
        }

        // Decoder
        if (codeType==0 || codeType==3) {
            std::cout << "\%Decoder: decIter=" << decIter << std::endl;
        }else{
            std::cout << "\%Decoder: decIter=" << decIter
                      << ", windowLen=" << windowLen << std::endl;

        }


        if (decodeMode>0) {
            std::cout<< "\%p_chase=" << p_chase << ", alpha=[";;
            for (size_t i = 0; i < alpha.size(); ++i) {
                std::cout << alpha[i];
                if (i != alpha.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n\%beta=[";
            for (size_t i = 0; i < beta.size(); ++i) {
                std::cout << beta[i];
                if (i != beta.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            if (!MDscale.empty()) {
                std::cout<<"\%scaling = [";
                for (size_t i = 0; i < MDscale.size(); ++i) {
                    std::cout << MDscale[i];
                    if (i != MDscale.size() - 1) std::cout << ", ";

                }
                std::cout << "]\n";

            }

            if (!top2Threshold.empty()) {
                std::cout<<"\%md = [";
                for (size_t i = 0; i < top2Threshold.size(); ++i) {
                    std::cout << top2Threshold[i];
                    if (i != top2Threshold.size() - 1) std::cout << ", ";
                }
                std::cout << "]\n";
            }

            if (!NotMDscale.empty()) {
                std::cout<<"\%notmd = [";
                for (size_t i = 0; i < NotMDscale.size(); ++i) {
                    std::cout << NotMDscale[i];
                    if (i != NotMDscale.size() - 1) std::cout << ", ";
                }
                std::cout << "]\n";
                }
            }






    }

    std::string getCurrentTimestamp() {
        std::time_t now = std::time(nullptr);
        std::tm* tm_ptr = std::localtime(&now);
        std::ostringstream oss;
        oss << std::put_time(tm_ptr, "%Y%m%d_%H%M%S");
        return oss.str();
    }



};


#endif //GPC_SIMULATION_H
