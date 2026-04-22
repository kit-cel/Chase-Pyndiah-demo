#ifndef GPC_H
#define GPC_H

#include"BCH.h"
#include "SimpleNN.h"
#include <unordered_set>
#include <algorithm>



class GPC {
public:
    const static bool genie_aided = false;
    //component code parameters============
    std::shared_ptr<BCH> mBCH;
    int mN;
    int mT;
    int mK;
    int ml; //number of shortened positions, at the beginning of the information bits
    bool even_check;

    //component code parameters============

    //for debugging============
    BitArray DRSini; //store the DRS for each bit
    BitArray Databufferint;
    //for debugging============


    //this is the data buffer transmitted through the channel==========
    int mBufferLen;
    int mBufferWidth;
    std::vector<int> mVariantBufferWidth; //for databuffer that are not squared
    BitArray Databuffer;
    BitArray x; //for random cw, save a copy of the originial data buffer
    int countErrorStart; //for spatially coupled codes, do not count the starting and ending blocks
    int countErrorEnd;
    double BufferSize;
    double miscorrectionBit;
    //this is the data buffer transmitted through the channel==========


    //Tanner graph perspective===================
    int mNumCNs;
    SynArray Syn;
    std::vector<bool> zeroSyn;
    std::vector<uint8_t> even; //saved for every CN
    bool reached_all_zero_syndrome;
    //Tanner graph perspective===================


    //generally for decoding===================
    int decIter;
    int windowLen;
    //generally for decoding===================


    //for SISO decoding ===================
    std::vector<std::vector<double>> Rin; //store the LLR for each bit
    std::vector<std::vector<double>> Rout;
    std::vector<std::vector<double>> extrinsic;
    int p_chase;
    std::vector<double> alpha;
    std::vector<double> beta;
    std::vector<double> MDsclae;
    std::vector<double> top2Threshold;
    std::vector<double> NotMDscale;
    std::vector<BitVector> MLcodewords;
    std::vector<bool> is_miscorrection;
    std::vector<double> phi;
    std::vector<BitVector> ChasePatterns;
    int ChasePatternNum;
    bool ChaseII = true;
    bool top2 = false;
    bool NN4MDOnly = true;

    //for SISO decoding ===================




    //parameter of the GPC===================
    bool mAllzeroCw = false;
    double mRate;
    int totalErrors;
    //parameter of the GPC===================

    std::random_device rd;
    std::mt19937 gen{rd()};

    explicit GPC(const std::shared_ptr<BCH> &code);
    virtual ~GPC() = default;
    void encode();

    void simulate_transmission_BIAWGN_SISO(double stddev);
    void simulate_transmission_BSC(double delta);

    double iBDD_block();
    double iBDD_window();


    void SISO_step_x(int CNidx, int iter);
    double SISO_block_x();

    void original_CP_step(int CNidx, int iter);
    double original_CP_block();

    void buildFillingPattern(int era_num, int J, std::vector<int> &v);
    void iBDDcorrection(std::vector<int> ErrLoc, int numErr, int CNidx);


    virtual void calculate_rate() = 0;
    virtual void syn2syn(int CNidx, int VNidx, int &otherCN, int &otherVN) = 0;
    virtual void syn2data(int CNidx, int VNidx, int &i, int &j) = 0;
    virtual std::vector<std::vector<int>> data2syn(int i, int j) = 0;
    virtual void setValidBufferRange() = 0;
    void clearErasures(int CN_l, int CN_r);
    BitVector get_CN_cw(int CNidx) {
        BitVector cw(mN, 0);
        for (int VNidx = 0; VNidx < mN; VNidx++) {
            int i, j;
            syn2data(CNidx, VNidx, i, j);
            if (i >= 0) cw[VNidx] = Databuffer[i][j];

        }
        return cw;
    }


    void clear_buffers() {
        Databuffer.clear();
        Rin.clear();
        Rout.clear();
        extrinsic.clear();
        x.clear();
        Syn.clear();
        even.clear();
        zeroSyn.clear();
        totalErrors = 0;
    }



    void throughtest() {
        int errorCount = 0;
        int erasureCount = 0;

        for (int i = countErrorStart; i < countErrorEnd; i++) {
            for (int j = 0; j < mBufferWidth; j++) {
                if (Databuffer[i][j] == 2) {
                    erasureCount++;
                }
                if (mAllzeroCw) {
                    if (Databuffer[i][j] == 1) {
                        errorCount++;
                    }
                } else {
                    if (Databuffer[i][j] !=2 && Databuffer[i][j] != x[i][j]) {
                        errorCount++;
                    }
                }
            }
        }

        for (int CNidx = 0; CNidx < mNumCNs; CNidx++) {
            auto const cw = get_CN_cw(CNidx);
            SynVec sync;
            bool zeroSynC = mBCH->compute_syndrome(cw, sync);


            int even_cw = 0;
            bool haserror = false;
            for (auto c: cw) {
                if (c == 1) {
                    even_cw ^= 1;
                    haserror = true;
                }
                if (c != 0 && c != 1 && c != 2) {
                    std::cout << ", CNidx: " << CNidx << "\n";
                    throw std::runtime_error("bit value wrong after something");
                }
            }

            if (zeroSynC != zeroSyn[CNidx]) {
                if (haserror) {
                    std::cout << "codeword:\n";
                    displayVector(cw);
                    std::cout << "syndrome:\n";
                    displayVector(sync);
                    std::cout << "syndrome in register:\n";
                    displayVector(Syn[CNidx]);
                    std::cout << "zero syn by BCH " << zeroSynC << ", by code: " << zeroSyn[CNidx] << "\n";
                }
                throw std::runtime_error("syndrome track wrong");
            }
            if (even_cw != even[CNidx]) {
                std::cout << ", CNidx: " << CNidx << "\n";
                displayVector(cw);
                std::cout << "even should be " << even_cw << " but tracked as " << even[CNidx] << "\n";
                throw std::runtime_error("even check went wrong");
            }

        }
        if (errorCount != totalErrors) {
            std::cout << "\n actual error number " << errorCount << ", in code " << totalErrors << "\n";
            throw std::runtime_error("total error count went wrong");
        }



    }

    inline bool checkZeroSyn(int CNidx){
        for (auto s : Syn[CNidx]) {
            if (s!=0) {return false;}
        }
        return true;
    }

    inline bool isCodeword(int CNidx){
        return  (zeroSyn[CNidx] && even[CNidx]==0);
    }

    inline bool noErrorDecected(){
        for (int i=0; i<mNumCNs; i++) {if (!zeroSyn[i] || even[i]!=0) {return false; break;}}
        return true;
    }

    inline bool noErrorDecectedBDD(){
        for (int i=0; i<mNumCNs; i++) {if (!zeroSyn[i] || even[i]!=0) {return false; break;}}
        return true;
    }



    void makeChasePatterns() {
        if (ChaseII) {
            ChasePatternNum = 1 << p_chase;
            ChasePatterns = std::vector<BitVector> (pow(2,p_chase));
            for (int mask = 0; mask < (1 << p_chase); ++mask) {
                //decimal to binary
                BitVector filling_pattern(p_chase, 0);
                for (int i = 0; i < p_chase; ++i) {
                    filling_pattern[i] = (mask & (1 << i)) ? 1 : 0;
                }
                ChasePatterns[mask] = filling_pattern;
            }
        }else {
            ChasePatternNum = 1 + p_chase + p_chase * (p_chase - 1) / 2 + p_chase * (p_chase - 1) * (p_chase - 2) / 6;
            ChasePatterns = std::vector<BitVector> (ChasePatternNum);
            ChasePatterns[0] = BitVector(p_chase, 0); //the all-zero pattern
            for (int i = 0; i < p_chase; ++i) {
                BitVector filling_pattern(p_chase, 0);
                filling_pattern[i] = 1;
                ChasePatterns[1 + i] = filling_pattern; //single error patterns
            }
            int idx = 1 + p_chase;
            for (int i = 0; i < p_chase; ++i) {
                for (int j = i + 1; j < p_chase; ++j) {
                    BitVector filling_pattern(p_chase, 0);
                    filling_pattern[i] = 1;
                    filling_pattern[j] = 1;
                    ChasePatterns[idx++] = filling_pattern; //double error patterns
                }
            }
            for (int i = 0; i < p_chase; ++i) {
                for (int j = i + 1; j < p_chase; ++j) {
                    for (int k = j + 1; k < p_chase; ++k) {
                        BitVector filling_pattern(p_chase, 0);
                        filling_pattern[i] = 1;
                        filling_pattern[j] = 1;
                        filling_pattern[k] = 1;
                        ChasePatterns[idx++] = filling_pattern; //triple error patterns
                    }
                }
            }
        }





    }


    // existing constructors / methods ...

    void load_neural_network(const std::string& filename);
    double nn_score(const std::vector<double>& features) const;
    int nn_predict(const std::vector<double>& features, double threshold = 0.5) const;

    bool has_neural_network() const { return mUseNN && mNN.loaded; }


    // existing members ...

    bool mUseNN = false;
    SimpleNN mNN;
    double stddev;

};



#endif //GPC_H
