#include "../lib/GPC.h"

GPC::GPC(const std::shared_ptr<BCH>& code) : mBCH(code) {
    mN = code->mN;
    mT = code->mT;
    mK = code->mK;
    ml = code->mShorten;
    even_check = code->mEven || code ->mExtended;
}



void GPC::encode() { //only for code where VN degrees are all 2
    Databuffer = BitArray(mBufferLen, BitVector(mBufferWidth,0));
    if (!mAllzeroCw) {
        std::random_device rd;
        std::mt19937 mt_eng(rd());
        std::uniform_int_distribution<> distmsg(0, 1);
        int otherCN, otherVN;
        int i,j;

        //encoding is performed CN by CN
        for (int CNidx = 0; CNidx < mNumCNs; CNidx++) {
            //before generating random information bits, we need to know if some of the bits are copied from previous CN
            //if yes, we need to copy the bits and generate the rest, if not, we generate k bits
            //to do so, we use the assumption that CN i is encoded before CN j if i<j
            //For spatially-coupled codes, the first block is initialized as 0, therefore, we use the same CN index for the
            //other CN. Then, for this case, we do not handle it, but use the default 0s in the msg.
            BitVector msg = BitVector(mK,0);
            for(int VNidx=0; VNidx<mK; VNidx++) {
                syn2syn(CNidx,VNidx, otherCN, otherVN);
                if (otherCN<0){msg[VNidx] = 0;} //shortened beginning bits for spatially coupled codeds
                else {
                    if (otherCN > CNidx) {
                        msg[VNidx] = distmsg(mt_eng);
                    }else if (otherCN < CNidx){
                        syn2data(CNidx, VNidx, i,j);
                        msg[VNidx] = Databuffer[i][j];
                    }
                }
            }
            BitVector cw = BitVector(mN);
            mBCH->encode(msg, cw);
            for(int VNidx=0; VNidx<mN; VNidx++) {
                syn2data(CNidx, VNidx, i,j);
                if (i>=0) Databuffer[i][j] = cw[VNidx];
            }
        }


    }
}


void GPC::simulate_transmission_BSC(double delta) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::geometric_distribution<> d(delta);

    if (!mAllzeroCw) x = Databuffer;
    int totallen = mBufferLen*mBufferWidth;
    zeroSyn  = std::vector<bool>(mNumCNs, true);
    even =BitVector(mNumCNs,0);
    Syn = SynArray(mNumCNs, std::vector<uint16_t>(mT, 0));
    totalErrors = 0;

    int errorPos =countErrorStart*mBufferWidth  -1; //for spatially coupled codes, the beginning and end doesn't have errors
    while (delta!=0) {
        errorPos += d(gen);
        errorPos++;
        if (errorPos >= totallen) {
            break;
        }
        auto errorPosRow = errorPos / mBufferWidth;
        if (errorPosRow >= countErrorEnd){
            break;
        }
        auto errorPosCol = errorPos % mBufferWidth;

        if (!mVariantBufferWidth.empty() && errorPosCol >= mVariantBufferWidth[errorPosRow]) {
            continue;
        }
        Databuffer[errorPosRow][errorPosCol]^= 1;
        totalErrors++;

        auto pairs = data2syn(errorPosRow, errorPosCol);
        for (auto pair : pairs) {
            int CNidx = pair[0];
            int VNidx = pair[1];
            if (VNidx+ml<mBCH->primitiveLength) {
                for (int t = 0; t < mT; t++) {
                    Syn[CNidx][t]^=mBCH->mGF.a_pow_tab[((2 * t + 1) * (VNidx+ml)) % mBCH->primitiveLength];
                }
            }
            if (even_check) even[CNidx]^=1;
        }

    }

    for (int CNidx=0; CNidx<mNumCNs; CNidx++) {
        zeroSyn[CNidx] = checkZeroSyn(CNidx);
    }
}


void GPC::simulate_transmission_BIAWGN_SISO(double stddev_) {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<> U(0,stddev_);
    stddev = stddev_;

    if (!mAllzeroCw) x = Databuffer;
    zeroSyn  = std::vector<bool>(mNumCNs, true);
    even =BitVector(mNumCNs,0);
    Syn = SynArray(mNumCNs, std::vector<uint16_t>(mT, 0));
    Rin = std::vector(mBufferLen, std::vector<double>(mBufferWidth, 0));
    totalErrors = 0;

    for (int i = countErrorStart; i < countErrorEnd; i++) {
        for (int j = 0; j < mBufferWidth; j++) {
            const auto r = Databuffer[i][j];
            double v, recieved, tmp;
            v = U(gen); //generate the noise

            recieved = 1.0 - 2.0*double(r) + v;
            Rin[i][j] = recieved; //LLR for BIAWGN channel, can be scaled by a factor
            tmp = fabs(recieved);
            unsigned char hard_decision;

            if (recieved < 0) {
                hard_decision = 1;
            }
            else {
                hard_decision = 0;
            }
            if (hard_decision != r)
                totalErrors++;

            auto pairs = data2syn(i,j);
            for (auto pair : pairs) {
                int CNidx = pair[0];
                int VNidx = pair[1];
                 {
                    if (hard_decision!=r) {
                        if (VNidx+ml < mBCH->primitiveLength) {
                            for (int t = 0; t < mT; t++) {
                                Syn[CNidx][t]^=mBCH->mGF.a_pow_tab[((2 * t + 1) * (VNidx+ml)) % mBCH->primitiveLength];
                            }
                        }
                        if (even_check)
                            even[CNidx]^=1;
                    }
                }
            }
            Databuffer[i][j] = hard_decision;
        }
    }
    for (int CNidx=0; CNidx<mNumCNs; CNidx++) {
        zeroSyn[CNidx] = checkZeroSyn(CNidx);
    }
}






double GPC::iBDD_block() {
    Databufferint = Databuffer;
    //for iBDD with block codes
    for (int iter = 0; iter<decIter; iter++) {
        auto back_up_zero_syn = zeroSyn;
        auto back_up_even = even;

        //iterative decoding ===============
        for (int CNidx = 0; CNidx < mNumCNs; CNidx++) {
            if (!zeroSyn[CNidx] || (zeroSyn[CNidx] && even[CNidx] != 0)) {
                std::vector<int> ErrLoc;
                int numErr = mBCH->decode(zeroSyn[CNidx], even[CNidx], Syn[CNidx], ErrLoc);
                if (!mVariantBufferWidth.empty() && numErr >0 && find(ErrLoc.begin(), ErrLoc.end(), mVariantBufferWidth[CNidx]-1) != ErrLoc.end()) {
                    continue;
                }
                if (numErr != -1)
                    iBDDcorrection(ErrLoc, numErr, CNidx);
            }
        }
        //iterative decoding ===============
        if (noErrorDecectedBDD()) {
            return double(totalErrors);
        }
    }

    if (noErrorDecectedBDD()){
        if (totalErrors>0 && totalErrors<(mBCH->dmin * mBCH->dmin)) //as long as not valid PC codeword, at least one component code will fail
            throw std::runtime_error("strange behavior in iBDD, not undetected stall pattern but not detected");
        reached_all_zero_syndrome = true;
        return double (totalErrors);
    }
    return double (totalErrors);
}

double GPC::iBDD_window() {
    /*decIter: the number of iterations per decoding window
 *Window_len: number of CNs in each window
 *move_forward: number of CNs the decoding window moves every time
 */

    int window_len = windowLen * mBufferWidth;
    int move_forward = mBufferWidth;

    for (int start = 0; start+window_len+move_forward<mNumCNs; start+=move_forward) {
        bool earlyStop;
        for (int iter=0; iter<decIter; iter++) {
            for (int CNidx = start; CNidx<start+window_len; CNidx++){
                if (!zeroSyn[CNidx] || (zeroSyn[CNidx] && even[CNidx]!=0)) {
                    std::vector<int> ErrLoc;
                    int numErr = mBCH->decode(zeroSyn[CNidx], even[CNidx], Syn[CNidx], ErrLoc);
                    if (numErr != -1)
                        iBDDcorrection(ErrLoc, numErr, CNidx);
                }
                earlyStop = true;
                std::vector<int> errornousindx;
                for (int i=start; i<start + window_len; i++) {
                    if (!zeroSyn[i] || even[i]!=0) {
                        earlyStop = false;
                        break;
                    }
                }
                if (earlyStop) {
                    break;
                }
            }
        }
    }

    return double(totalErrors);
}




void GPC::buildFillingPattern(int era_num, int J, std::vector<int> &v) {
    int toPermute = 1<<(era_num-1);
    std::vector<int> nums(toPermute,0);
    for (int i = 0; i <toPermute; i++) {
        nums[i] = i;
    }
    std::shuffle(nums.begin(), nums.end(), gen);
    for (int i = 0; i <J/2; i++) {
        v[i]=nums[i] ;
        v[i+J/2] = (toPermute<<1) - nums[i] -1;
    }
}





void GPC::iBDDcorrection(std::vector<int> ErrLoc, int numErr, int CNidx) {
    int i,j, otherCN, otherVN;

    if (countErrorStart>0) //for spatially coupled codes
    if (CNidx < countErrorStart+mBufferWidth || CNidx>= countErrorEnd - mBufferWidth) {
        for (int e = 0; e < numErr; e++) {
            int VNidx = ErrLoc[e];
            syn2data(CNidx, VNidx, i, j);
            if (i < countErrorStart|| i>= countErrorEnd){
                return; //flipping padded position, return
            }
        }
    }


    for (int e=0; e<numErr; e++) {
        int VNidx = ErrLoc[e];
        syn2data(CNidx, VNidx, i, j);
        syn2syn(CNidx, VNidx, otherCN, otherVN);
        if (i<0)
            throw std::runtime_error("accessing 0 block\n");
        //change the data buffer
        Databuffer[i][j]^=1;
        //count the number of errors
        if (i>=countErrorStart && i<countErrorEnd) {
            int errNumChange;
            if (!mAllzeroCw){errNumChange= (Databuffer[i][j]!=x[i][j] ? 1 : -1);}
            else {errNumChange =  (Databuffer[i][j]!=0 ? 1 : -1);}
            totalErrors += errNumChange;

        }

        //mask the other syndrome buffer
        if (even_check) even[otherCN]^=1;
        if (otherVN+ml<mBCH->primitiveLength) {
            for (int t = 0; t < mT; t++) {
                Syn[otherCN][t]^=mBCH->mGF.a_pow_tab[((2 * t + 1) * (otherVN+ml)) % mBCH->primitiveLength];
            }
            zeroSyn[otherCN] = checkZeroSyn(otherCN);
        }
    }
    // set the successfully decoded cw to syndrome 0
    zeroSyn[CNidx] = true;
    Syn[CNidx] = std::vector<uint16_t>(mT,0);
    even[CNidx] = 0;
}



void GPC::SISO_step_x(int CNidx, int iter) {
    //get the current codeword and reliabilities for the CNidx
    BitVector cw(mN, 0);
    std::vector<double> reliabilities(mN, 0.0);
    std::set<BitVector> seen_candidate_codewords;
    std::vector<BitVector> candidate_codewords(0);
    std::vector<bool> accepted_as_candidate(0);
    std::vector<double> metrics(0);
    std::vector<double> metricsDE(0);
    int best_candidate_idx = -1;

    for (int VNidx = 0; VNidx < mN; VNidx++) {
        int i, j;
        syn2data(CNidx, VNidx, i, j);
        if (i >= 0) {cw[VNidx] = Databuffer[i][j]; reliabilities[VNidx] = Rout[i][j];}
        else {cw[VNidx] = 0; reliabilities[VNidx] = 1000;}
    }

    // sort
    std::vector<int> idx(mN);
    for (int i = 0; i < mN; ++i)
        idx[i] = i;
    std::sort(idx.begin(), idx.end(),
        [&reliabilities](int i1, int i2) {
            return fabs(reliabilities[i1]) < fabs(reliabilities[i2]);
        });


    double best_metric = 1e9;
    std::set<int> touched_positions;
    for (int mask = 0; mask < ChasePatternNum; ++mask) {
        //decimal to binary
        std::vector<uint8_t> filling_pattern = ChasePatterns[mask];
        std::set<int> flip_pos;
        BitVector test_cw = cw;
        auto test_syn = Syn[CNidx];
        uint8_t even0 = even[CNidx];
        for (int i = 0; i < p_chase; ++i) {
            //compute syndrome of the test pattern
            if (filling_pattern[i]==1) {
                test_cw[idx[i]] ^= filling_pattern[i];
                flip_pos.insert(idx[i]);
                even0 ^= 1;
                if (idx[i]+ml < mBCH->primitiveLength) {
                    for (int t = 0; t < mT; t++) {
                        test_syn[t]^=mBCH->mGF.a_pow_tab[((2 * t + 1) * (idx[i]+ml)) % mBCH->primitiveLength];
                    }
                }
            }
        }
        bool zeroSyn0 = true;
        for (auto s: test_syn) {
            if (s != 0) {
                zeroSyn0 = false;
                break;
            }
        }

        std::vector<int> ErrLoc0;
        int numErr = mBCH->decode(zeroSyn0, even0, test_syn, ErrLoc0);
        if (numErr != -1) {
            double metric = 0;
            for (auto loc: ErrLoc0) {
                test_cw[loc]^=1;
                if (flip_pos.find(loc) == flip_pos.end()) {flip_pos.insert(loc);}
                else{flip_pos.erase(loc);}
            }
            for (int i=0; i<mN; i++) {
                metric += (test_cw[i]==0? -1.0 : 1.0) * reliabilities[i]; //argmin \sum -(1-2c_i)y_i
            }

            double destru = 0.0;

            for (int loc :flip_pos) {
                double diff = reliabilities[loc] - (test_cw[loc]==0? 1.0 : -1.0);
                destru += diff * diff;
            }


            if (seen_candidate_codewords.insert(test_cw).second) {//inserting into set success, new candidate
                // accepted_as_candidate.push_back(!miscorrection);
                candidate_codewords.push_back(test_cw);
                metrics.push_back(metric);
                metricsDE.push_back(destru);
                if ( metric < best_metric) {
                    best_metric = metric;
                    best_candidate_idx = candidate_codewords.size() - 1;
                }
                touched_positions.insert(flip_pos.begin(), flip_pos.end());
            }
        }
    }


    //uncomment this block enables genie-aided decoding, where all miscorrections are rejected
    if (genie_aided) {
        if (!candidate_codewords.empty())
        for (auto bit : candidate_codewords[best_candidate_idx]) {
            if (bit == 1) {
                is_miscorrection[CNidx] = true;
                return;
            }
        }
    }



    if (!candidate_codewords.empty()) {
        //this line enables top1 roll back
        // if (best_metric > top1Rollback)
        //     return;

        //this line enables anchor bits based roll back
        // if (!accepted_as_candidate[best_candidate_idx])
        //     is_miscorrection[CNidx] = true;

        int omega = candidate_codewords.size();
        std::vector<int> idx2(omega);
        for (int i = 0; i < omega; ++i)
            idx2[i] = i;
        std::sort(idx2.begin(), idx2.end(),
            [&metrics](int i1, int i2) {
                return metrics[i1] < metrics[i2];
            });

        // this block enables top2 roll back
        if (top2) {
            if (omega>1) {
                auto tmp = metrics[idx2[1]] - best_metric;
                if (metrics[idx2[1]] - best_metric < top2Threshold[iter]) {
                    is_miscorrection[CNidx] = true;
                }else {
                    is_miscorrection[CNidx] = false;
                }
            }
        }
        // this block enables top2 roll back

        //this block enables NN miscorrection detection====================
        if (NN4MDOnly) {
            std::vector<double> features;
            features.push_back(stddev);
            features.push_back(double (omega) / double(ChasePatternNum));
            for (int ii=0; ii<4; ii++) {
                if (ii>=omega) {
                    features.push_back(0);
                }else {
                    features.push_back(metrics[idx2[ii]] / double (mN));
                }
            }

            for (int ii=0; ii<4; ii++) {
                if (ii>=omega) {
                    features.push_back(0);
                }else {
                    auto can = candidate_codewords[idx2[ii]];
                    features.push_back(metricsDE[idx2[ii]]);
                }
            }

            double prob = nn_score(features);
            is_miscorrection[CNidx] = ((prob>0));
            phi[CNidx] = prob;
        }
        //this block enables NN miscorrection detection====================
        MLcodewords[CNidx] = candidate_codewords[best_candidate_idx];
        auto mlcw = MLcodewords[CNidx];
        std::vector<bool> foundDj(mN, false);
        for (auto VNidx : touched_positions) { //for positions where no test patterns has flipped when decoding, we know that the compteting codewrods won't exist, extrinsic info stay as 0
            int i, j;
            syn2data(CNidx, VNidx, i, j);
            //find competing codeword
            int dj = candidate_codewords[best_candidate_idx][VNidx];
            if (mAllzeroCw && dj == 1)
                miscorrectionBit += 1;

            for (int search = 1; search<omega; search++) {
                if (candidate_codewords[idx2[search]][VNidx]!=dj) { //found, if none found, extrinsic info stay as 0
                    double message =  metrics[idx2[search]] - metrics[best_candidate_idx];
                    if (dj==1) message*= -1.0;
                    message -= 2.0 * Rout[i][j];
                    extrinsic[i][j] =  message * 0.5;
                    foundDj[VNidx] = true;
                    break;
                }
            }
        }

    }

}


double GPC::SISO_block_x() {
    extrinsic = std::vector(mBufferLen, std::vector<double>(mBufferWidth, 0.0));
    Rout = Rin;
    miscorrectionBit = 0;

    makeChasePatterns();
    if (NN4MDOnly) {
        std::string filename = "../bch_nn_weights_small_p" + std::to_string(p_chase) + ".txt";
        load_neural_network(filename);
    }


    for (int iter = 0; iter < 2 * decIter; iter++) {
        MLcodewords = std::vector<BitVector>(mNumCNs);
        extrinsic = std::vector(mBufferLen, std::vector<double>(mBufferWidth, 0.0));
        is_miscorrection = std::vector<bool> (mNumCNs, false);
        phi = std::vector<double> (mNumCNs, 0.0);
        if (iter & 1) {
            for (int CNidx = mNumCNs / 2; CNidx < mNumCNs; CNidx++) {
                SISO_step_x(CNidx, iter);
            }
            //test miscorrection detection=============
            // int TN = 0;
            // int FP = 0;
            // int TP = 0;
            // for (int CNidx = mNumCNs / 2; CNidx < mNumCNs; CNidx++) {
            //     bool mis = false;
            //     for (auto bit : MLcodewords[CNidx]) {
            //         if (bit!=0) {
            //             mis = true;
            //             break;
            //         }
            //     }
            //     if (!mis && is_miscorrection[CNidx]) FP++;
            //     if (mis && !is_miscorrection[CNidx]) TN++;
            //     if (mis && is_miscorrection[CNidx]) TP++;
            // }
            // int a =1;

            //test miscorrection detection=============
        }else {
           for (int CNidx = 0; CNidx < mNumCNs/2; CNidx++) {
               SISO_step_x(CNidx, iter);
           }
            //test miscorrection detection=============
            // int TN = 0;
            // int FP = 0;
            // int TP = 0;
            // for (int CNidx = 0; CNidx < mNumCNs/2; CNidx++) {
            //     bool mis = false;
            //     for (auto bit : MLcodewords[CNidx]) {
            //         if (bit!=0) {
            //             mis = true;
            //             break;
            //         }
            //     }
            //     if (!mis && is_miscorrection[CNidx]) FP++;
            //     if (mis && !is_miscorrection[CNidx]) TN++;
            //     if (mis && is_miscorrection[CNidx]) TP++;
            // }
            // int a =1;

            //test miscorrection detection=============
        }


        if (top2 || NN4MDOnly || genie_aided) {
            double Wsum=0;
            int nonzeroCount = 0;
            for (int i=0; i<mBufferLen; i++) {
                for (int j=0; j<mBufferWidth; j++) {
                    if (iter & 1 && MLcodewords[j+mN].empty())
                        continue;
                    if (!(iter & 1) && MLcodewords[i].empty())
                        continue;
                    nonzeroCount++;
                    if (extrinsic[i][j]!=0) {
                        Wsum+=fabs(extrinsic[i][j]);
                    }
                    else {
                        Wsum += beta[iter];
                    }
                }
            }

            double Wavg = Wsum/double (nonzeroCount);

            for (int i=0; i<mBufferLen; i++) {
                for (int j=0; j<mBufferWidth; j++) {
                    if (extrinsic[i][j]!=0) {
                        extrinsic[i][j] /= Wavg;
                        if (iter & 1) {//column decoding
                            // this block does message scaling for miscorrections
                            if (is_miscorrection[j+mN])
                                extrinsic[i][j]  *= MDsclae[iter];
                            // this block does message scaling for miscorrections
                        }
                        else {
                            // this block does message scaling for miscorrections
                            if (is_miscorrection[i])
                                extrinsic[i][j]  *= MDsclae[iter];
                            // this block does message scaling for miscorrections
                        }

                    }else {
                        BitVector mlcodeword;
                        if (iter & 1) {//column decoding
                            mlcodeword = MLcodewords[j +mBufferLen];
                            if (!mlcodeword.empty()) {
                                extrinsic[i][j] = (mlcodeword[i]==0?1.0:-1.0) * beta[iter] / Wavg;
                                // this block does message scaling for miscorrections
                                if (is_miscorrection[j+mN])
                                    extrinsic[i][j]  *= MDsclae[iter];
                                // this block does message scaling for miscorrections


                            }
                        }
                        else {
                            mlcodeword = MLcodewords[i];
                            if (!mlcodeword.empty()) {//row decoding
                                extrinsic[i][j] = (mlcodeword[j]==0?1.0:-1.0) * beta[iter] / Wavg;
                                // this block does message scaling for miscorrections
                                if (is_miscorrection[i])
                                    extrinsic[i][j]  *= MDsclae[iter];
                                // this block does message scaling for miscorrections


                            }
                        }

                    }

                }
        }
        }

        bool earlyStop = true;
        if (iter>2*decIter - 1) { //last round use ML codewords, disabled now as using iBDD to terminate
            for (int i = countErrorStart; i < countErrorEnd; i++) {
                for (int j = 0; j < mBufferWidth; j++) {
                    BitVector mlcodeword;
                    if (iter & 1) {//column decoding
                        mlcodeword = MLcodewords[j +mBufferLen];
                        if (!mlcodeword.empty()) {
                            Databuffer[i][j] = mlcodeword[i];
                        }
                    }
                    else {
                        mlcodeword = MLcodewords[i];
                        if (!mlcodeword.empty()) {//row decoding
                            Databuffer[i][j] =mlcodeword[i];
                        }
                    }

                }
            }
        }else {
            for (int i=0; i<mBufferLen; i++) {
                for (int j=0; j<mBufferWidth; j++) {
                    Rout[i][j] = Rin[i][j] + extrinsic[i][j]*alpha[iter];
                }
            }





            //make hard decision and update the syndromes based on the hard decision
            for (int i = countErrorStart; i < countErrorEnd; i++) {
                for (int j = 0; j < mBufferWidth; j++) {
                    int hard_decision = Rout[i][j] < 0 ? 1 : 0;
                    if (hard_decision!=Databuffer[i][j]) {
                        auto pairs = data2syn(i,j);
                        for (auto pair : pairs) {
                            int CNidx = pair[0];
                            int VNidx = pair[1];
                            {
                                if (VNidx+ml < mBCH->primitiveLength) {
                                    for (int t = 0; t < mT; t++) {
                                        Syn[CNidx][t]^=mBCH->mGF.a_pow_tab[((2 * t + 1) * (VNidx+ml)) % mBCH->primitiveLength];
                                    }
                                }
                                if (even_check)
                                    even[CNidx]^=1;
                            }
                        }
                    }
                    Databuffer[i][j] = hard_decision;
                }

            }
            for (int CNidx=0; CNidx<mNumCNs; CNidx++) {
                zeroSyn[CNidx] = checkZeroSyn(CNidx);
            }

            for (int CNidx=0; CNidx<mNumCNs; CNidx++) {
                if (!zeroSyn[CNidx] || even[CNidx] != 0) {
                    earlyStop = false;
                    break;
                }
            }

        }



        totalErrors = 0;
        for (int i = countErrorStart; i < countErrorEnd; i++) {
            for (int j = 0; j < mBufferWidth; j++) {
                if (!mAllzeroCw && Databuffer[i][j]!=x[i][j]) {
                    totalErrors++;
                }else if (mAllzeroCw && Databuffer[i][j]!=0) {
                    totalErrors++;
                }
            }
        }


        if (earlyStop)
            break;
    }
    // return totalErrors;
    decIter = 2;
    return iBDD_block();
}



void GPC::load_neural_network(const std::string& filename) {
    mNN.load_from_file(filename);
    mUseNN = true;
}

double GPC::nn_score(const std::vector<double>& features) const {
    if (!mUseNN || !mNN.loaded) {
        throw std::runtime_error("GPC::nn_score called but no neural network is loaded");
    }
    return mNN.forward(features);
}

void GPC::original_CP_step(int CNidx, int iter) {
     //get the current codeword and reliabilities for the CNidx
    BitVector cw(mN, 0);
    std::vector<double> reliabilities(mN, 0.0);
    std::set<BitVector> seen_candidate_codewords;
    std::vector<BitVector> candidate_codewords(0);
    std::vector<double> metrics(0);
    int best_candidate_idx = -1;



    for (int VNidx = 0; VNidx < mN; VNidx++) {
        int i, j;
        syn2data(CNidx, VNidx, i, j);
        if (i >= 0) {cw[VNidx] = Databuffer[i][j]; reliabilities[VNidx] = Rout[i][j];}
        else {cw[VNidx] = 0; reliabilities[VNidx] = 1000;}
    }

    // sort
    std::vector<int> idx(mN);
    for (int i = 0; i < mN; ++i)
        idx[i] = i;
    std::sort(idx.begin(), idx.end(),
        [&reliabilities](int i1, int i2) {
            return fabs(reliabilities[i1]) < fabs(reliabilities[i2]);
        });


    double best_metric = 1e9;
    std::set<int> touched_positions;
    for (int mask = 0; mask < (1 << p_chase); ++mask) {
        //decimal to binary
        std::vector<uint8_t> filling_pattern(p_chase, 0);
        std::set<int> flip_pos;
        for (int i = 0; i < p_chase; ++i) {
            filling_pattern[i] = (mask & (1 << i)) ? 1 : 0;
        }
        BitVector test_cw = cw;
        auto test_syn = Syn[CNidx];
        uint8_t even0 = even[CNidx];
        for (int i = 0; i < p_chase; ++i) {
            //compute syndrome of the test pattern
            if (filling_pattern[i]==1) {
                test_cw[idx[i]] ^= filling_pattern[i];
                flip_pos.insert(idx[i]);
                even0 ^= 1;
                if (idx[i]+ml < mBCH->primitiveLength) {
                    for (int t = 0; t < mT; t++) {
                        test_syn[t]^=mBCH->mGF.a_pow_tab[((2 * t + 1) * (idx[i]+ml)) % mBCH->primitiveLength];
                    }
                }
            }
        }
        bool zeroSyn0 = true;
        for (auto s: test_syn) {
            if (s != 0) {
                zeroSyn0 = false;
                break;
            }
        }

        std::vector<int> ErrLoc0;
        int numErr = mBCH->decode(zeroSyn0, even0, test_syn, ErrLoc0);
        if (numErr != -1) {
            double metric = 0;
            for (auto loc: ErrLoc0) {
                test_cw[loc]^=1;
                if (flip_pos.find(loc) == flip_pos.end()) {flip_pos.insert(loc);}
                else{flip_pos.erase(loc);}
            }
            for (int i=0; i<mN; i++) {
                metric += (test_cw[i]==0? -1.0 : 1.0) * reliabilities[i]; //argmin \sum -(1-2c_i)y_i
            }

            if (seen_candidate_codewords.insert(test_cw).second) {//inserting into set success, new candidate
                candidate_codewords.push_back(test_cw);
                metrics.push_back(metric);
                if (metric < best_metric) {
                    best_metric = metric;
                    best_candidate_idx = candidate_codewords.size() - 1;
                }
                touched_positions.insert(flip_pos.begin(), flip_pos.end());
            }
        }
    }
    if (!candidate_codewords.empty()) {
        int omega = candidate_codewords.size();
        MLcodewords[CNidx] = candidate_codewords[best_candidate_idx];
        std::vector<int> idx2(omega);
        for (int i = 0; i < omega; ++i)
            idx2[i] = i;
        std::sort(idx2.begin(), idx2.end(),
            [&metrics](int i1, int i2) {
                return metrics[i1] < metrics[i2];
            });

        for (auto VNidx : touched_positions) { //for positions where no test patterns has flipped when decoding, we know that the compteting codewrods won't exist, extrinsic info stay as 0
            int i, j;
            syn2data(CNidx, VNidx, i, j);
            //find competing codeword
            int dj = candidate_codewords[best_candidate_idx][VNidx];
            for (int search = 1; search<omega; search++) {
                if (candidate_codewords[idx2[search]][VNidx]!=dj) { //found, if none found, extrinsic info stay as 0
                    double message =  metrics[idx2[search]] - metrics[best_candidate_idx];
                    if (dj==1) message*= -1.0;
                    message -= 2.0 * Rout[i][j];
                    extrinsic[i][j] = message * 0.5;
                    break;
                }
            }


        }
    }
}


double GPC::original_CP_block() {
    extrinsic = std::vector(mBufferLen, std::vector<double>(mBufferWidth, 0.0));
    Rout = Rin;
    miscorrectionBit = 0;
    is_miscorrection = std::vector<bool> (mNumCNs, false);
    phi = std::vector<double> (mNumCNs, 0.0);
    makeChasePatterns();

    for (int iter = 0; iter < 2 * decIter; iter++) {
        MLcodewords = std::vector<BitVector>(mNumCNs);
        extrinsic = std::vector(mBufferLen, std::vector<double>(mBufferWidth, 0.0));

        if (iter & 1) {
            for (int CNidx = mNumCNs / 2; CNidx < mNumCNs; CNidx++) {
                original_CP_step(CNidx, iter);
            }

        }else {
           for (int CNidx = 0; CNidx < mNumCNs/2; CNidx++) {
               original_CP_step(CNidx, iter);
           }
        }

        //original =======================
        double Wsum=0;
        int nonzeroCount = 0;
        for (int i=0; i<mBufferLen; i++) {
            for (int j=0; j<mBufferWidth; j++) {
                if (iter & 1 && MLcodewords[j+mN].empty())
                    continue;
                if (!(iter & 1) && MLcodewords[i].empty())
                    continue;
                nonzeroCount++;
                if (extrinsic[i][j]!=0) {
                    Wsum+=fabs(extrinsic[i][j]);
                }
                else {
                    Wsum += beta[iter];
                }
            }
        }

        double Wavg = Wsum/double (nonzeroCount);
        for (int i=0; i<mBufferLen; i++) {
            for (int j=0; j<mBufferWidth; j++) {
                if (extrinsic[i][j]!=0) {
                    extrinsic[i][j] /= Wavg;

                }else {
                    BitVector mlcodeword;
                    if (iter & 1) {//column decoding
                        mlcodeword = MLcodewords[j +mBufferLen];
                        if (!mlcodeword.empty()) {
                            extrinsic[i][j] = (mlcodeword[i]==0?1.0:-1.0) * beta[iter] / Wavg;
                        }
                    }
                    else {
                        mlcodeword = MLcodewords[i];
                        if (!mlcodeword.empty()) {//row decoding
                            extrinsic[i][j] = (mlcodeword[j]==0?1.0:-1.0) * beta[iter] / Wavg;
                        }
                    }
                }

            }
        }
        // //original =======================
        bool earlyStop = true;

        if (iter>2*decIter - 1) { //last round use ML codewords, disabled now as using iBDD to terminate
            for (int i = countErrorStart; i < countErrorEnd; i++) {
                for (int j = 0; j < mBufferWidth; j++) {
                    BitVector mlcodeword;
                    if (iter & 1) {//column decoding
                        mlcodeword = MLcodewords[j +mBufferLen];
                        if (!mlcodeword.empty()) {
                            Databuffer[i][j] = mlcodeword[i];
                        }
                    }
                    else {
                        mlcodeword = MLcodewords[i];
                        if (!mlcodeword.empty()) {//row decoding
                            Databuffer[i][j] =mlcodeword[i];
                        }
                    }

                }
            }
        }else {
            for (int i=0; i<mBufferLen; i++) {
                for (int j=0; j<mBufferWidth; j++) {
                    Rout[i][j] = Rin[i][j] + extrinsic[i][j]*alpha[iter];
                }
            }


            //make hard decision and update the syndromes based on the hard decision
            for (int i = countErrorStart; i < countErrorEnd; i++) {
                for (int j = 0; j < mBufferWidth; j++) {
                    int hard_decision = Rout[i][j] < 0 ? 1 : 0;
                    if (hard_decision!=Databuffer[i][j]) {
                        auto pairs = data2syn(i,j);
                        for (auto pair : pairs) {
                            int CNidx = pair[0];
                            int VNidx = pair[1];
                            {
                                if (VNidx+ml < mBCH->primitiveLength) {
                                    for (int t = 0; t < mT; t++) {
                                        Syn[CNidx][t]^=mBCH->mGF.a_pow_tab[((2 * t + 1) * (VNidx+ml)) % mBCH->primitiveLength];
                                    }
                                }
                                if (even_check)
                                    even[CNidx]^=1;
                            }
                        }
                    }
                    Databuffer[i][j] = hard_decision;
                }

            }
            for (int CNidx=0; CNidx<mNumCNs; CNidx++) {
                zeroSyn[CNidx] = checkZeroSyn(CNidx);
            }

            for (int CNidx=0; CNidx<mNumCNs; CNidx++) {
                if (!zeroSyn[CNidx] || even[CNidx] != 0) {
                    earlyStop = false;
                    break;
                }
            }

        }



        totalErrors = 0;
        for (int i = countErrorStart; i < countErrorEnd; i++) {
            for (int j = 0; j < mBufferWidth; j++) {
                if (!mAllzeroCw && Databuffer[i][j]!=x[i][j]) {
                    totalErrors++;
                }else if (mAllzeroCw && Databuffer[i][j]!=0) {
                    totalErrors++;
                }
            }
        }

        if (earlyStop)
            break;
    }
    // return totalErrors;
    decIter = 2;
    return iBDD_block();
}

