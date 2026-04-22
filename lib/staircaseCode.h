#ifndef STAIRCASECODES_H
#define STAIRCASECODES_H
#include "GPC.h"


class staircaseCode : public GPC{
public:
    staircaseCode(const std::shared_ptr<BCH>& code, int numBlocks) : GPC(code) {
        initialize(numBlocks);
    }

    void calculate_rate() override{
        mRate = 1 - 2.0*(1.0 - (double (mK) / double (mN)));
    }

    void initialize(int numBlocks) {
        if (mN%2 != 0) throw std::invalid_argument("component code length must be even");
        mBufferWidth = mN / 2;
        mBufferLen = numBlocks * mBufferWidth;
        mNumCNs = mBufferLen;
        int skipBegin = 10;
        int skipEnd = 10;
        countErrorStart = skipBegin * mBufferWidth;
        countErrorEnd = mBufferLen - skipEnd * mBufferWidth;
        calculate_rate();
        BufferSize = double((countErrorEnd- countErrorStart) * mBufferWidth);
        mNumBlock = numBlocks;
    }
    void build_zipper_buffer(BitArray &b);
    int mNumBlock;



    void syn2data(int CNidx, int VNidx, int &i, int&j)override {
        //use the zipper code structure, then the CNidx is the row index of the data buffer
        i = CNidx;
        j = VNidx - mN/2;
        if (VNidx<mN/2) {
            int blockIdx = CNidx / (mN/2);
            int inBlockCNidx = CNidx - blockIdx*(mN/2);
            i = (mN/2) * (blockIdx-1) + VNidx;
            j = inBlockCNidx;
        }

    }
    void syn2syn(int CNidx, int VNidx, int &otherCN, int &otherVN)override {
        int blockIdx = CNidx / (mN/2);
        int inBlockCNidx = CNidx - blockIdx*(mN/2);

        if (VNidx<mN/2) {
            otherVN = inBlockCNidx + (mN/2);
            otherCN = (mN/2) * (blockIdx-1) + VNidx;

        }else {
            otherVN = inBlockCNidx;
            otherCN = (mN/2) * (blockIdx+1) + VNidx - mN/2;
        }

    }

    std::vector<std::vector<int>> data2syn(int i, int j) override {
        std::vector<std::vector<int>> pairs;
        int blockIdx = i / (mN/2);
        int inBlockCNidx = i - blockIdx*(mN/2);
        int x = i;
        int y = j + mN/2;
        pairs.push_back({x,y});

        x = (blockIdx+1)*(mN/2) + j;
        y = inBlockCNidx;
        if (x<mBufferLen) {
            pairs.push_back({x,y});
        }
        return pairs;
    }


    void setValidBufferRange() override{
        countErrorStart = windowLen*mBufferWidth;
        countErrorEnd = mBufferLen - windowLen*mBufferWidth;
    }
};



#endif //STAIRCASECODES_H
