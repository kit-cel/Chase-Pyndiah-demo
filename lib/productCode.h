#ifndef PRODUCTCODES_H
#define PRODUCTCODES_H

#include "GPC.h"
#include "unordered_set"

class productCode : public GPC{
public:
    productCode(const std::shared_ptr<BCH>& code) : GPC(code) {
        mBufferLen = mN;
        mBufferWidth = mN;
        mNumCNs = 2*mN;
        BufferSize = double(mBufferLen * mBufferWidth);
        calculate_rate();
        setValidBufferRange();
    }


    void calculate_rate() override {
        mRate = double(mK)/double(mN);
        mRate*=mRate;
    }

    std::vector<std::vector<int>> data2syn(int i, int j)override {
        std::vector pairs(2, std::vector<int>(2,0));
        pairs[0][0] = i;
        pairs[0][1] = j;
        pairs[1][0] = j + mN;
        pairs[1][1] = i;
        return pairs;
    }

    void syn2syn(int CNidx, int VNidx, int &otherCN, int &otherVN)override {
        otherCN = CNidx<mN? VNidx +mN : VNidx;
        otherVN = CNidx<mN? CNidx : CNidx - mN;
    }
    void syn2data(int CNidx, int VNidx, int &i, int&j)override {
        i = CNidx<mN? CNidx: VNidx;
        j = CNidx<mN? VNidx: CNidx -mN;
    }

    void setValidBufferRange() override {
        countErrorEnd = mN;
        countErrorStart = 0;
    }

};



#endif //PRODUCTCODES_H
