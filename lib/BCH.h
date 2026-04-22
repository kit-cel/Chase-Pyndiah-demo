#ifndef BCH_H
#define BCH_H

#include "GF_N.h"

class BCH {
public:
    GF_N mGF;

    int mM;
    int mS;
    int mK;
    int mN;
    int mT;
    int dmin;
    int mShorten;
    int mPuncture;
    int primitiveLength;
    bool mEven;
    bool mExtended;
    BitArray &mG;

    BCH(int n, int t, bool even, bool extended, std::vector<std::vector<uint8_t>> &G, int shorten=0, int puncture = 0);
    
    void encode(BitVector& m, BitVector& cw) const;
    BitVector encode(BitVector &m) const;

    
    int decode_given_syndrome_t_1(SynVec const &syn, std::vector<int> &ErrorLoc) const;
    int decode_given_syndrome_t_2(SynVec const &syn, std::vector<int> &ErrorLoc) const;
    int decode_given_syndrome_t_3(SynVec const &syn, std::vector<int> &ErrorLoc) const;

    std::tuple<bool, SynVec> compute_syndrome(const BitVector& cw) const;
    bool compute_syndrome(const BitVector& cw, SynVec& syndrome) const;

    int decode(bool zeroSyn, uint8_t weight, SynVec const &syn, std::vector<int> &ErrorLoc) const;
    std::tuple<int, std::vector<int>> decode(BitVector cw) const;
    int decode_primitive_code(SynVec const &syn, std::vector<int> &ErrorLoc) const;

    
};



#endif //BCH_H
