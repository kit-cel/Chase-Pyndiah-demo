#include "BCH.h"

BCH::BCH(int n, int t, bool even, bool extended, std::vector<std::vector<uint8_t>> &G, int shorten, int puncture): mG(G), mGF(n+1){
    mN = n;
    primitiveLength = n;
    mT = t;
    mS = mGF.S;
    mK = mN - mS * mT;
    mM = mN - mK;
    dmin = 2*t + 1;
    mEven = even;
    mExtended = extended;
    if (mExtended) { mN++; dmin++;}
    if (mEven) { mK--; dmin++;}

    mShorten = shorten;
    mK -= mShorten;
    mN -= mShorten;

    if (puncture!=0) throw std::runtime_error("puncture not yet implemented");
    mPuncture = puncture;
    mM -= mPuncture;
    mN -= mPuncture;
    mG = G;
}

BitVector BCH::encode(BitVector &m) const {BitVector cw(mN); encode(m, cw); return cw;}
void BCH::encode(BitVector& m, BitVector& cw) const {
    std::copy(m.begin(), m.end(), cw.begin());
    for (unsigned int i = mK; i < mN; i++) {
        cw[i] = 0;
        for (unsigned int j = 0; j < mK; j++) {
            cw[i] ^= (m[j]&mG[i-mK][j+mShorten]);
        }
    }
}




int BCH::decode_given_syndrome_t_1(SynVec const &syn, std::vector<int> &ErrorLoc) const {
    //for t=1, no actual decoding is needed, just use the log table to find the error location
    ErrorLoc.push_back(mGF.a_log_tab[syn[0]]);
    return 1;
}

int BCH::decode_given_syndrome_t_2(SynVec const &syn, std::vector<int> &ErrorLoc) const {
    //for t=2
    uint16_t S1 = syn[0];
    uint16_t S3 = syn[1];
    if (S1==0) {
        return -1;
    }

    uint16_t S1_log = mGF.a_log_tab[S1];
    uint16_t S1_sqr_log = mGF.mod_s(2 * S1_log);
    uint16_t S1_cub_log = mGF.mod_s(S1_sqr_log +S1_log);
    uint16_t S1_cub = mGF.a_pow_tab[S1_cub_log];
    uint16_t D3 = (S1_cub^S3);

    if (D3==0) {
        ErrorLoc.push_back(mGF.a_log_tab[syn[0]]);
        return 1;
    }
    uint16_t D3_log = mGF.a_log_tab[D3];

    //Chien search, but will not use as too complex
    // uint16_t l = mGF.gf_div(D3,S1);
    //
    // for (uint16_t x=0; x< mGF.N; x++) {
    //     auto tmp1 = mGF.gf_mul(S1,x);
    //     auto tmp2 = mGF.gf_mul(l,mGF.gf_sqr(x));
    //     uint16_t tmp = (tmp1^tmp2);
    //     if ((tmp^1)==0) {
    //         ErrorLoc.push_back(mGF.a_log_tab[mGF.gf_div(1,x)]);
    //         numErr++;
    //     }
    // }

    //ELP is Lambda(x) = 1+S_1 x + D_3/S_1 x^2, now solve for its root
    uint16_t c = mGF.mod_s(D3_log + primitiveLength - S1_cub_log); // in log

    int numErr = mGF.DP2[c][0];
    if (numErr == 0) {
        return -1;
    }
    for (unsigned int i = 0; i < numErr; i++) {
        uint16_t y = mGF.DP2[c][i+1];
        auto loc = (y!=0) ? mGF.mod_s( y + S1_log) : 0;
        ErrorLoc.push_back(loc);
    }
    return numErr;
}

int BCH::decode_given_syndrome_t_3(SynVec const &syn, std::vector<int> &ErrorLoc)const {
    uint16_t S1 = syn[0];
    uint16_t S3 = syn[1];
    uint16_t S5 = syn[2];

    uint16_t S1_log = mGF.a_log_tab[S1];
    uint16_t S1_sqr_log = mGF.mod_s(2 * S1_log);
    uint16_t S1_cub_log = mGF.mod_s(S1_sqr_log +S1_log);
    uint16_t S1_cub = S1!=0? mGF.a_pow_tab[S1_cub_log]:0;
    uint16_t D3 = (S1_cub^S3);


    uint16_t l2 = mGF.gf_div((mGF.gf_mul(mGF.gf_sqr(S1),S3)^S5),D3);
    uint16_t l3 = D3^mGF.gf_mul(S1,l2);
    if (l3==0) {
        return decode_given_syndrome_t_2(syn, ErrorLoc);
    }
    if (S1==0 && S5==0) {
        int numErr=0;
        for (uint16_t x=1; x< mGF.N; x++) {
            uint16_t tmp = mGF.gf_mul(mGF.gf_mul(x,mGF.gf_sqr(x)),S3);
            if ((tmp^1)==0) {
                ErrorLoc.push_back(mGF.a_log_tab[mGF.gf_div(1,x)]);
                numErr++;
            }
        }
        if (numErr <2) numErr = -1;
        return numErr;
    }
    uint16_t eta = mGF.gf_sqr(S1)^l2;
    if (eta!=0) {
        uint16_t delta = mGF.gf_mul(S1,l2)^l3;
        uint16_t eta_sqrt = mGF.a_sqr_root[eta];
        uint16_t c = mGF.gf_div(delta, mGF.gf_mul(eta, eta_sqrt));
        int numErr = mGF.DP3[c][0];
        if (numErr == 0) {return -1;}

        for (unsigned int i = 0; i < numErr; i++) {
            uint16_t y = mGF.DP3[c][i+1];
            uint16_t loc = mGF.gf_mul(y, mGF.a_sqr_root[eta])^S1;
            ErrorLoc.push_back(mGF.a_log_tab[loc]);
        }
        return numErr;
    }
    uint16_t p = mGF.gf_div(l2,l3);
    uint16_t q = mGF.gf_div(S1,l3);
    uint16_t r = mGF.gf_inv(l3);
    eta = mGF.gf_sqr(p)^q;
    uint16_t eta_sqrt = mGF.a_sqr_root[eta];
    uint16_t delta = mGF.gf_mul(p,q)^r;
    uint16_t c = mGF.gf_div(delta, mGF.gf_mul(eta, eta_sqrt));
    int numErr = mGF.DP3[c][0];
    if (numErr == 0) {return -1;}

    for (unsigned int i = 0; i < numErr; i++) {
        uint16_t y = mGF.DP3[c][i+1];
        uint16_t loc = mGF.gf_mul(y, mGF.a_sqr_root[eta])^p;
        loc = (loc==1) ? 0 : primitiveLength - mGF.a_log_tab[loc];
        ErrorLoc.push_back(loc);
    }
    return numErr;


}

std::tuple<bool, SynVec> BCH::compute_syndrome(const BitVector& cw) const {
    SynVec syn;
    bool zeroSyn = compute_syndrome(cw, syn);
    return std::make_tuple(zeroSyn, syn);
}

bool BCH::compute_syndrome(const BitVector &cw, std::vector<uint16_t> &syn) const {
    /*for our high rate BCH codes, syndrome is t GF elements
     *For primitive codes, the PCM (t rows, n columns) is:
     * 1     a      ...  a^{n-2}          a^{n-1}
     * 1    a^2     ...  a^{2(n-2)}       a^{2(n-1)}
     *              ...
     * 1    a^{2t}  ...  a^{2t(n-2)}      a^{2t(n-1)}
     *where a is the primitive element
     *For extended codes, the last bit is not used for calculating syndrome, but used in
     *post-processing
     *For even-weight subcodes, syndrome is calculated in the same way, also post-processing needed
     *to check if even weight is true
     */
    if (syn.empty()) syn = SynVec(mT,0);
    bool zeroSyn = true;
    for (unsigned int i = 0; i < mT; i++) {
        syn[i]=0;
        for (unsigned int j = 0; j < mN; j++) {
            if (cw[j]==1 && j+mShorten<primitiveLength) {
                syn[i] ^= mGF.a_pow_tab[((2 * i + 1) * (j+mShorten)) % primitiveLength];
            }
        }
        if (syn[i]!=0) zeroSyn = false;
    }
    //zero-syndrome indicates  valid codeword for the primitive code case, even-weight condition needs to be checked additionally
    return zeroSyn;
}

std::tuple<int, std::vector<int>> BCH::decode(BitVector cw) const {
    auto [zeroSyn, syn] = compute_syndrome(cw);
    int count1 = std::accumulate(cw.begin(), cw.end(), 0);
    std::vector<int> ErrorLoc;
    int numErr = decode(zeroSyn, count1%2, syn, ErrorLoc);
    return std::make_tuple(numErr, ErrorLoc);
}


int BCH::decode(bool zeroSyn, uint8_t weight, SynVec const &syn, std::vector<int> &ErrorLoc) const {
    int numErr;
    if (!zeroSyn) {
        numErr = decode_primitive_code(syn, ErrorLoc);
    }else {
        numErr = 0;
    }
    if (numErr==-1) return -1;

    for (auto e : ErrorLoc) {
        if (e<mShorten) {//errors cannot occur on the shortended positions
            return -1;
        }
    }
    for (int i=0; i<numErr; i++){ErrorLoc[i]-=mShorten;}
    if (!mEven && !mExtended) return numErr;

    uint8_t newWeight = (weight + numErr)%2;
    if (mExtended) {
        //for extended codes, the last bit is responsible for making the codeword even weight
        //Therefore, if the current weight is odd, it means the last bit is in error
        //However, if adding this error makes the total nubmer of errors exceed t, it means this is miscorrection
        if (newWeight!=0) {
            if (numErr==mT) {return -1;}
            numErr++;
            ErrorLoc.push_back(mN-1);
            return numErr;
        }
    }
    if (mEven) {
        //for even-weight subcode, simply check if the even weight condition is fulfilled
        if (newWeight == 1) {
            return -1;
        }
    }

    return numErr;
}


int BCH::decode_primitive_code(SynVec const &syn, std::vector<int> &ErrorLoc) const {
    //only used when syndrome is nonzero
    if (mT==1) {
        return decode_given_syndrome_t_1(syn, ErrorLoc);
    }
    if (mT==2) {
        return decode_given_syndrome_t_2(syn, ErrorLoc);
    }
    if (mT==3) {
        return decode_given_syndrome_t_3(syn, ErrorLoc);
    }
    if (mT>3) {
        throw std::invalid_argument("BCH::decode: not implemented yet");
    }
    throw std::invalid_argument("invalid t");
}
