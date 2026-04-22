#ifndef GF_N_H
#define GF_N_H

#include "helpers.h"

class GF_N {
public:
    int N;
    int S;
    explicit GF_N(int n);
    ~GF_N();

    const int mPrimPolyIdxOff = 1; //only support field bigger than or equal to 32
    const std::vector<int> mPrimPoly{0x1, 0x7, 0xb, 0x13, 0x25, 0x43, 0x89, 0x11d, 0x211,
                                              0x409, 0x805, 0x1053, 0x201b,
                                              0x4443, 0x8003};

    uint16_t *a_pow_tab;
    uint16_t *a_log_tab;
    uint16_t *a_sqr_root;
    uint16_t **DP2;
    uint16_t **DP3;
    
    /* Galois field basic operations: multiply, divide, inverse, etc. */
    inline int gf_mul(int a, int b) const{
        //a&&b is 0 if any of a or b is 0, then the multiplication result is 0
        //otherwise multiplication is the same as adding the power
        return (a && b) ? a_pow_tab[mod_s(a_log_tab[a] +a_log_tab[b])] : 0;
    }

    inline int gf_sqr(int a) const{
        //square of zero element is still zero
        //for nonzero elements, log_table converts a to x, where a = alpha^x
        //Then, power_table gives the element which equals alpha^{2x}
        return a ? a_pow_tab[mod_s(2 * a_log_tab[a])] : 0;
    }

    inline int gf_div(int a, int b) const{
        return a ? a_pow_tab[mod_s(a_log_tab[a] + N-1 - a_log_tab[b])] : 0;
    }

    inline int gf_inv( int a) const{
        return a_pow_tab[N -1 - a_log_tab[a]];
    }

    /*
     * shorter and faster modulo function, only works when v < 2N.
     */
    inline int mod_s(int v) const {
        const int n = N-1;
        return (v < n) ? v : (v - n);
    }

    inline int deg(int poly) {
        /* polynomial degree is the most-significant bit index */
        return FLS(poly) - 1;
    }

    inline int FLS(uint32_t x) {
        int r = 0;
        if (x >= (1 << 16)) {
            r += 16;
            x >>= 16;
        }
        if (x >= (1 << 8)) {
            r += 8;
            x >>= 8;
        }
        if (x >= (1 << 4)) {
            r += 4;
            x >>= 4;
        }
        if (x >= (1 << 2)) {
            r += 2;
            x >>= 2;
        }
        if (x >= (1 << 1)) {
            r += 1;
            x >>= 1;
        }
        return r + x;
    }

    int build_gf_tables(int poly) ;
    int build_DP2();
    int build_DP3();
    int build_sqrt();
};



#endif //GF_N_H
