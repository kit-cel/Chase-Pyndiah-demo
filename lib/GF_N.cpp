#include "GF_N.h"


GF_N::GF_N(int n) {
    if ((n & (n - 1)) != 0) throw "GF_N::GF_N: n must be power of 2!";
    N = n;
    S = log2(N);
    int poly = mPrimPoly[S - mPrimPolyIdxOff];

    a_pow_tab = (uint16_t *)malloc(sizeof(uint16_t *) * N);
    a_log_tab = (uint16_t *)malloc(sizeof(uint16_t *) * N);
    DP2 = (uint16_t **)malloc(N * sizeof(uint16_t *));
    for (unsigned int i = 0; i < N; i++) {
        DP2[i] = (uint16_t *)malloc(3 * sizeof(uint16_t));
    }

    DP3 = (uint16_t **)malloc(N * sizeof(uint16_t *));
    for (unsigned int i = 0; i < N; i++) {
        DP3[i] = (uint16_t *)malloc(4 * sizeof(uint16_t));
    }
    build_gf_tables(poly);
    build_sqrt();
    build_DP2();
    build_DP3();

}



int GF_N::build_gf_tables(int poly) {
    int i, x = 1;
    const int k = 1 << deg(poly);

    if (k != (1u << S)) {
        throw std::invalid_argument("GF_N::build_gf_tables: k must be power of 2");
    }
    //The polynomial x is assumed to be the primitive element alpha, start with it
    for (i = 0; i < N-1; i++) {
        //record each entry of the power table
        a_pow_tab[i] = x;
        //log table is just the reverse of power table
        a_log_tab[x] = i;
        if (i && (x == 1)) //the chosen prim. poly. has to make sure that x is the prim. ele.
            throw std::invalid_argument("polynomial is not primitive (a^i=1 with 0<i<2^m-1)");
        x <<= 1; //left shift x by 1 position, i.e., multiplying the current polynomial with x, i.e., alpha^i -> alpha ^ {i+1}
        //If the degree of the current polynomial is equal to the prim. poly. degree,
        //then x = x + primitive_poly (binary addition of the coefficients), as the primtive poly is like 0, returning to field elements
        if (x & k)
            x ^= poly;
    }
    a_pow_tab[N-1] = 1;
    a_log_tab[0] = 0;
    return 0;
}


GF_N::~GF_N() {
    free(a_pow_tab);
    free(a_log_tab);
    free(a_sqr_root);
    for (unsigned int i = 0; i < N; i++) {
        free(DP2[i]);
        free(DP3[i]);
    }
    free(DP2);
    free(DP3);
}

int GF_N::build_DP2() {
    //DP2 stores the root for x^2+x+c=0 for c ranging over all elements in the GF
    //the case where c=0 is excluded
    for (uint16_t c = 0; c < N; c++) {
        DP2[c][0] = 0;
        for (uint16_t d = 1; d < 3; d++) {
            DP2[c][d] = 65535;
        }
    }
    for (uint16_t c = 1; c < N; c++) {
        int counter = 0;
        for (uint16_t x = 0; x <N; x++) {
            auto temp = gf_sqr(x)^x;
            if (temp == c) {
                DP2[a_log_tab[c]][counter+1] = a_log_tab[x];

                counter++;
            }
        }
        DP2[a_log_tab[c]][0] = counter;


    }
    return 0;
}


int GF_N::build_DP3() {
    //DP2 stores the root for x^2+x+c=0 for c ranging over all elements in the GF
    //the case where c=0 is excluded
    for (uint16_t c = 0; c < N; c++) {
        DP3[c][0] = 0;
        for (uint16_t d = 1; d < 4; d++) {
            DP3[c][d] = 65535;
        }
    }
    for (uint16_t c = 1; c < N; c++) {
        int counter = 0;
        for (uint16_t x = 0; x <N; x++) {
            auto temp = (gf_mul(gf_sqr(x),x)^x);
            if (temp == c) {
                DP3[c][counter+1] = x;
                counter++;
            }
        }
        if (counter==3)//TODO:is this correct? how many roots are allowed?
            DP3[c][0] = counter;


    }
    return 0;
}

int GF_N::build_sqrt() {
    a_sqr_root = (uint16_t *)malloc(sizeof(uint16_t) * N);
    for (uint16_t c = 0; c < N; c++) {
        for (uint16_t x = 0; x < N; x++) {
            if (gf_sqr(x)==c) {
                a_sqr_root[c] = x;
            }
        }
    }
    return 0;
}
