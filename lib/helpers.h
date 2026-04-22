#ifndef BCH2025_HELPERS_H
#define BCH2025_HELPERS_H

#pragma once

#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <memory>
#include "random"

#include <fstream>
#include <sstream>
#include <cstdint>
#include <chrono>
#include "set"
#include <iomanip>
#include <omp.h>
#include <mutex>
#include <atomic>
#include "unordered_set"


typedef std::vector<uint8_t> BitVector;
typedef std::vector<std::vector<uint8_t>> BitArray;
typedef std::vector<uint16_t> SynVec;
typedef std::vector<std::vector<uint16_t>> SynArray;




// Function to display the matrix
template <typename T>
void displayMatrix(const std::vector<std::vector<T>> &matrix)
{
    for (const auto &row : matrix)
    {
        for (int val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void displayVector(const std::vector<T> &vec)
{
    for (const T &elem : vec)
    {
        std::cout << int(elem) << " ";
    }
    std::cout << std::endl;
}


template <typename T>
void displayArray(const T*array, int size)
{
    for (int i = 0; i < size; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}


inline bool readArrayFromFile(const std::string &filename, std::vector<std::vector<uint8_t>> &arr) {
    std::ifstream file(filename);
    if (!file) {
        if (filename.rfind("../", 0) == 0) { // starts with "../"
            std::string filenamenew = filename.substr(3);  // remove "../"
            file.open(filenamenew);
        }

    }

    if (!file) {
        std::cerr << "Error opening file !" << filename<<std::endl;
        return false;
    }
    std::vector<std::vector<uint8_t>> tempArr;
    std::string line;
    int rowCount = 0, colCount = -1;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<uint8_t> row;
        int value;

        while (iss >> value) {
            row.push_back(static_cast<uint8_t>(value));
        }

        if (colCount == -1) colCount = row.size();  // Set column count based on first row
        else if (row.size() != colCount) {
            std::cerr << "Inconsistent column count in file!" << std::endl;
            return false;
        }

        tempArr.push_back(row);
        rowCount++;
    }

    file.close();

    // Assign the results
    arr = tempArr;
    return true;
}

inline void readG(std::vector<std::vector<uint8_t>> &G, unsigned int n, unsigned int t, bool even, bool extended) {
//    std::cout << "working directory"<<std::filesystem::current_path() << std::endl;
    std::ostringstream filename;
    if (!even && !extended) {
        filename  <<"../Gmatrices/primitive/"<< n << "_" << t << ".txt";
        std::cout<<filename.str()<<"\n";
    }else if (even) {
        filename  <<"../Gmatrices/even/"<< n << "_" << t << ".txt";
    }else if (extended) {
        filename  <<"../Gmatrices/extended/"<< (n+1) << "_" << t << ".txt";
    }

    readArrayFromFile(filename.str(), G);
}

inline double Qc(double x) {
    return 1-0.5 * std::erfc( x / std::sqrt(2.0) );
}


inline std::vector<double> calculate_parameters(double EbNo_dB, double rate, double T = 0){
    /*caluate the key parameters of a given SNR for code with certain rate
     * #0 EbNo (dB)
     * #1 EsNo (dB)
     * #2 delta
     * #3 gaussian noise variance
     * #4 erasure probability for erasure threshold T
     * */
    std::vector<double> parameters;
    parameters.push_back(EbNo_dB);
    double EbNo_num = pow(10, EbNo_dB / 10.0);
    double SNR = EbNo_num * rate;
    double EsNo_dB = 10.0* log10(SNR);
    parameters.push_back(EsNo_dB);

    double z = sqrt(SNR);
    double delta = 0.5 * erfc(z*(T+1));
    parameters.push_back(delta);

    double sigma_2 = 1.0 / (2.0*SNR);

    parameters.push_back(sigma_2);

    double erausreProb = 1-0.5*(erfc((T+1.0)*z) + erfc((T-1.0)*z));
    parameters.push_back(erausreProb);

    double cross_over = 0.5 * erfc(z);

    return parameters;
}

template <typename T>
void removeValue(std::vector<T>& vec, T value) {
    auto it = std::find(vec.begin(), vec.end(), value);
    if (it != vec.end()) {
        vec.erase(it);  // Remove the element if found
    }
}





inline int hamming_weight_vec(const std::vector<uint8_t>& v) {
    int cnt = 0;
    for (uint8_t b : v) cnt += (b & 1);
    return cnt;
}

// XOR v2 into v1 (in place)
inline void xor_into(std::vector<uint8_t>& v1, const std::vector<uint8_t>& v2) {
    size_t m = v1.size();
    for (size_t i = 0; i < m; ++i) v1[i] ^= v2[i];
}

// Compute x_particular and nullspace basis vectors (v_j for each free var)
static void gaussian_gf2_with_basis(std::vector<std::vector<uint8_t>> A,
                                    std::vector<uint8_t> b,
                                    std::vector<uint8_t>& x_part,
                                    std::vector<std::vector<uint8_t>>& basis,
                                    bool &consistent)
{
    int n = (int)A.size();
    int m = (int)A[0].size();
    consistent = true;

    x_part.assign(m, 0);
    std::vector<int> pivot_col(n, -1);
    int row = 0;

    // Forward elimination (produce row-echelon)
    for (int col = 0; col < m && row < n; ++col) {
        int sel = -1;
        for (int i = row; i < n; ++i) if (A[i][col] & 1) { sel = i; break; }
        if (sel == -1) continue;
        std::swap(A[sel], A[row]);
        std::swap(b[sel], b[row]);
        pivot_col[row] = col;
        for (int i = row + 1; i < n; ++i) {
            if (A[i][col] & 1) {
                for (int c = col; c < m; ++c) A[i][c] ^= A[row][c];
                b[i] ^= b[row];
            }
        }
        ++row;
    }

    // Consistency check
    for (int i = row; i < n; ++i) {
        bool all_zero = true;
        for (int c = 0; c < m; ++c) if (A[i][c] & 1) { all_zero = false; break; }
        if (all_zero && (b[i] & 1)) { consistent = false; return; }
    }

    // Identify free variables
    std::vector<uint8_t> is_free(m, 1);
    for (int i = 0; i < row; ++i)
        if (pivot_col[i] != -1) is_free[pivot_col[i]] = 0;
    std::vector<int> free_list;
    for (int c = 0; c < m; ++c) if (is_free[c]) free_list.push_back(c);

    int f = (int)free_list.size();

    // Compute particular solution x_part with all free vars = 0 (back-substitute)
    for (int i = row - 1; i >= 0; --i) {
        int pc = pivot_col[i];
        if (pc == -1) continue;
        uint8_t rhs = b[i];
        for (int c = pc + 1; c < m; ++c)
            if (A[i][c] & 1) rhs ^= x_part[c];
        x_part[pc] = rhs;
    }

    // For each free variable j, compute basis vector v_j:
    // set that free var = 1 and other free = 0, back-substitute to get full vector,
    // then v_j = x_when_free_j_1 XOR x_part.
    basis.clear();
    basis.resize(f, std::vector<uint8_t>(m, 0));

    for (int idx = 0; idx < f; ++idx) {
        int free_col = free_list[idx];
        std::vector<uint8_t> x(m, 0);
        x[free_col] = 1;
        for (int i = row - 1; i >= 0; --i) {
            int pc = pivot_col[i];
            if (pc == -1) continue;
            uint8_t rhs = b[i];
            for (int c = pc + 1; c < m; ++c)
                if (A[i][c] & 1) rhs ^= x[c];
            x[pc] = rhs;
        }
        // basis[idx] = x XOR x_part
        for (int j = 0; j < m; ++j) basis[idx][j] = x[j] ^ x_part[j];
    }
}



#endif //BCH2025_HELPERS_H
