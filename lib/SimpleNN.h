// #pragma once
// #include <vector>
// #include <string>
// #include <fstream>
// #include <sstream>
// #include <cmath>
// #include <stdexcept>
//
// class SimpleNN {
// public:
//     bool loaded;
//     void load_from_file(const std::string& filename) {
//         std::ifstream f(filename);
//         if (!f.is_open()) throw std::runtime_error("Cannot open file");
//
//         std::string line;
//
//         while (std::getline(f, line)) {
//             if (line == "fc1.weight") fc1_w = read_matrix(f, 8, 10);
//             else if (line == "fc1.bias") fc1_b = read_vector(f, 8);
//             else if (line == "fc11.weight") fc11_w = read_matrix(f, 4, 8);
//             else if (line == "fc11.bias") fc11_b = read_vector(f, 4);
//             else if (line == "fc2.weight") fc2_w = read_matrix(f, 1, 4);
//             else if (line == "fc2.bias") fc2_b = read_vector(f, 1);
//         }
//         loaded = true;
//     }
//
//     double forward(const std::vector<double>& x_raw) const {
//         auto x1 = linear(x_raw, fc1_w, fc1_b);
//         relu(x1);
//
//         auto x2 = linear(x1, fc11_w, fc11_b);
//         relu(x2);
//
//         auto x3 = linear(x2, fc2_w, fc2_b);
//
//         return sigmoid(x3[0]);
//     }
//
// private:
//     std::vector<std::vector<double>> fc1_w, fc11_w, fc2_w;
//     std::vector<double> fc1_b, fc11_b, fc2_b;
//
//     static std::vector<std::vector<double>> read_matrix(std::ifstream& f, int rows, int cols) {
//         std::vector<std::vector<double>> mat(rows, std::vector<double>(cols));
//         std::string line;
//
//         for (int i = 0; i < rows; ++i) {
//             std::getline(f, line);
//             std::stringstream ss(line);
//             for (int j = 0; j < cols; ++j) {
//                 ss >> mat[i][j];
//             }
//         }
//         return mat;
//     }
//
//     static std::vector<double> read_vector(std::ifstream& f, int size) {
//         std::vector<double> vec(size);
//         std::string line;
//         std::getline(f, line);
//         std::stringstream ss(line);
//
//         for (int i = 0; i < size; ++i) {
//             ss >> vec[i];
//         }
//         return vec;
//     }
//
//     static std::vector<double> linear(
//         const std::vector<double>& x,
//         const std::vector<std::vector<double>>& W,
//         const std::vector<double>& b
//     ) {
//         std::vector<double> out(W.size(), 0.0);
//
//         for (size_t i = 0; i < W.size(); ++i) {
//             double sum = b[i];
//             for (size_t j = 0; j < x.size(); ++j) {
//                 sum += W[i][j] * x[j];
//             }
//             out[i] = sum;
//         }
//         return out;
//     }
//
//     static void relu(std::vector<double>& x) {
//         for (auto& v : x) v = std::max(0.0, v);
//     }
//
//     static double sigmoid(double x) {
//         return 1.0 / (1.0 + std::exp(-x));
//     }
// };
#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cmath>

struct SimpleNN {
    int input_dim = 0;
    int hidden_dim = 0;


    std::vector<std::vector<double>> fc1_weight; // [hidden_dim][input_dim]
    std::vector<double> fc1_bias;                // [hidden_dim]

    std::vector<double> fc2_weight;              // [hidden_dim]
    double fc2_bias = 0.0;

    bool loaded = false;

    static double sigmoid(double x) {
        if (x >= 0.0) {
            double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        } else {
            double z = std::exp(x);
            return z / (1.0 + z);
        }
    }


    double forward(const std::vector<double>& x_raw) const {
        if (!loaded) {
            throw std::runtime_error("SimpleNN::forward called before weights were loaded");
        }
        if ((int)x_raw.size() != input_dim) {
            throw std::runtime_error("SimpleNN::forward input size mismatch");
        }

        std::vector<double> x = x_raw;

        std::vector<double> hidden(hidden_dim, 0.0);
        for (int h = 0; h < hidden_dim; ++h) {
            double sum = fc1_bias[h];
            for (int i = 0; i < input_dim; ++i) {
                sum += fc1_weight[h][i] * x[i];
            }
            hidden[h] = sum; // identity activation
        }

        double logit = fc2_bias;
        for (int h = 0; h < hidden_dim; ++h) {
            logit += fc2_weight[h] * hidden[h];
        }

        return logit;
    }

    int predict(const std::vector<double>& x_raw, double threshold = 0.5) const {
        return forward(x_raw) >= threshold ? 1 : 0;
    }

    void load_from_file(const std::string& filename) {
        std::ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Cannot open NN weight file: " + filename);
        }

        in >> input_dim >> hidden_dim;
        if (!in) {
            throw std::runtime_error("Failed to read input_dim and hidden_dim");
        }

        std::string tag;

        in >> tag;
        if (tag != "fc1_weight") throw std::runtime_error("Expected tag: fc1_weight");
        fc1_weight.assign(hidden_dim, std::vector<double>(input_dim));
        for (int h = 0; h < hidden_dim; ++h) {
            for (int i = 0; i < input_dim; ++i) {
                in >> fc1_weight[h][i];
            }
        }

        in >> tag;
        if (tag != "fc1_bias") throw std::runtime_error("Expected tag: fc1_bias");
        fc1_bias.resize(hidden_dim);
        for (int h = 0; h < hidden_dim; ++h) in >> fc1_bias[h];

        in >> tag;
        if (tag != "fc2_weight") throw std::runtime_error("Expected tag: fc2_weight");
        fc2_weight.resize(hidden_dim);
        for (int h = 0; h < hidden_dim; ++h) in >> fc2_weight[h];

        in >> tag;
        if (tag != "fc2_bias") throw std::runtime_error("Expected tag: fc2_bias");
        in >> fc2_bias;

        if (!in) {
            throw std::runtime_error("Failed while parsing NN weight file");
        }

        loaded = true;
    }
};