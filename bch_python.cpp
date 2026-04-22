#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "lib/BCH.h"

#include <random>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

// Keep your existing helper
void makpatterns(
    int p_chase,
    int &ChaseIpatternNum,
    int &ChaseIIpatternNum,
    std::vector<BitVector> &ChaseIpatterns,
    std::vector<BitVector> &ChaseIIpatterns
) {
    ChaseIIpatternNum = 1 << p_chase;
    ChaseIIpatterns = std::vector<BitVector>(ChaseIIpatternNum);

    for (int mask = 0; mask < (1 << p_chase); ++mask) {
        BitVector filling_pattern(p_chase, 0);
        for (int i = 0; i < p_chase; ++i) {
            filling_pattern[i] = (mask & (1 << i)) ? 1 : 0;
        }
        ChaseIIpatterns[mask] = filling_pattern;
    }

    ChaseIpatternNum = 1 + p_chase
                     + p_chase * (p_chase - 1) / 2
                     + p_chase * (p_chase - 1) * (p_chase - 2) / 6;

    ChaseIpatterns = std::vector<BitVector>(ChaseIpatternNum);
    ChaseIpatterns[0] = BitVector(p_chase, 0);

    for (int i = 0; i < p_chase; ++i) {
        BitVector filling_pattern(p_chase, 0);
        filling_pattern[i] = 1;
        ChaseIpatterns[1 + i] = filling_pattern;
    }

    int idx = 1 + p_chase;
    for (int i = 0; i < p_chase; ++i) {
        for (int j = i + 1; j < p_chase; ++j) {
            BitVector filling_pattern(p_chase, 0);
            filling_pattern[i] = 1;
            filling_pattern[j] = 1;
            ChaseIpatterns[idx++] = filling_pattern;
        }
    }

    for (int i = 0; i < p_chase; ++i) {
        for (int j = i + 1; j < p_chase; ++j) {
            for (int k = j + 1; k < p_chase; ++k) {
                BitVector filling_pattern(p_chase, 0);
                filling_pattern[i] = 1;
                filling_pattern[j] = 1;
                filling_pattern[k] = 1;
                ChaseIpatterns[idx++] = filling_pattern;
            }
        }
    }
}

struct SampleResult {
    double is_miscorrection = 0.0;
    double stddev = 0.0;   // added
    double convertRatio = 0.0;
    std::vector<double> best_five_metrices = std::vector<double>(5, 0.0);
    std::vector<double> best_five_distrED = std::vector<double>(5, 0.0);
    std::vector<double> smallest_eight_abs_y = std::vector<double>(10, 0.0);

    std::vector<double> to_vector() const {
        std::vector<double> out;
        out.reserve(21);
        out.push_back(is_miscorrection);
        out.push_back(stddev);
        out.push_back(convertRatio);
        out.insert(out.end(), best_five_metrices.begin(), best_five_metrices.end());
        out.insert(out.end(), best_five_distrED.begin(), best_five_distrED.end());
        // out.insert(out.end(), smallest_eight_abs_y.begin(), smallest_eight_abs_y.end());
        return out;
    }
};

class BCHBatchGenerator {
public:
    BCHBatchGenerator(
        int n = 255,
        int t = 2,
        bool extend = true,
        int shorten = 0,
        bool even = false,
        int p_chase = 5,
        double EsNo_dB_min = 3.1,
        double EsNo_dB_max = 3.1
    )
        : n_(n),
          t_(t),
          extend_(extend),
          shorten_(shorten),
          even_(even),
          p_chase_(p_chase),
          EsNo_dB_min_(EsNo_dB_min),
          EsNo_dB_max_(EsNo_dB_max),
          gen_(std::random_device{}()),
          bit_dist_(0, 1),
          esno_dist_(EsNo_dB_min_, EsNo_dB_max_)
    {
        if (EsNo_dB_max_ < EsNo_dB_min_) {
            throw std::runtime_error("EsNo_dB_max must be >= EsNo_dB_min");
        }

        readG(G_, n_, t_, even_, extend_);
        code_ = std::make_unique<BCH>(n_, t_, even_, extend_, G_, shorten_);

        makpatterns(
            p_chase_,
            ChaseIpatternNum_,
            ChaseIIpatternNum_,
            ChaseIpatterns_,
            ChaseIIpatterns_
        );
    }

    SampleResult generate_one() {
        SampleResult result;

        // Sample EsNo_dB for this sample
        double EsNo_dB = (EsNo_dB_min_ == EsNo_dB_max_) ? EsNo_dB_min_ : esno_dist_(gen_);

        double EbNo_num = std::pow(10.0, EsNo_dB / 10.0);
        double sigma_2 = 1.0 / (2.0 * EbNo_num);
        double stddev = std::sqrt(sigma_2);

        result.stddev = stddev;

        std::normal_distribution<> noise_dist(0.0, stddev);

        BitVector message(code_->mK, 0);
        // for (int i = 0; i < code_->mK; ++i) {
        //     message[i] = static_cast<uint8_t>(bit_dist_(gen_));
        // }

        const BitVector cw = code_->encode(message);

        std::vector<double> y(code_->mN);
        BitVector yHD(code_->mN);
        for (int i = 0; i < code_->mN; ++i) {
            double v = noise_dist(gen_);
            y[i] = 1.0 - 2.0 * double(cw[i]) + v;
            yHD[i] = (y[i] > 0.0) ? 0 : 1;
        }

        auto const [zeroSyn, syn0] = code_->compute_syndrome(yHD);

        uint8_t weight = 0;
        for (int i = 0; i < code_->mN; ++i) {
            weight ^= yHD[i];
        }

        double best_metric = 1e18;

        std::vector<int> idx(code_->mN);
        for (int i = 0; i < code_->mN; ++i) {
            idx[i] = i;
        }

        std::sort(idx.begin(), idx.end(),
                  [&y](int i1, int i2) {
                      return std::fabs(y[i1]) < std::fabs(y[i2]);
                  });

        // Store the 8 smallest magnitudes of y
        int y_topk = std::min(8, code_->mN);
        for (int i = 0; i < y_topk; ++i) {
            result.smallest_eight_abs_y[i] = std::fabs(y[idx[i]]);
        }

        std::set<BitVector> seen_candidate_codewords;
        std::vector<BitVector> candidate_codewords;
        std::vector<double> metrics;
        std::vector<double> distED;
        int best_candidate_idx = -1;

        for (int mask = 0; mask < ChaseIIpatternNum_; ++mask) {
            const BitVector &filling_pattern = ChaseIIpatterns_[mask];

            std::set<int> flip_pos;
            BitVector test_cw = yHD;
            auto test_syn = syn0;
            uint8_t even0 = weight;

            for (int i = 0; i < p_chase_; ++i) {
                if (filling_pattern[i] == 1) {
                    test_cw[idx[i]] ^= 1;
                    flip_pos.insert(idx[i]);
                    even0 ^= 1;

                    if (idx[i] < code_->primitiveLength) {
                        for (int tt = 0; tt < code_->mT; ++tt) {
                            test_syn[tt] ^= code_->mGF.a_pow_tab[
                                ((2 * tt + 1) * idx[i]) % code_->primitiveLength
                            ];
                        }
                    }
                }
            }

            bool zeroSyn0 = true;
            for (auto s : test_syn) {
                if (s != 0) {
                    zeroSyn0 = false;
                    break;
                }
            }

            std::vector<int> ErrLoc0;
            int numErr = code_->decode(zeroSyn0, even0, test_syn, ErrLoc0);

            if (numErr != -1) {
                for (auto loc : ErrLoc0) {
                    test_cw[loc] ^= 1;
                    if (flip_pos.find(loc) == flip_pos.end()) {
                        flip_pos.insert(loc);
                    } else {
                        flip_pos.erase(loc);
                    }
                }

                double metric = 0.0;
                for (int i = 0; i < code_->mN; ++i) {
                    metric += (test_cw[i] == 0 ? -1.0 : 1.0) * y[i];
                }

                double destructionED = 0.0;
                for (auto loc : flip_pos) {
                    double hd_symbol = ((test_cw[loc] == 0) ? 1.0 : -1.0);
                    double diff = y[loc] - hd_symbol;
                    destructionED += diff * diff;
                }

                if (destructionED == 0) {
                    for (int i = 0; i < code_->mN; ++i) {
                        if (yHD[i] != test_cw[i]) {
                            throw std::runtime_error("some distance");
                        }
                    }
                }

                if (seen_candidate_codewords.insert(test_cw).second) {
                    candidate_codewords.push_back(test_cw);
                    metrics.push_back(metric);
                    distED.push_back(destructionED);

                    if (metric < best_metric) {
                        best_metric = metric;
                        best_candidate_idx = static_cast<int>(candidate_codewords.size()) - 1;
                    }
                }
            }
        }

        if (!candidate_codewords.empty()) {
            int omega = static_cast<int>(candidate_codewords.size());
            BitVector MLcodeword = candidate_codewords[best_candidate_idx];

            result.convertRatio = static_cast<double>(omega) / static_cast<double>(ChaseIIpatternNum_);

            for (int i = 0; i < code_->mN; ++i) {
                if (MLcodeword[i] != cw[i]) {
                    result.is_miscorrection = 1.0;
                    break;
                }
            }

            std::vector<int> idx2(omega);
            for (int i = 0; i < omega; ++i) {
                idx2[i] = i;
            }

            std::sort(idx2.begin(), idx2.end(),
                      [&metrics](int i1, int i2) {
                          return metrics[i1] < metrics[i2];
                      });

            int topk = std::min(5, omega);
            for (int i = 0; i < topk; ++i) {
                result.best_five_metrices[i] = metrics[idx2[i]] / double(code_->mN);
                result.best_five_distrED[i] = distED[idx2[i]];
            }
        }

        return result;
    }

    py::array_t<double> generate_batch(int batch_size = 100) {
        if (batch_size <= 0) {
            throw std::runtime_error("batch_size must be positive");
        }

        constexpr int feature_dim = 21;
        py::array_t<double> out({batch_size, feature_dim});
        auto buf = out.mutable_unchecked<2>();

        for (int i = 0; i < batch_size; ++i) {
            SampleResult s = generate_one();
            std::vector<double> row = s.to_vector();
            for (int j = 0; j < feature_dim; ++j) {
                buf(i, j) = row[j];
            }
        }

        return out;
    }

private:
    int n_;
    int t_;
    bool extend_;
    int shorten_;
    bool even_;
    int p_chase_;
    double EsNo_dB_min_;
    double EsNo_dB_max_;

    std::vector<std::vector<uint8_t>> G_;
    std::unique_ptr<BCH> code_;

    std::mt19937 gen_;
    std::uniform_int_distribution<> bit_dist_;
    std::uniform_real_distribution<> esno_dist_;

    int ChaseIpatternNum_;
    int ChaseIIpatternNum_;
    std::vector<BitVector> ChaseIpatterns_;
    std::vector<BitVector> ChaseIIpatterns_;
};

py::array_t<double> generate_batch(
    int batch_size = 100,
    int n = 255,
    int t = 2,
    bool extend = true,
    int shorten = 0,
    bool even = false,
    int p_chase = 5,
    double EsNo_dB_min = 3.1,
    double EsNo_dB_max = 3.1
) {
    BCHBatchGenerator gen(n, t, extend, shorten, even, p_chase, EsNo_dB_min, EsNo_dB_max);
    return gen.generate_batch(batch_size);
}

PYBIND11_MODULE(bch_simulator, m) {
    m.doc() = "pybind11 wrapper for BCH sample generation";

    py::class_<BCHBatchGenerator>(m, "BCHBatchGenerator")
        .def(py::init<int, int, bool, int, bool, int, double, double>(),
             py::arg("n") = 255,
             py::arg("t") = 2,
             py::arg("extend") = true,
             py::arg("shorten") = 0,
             py::arg("even") = false,
             py::arg("p_chase") = 5,
             py::arg("EsNo_dB_min") = 3.1,
             py::arg("EsNo_dB_max") = 3.1)
        .def("generate_batch", &BCHBatchGenerator::generate_batch,
             py::arg("batch_size") = 100,
             "Return a NumPy array of shape (batch_size, 21)");

    m.def("generate_batch", &generate_batch,
          py::arg("batch_size") = 100,
          py::arg("n") = 255,
          py::arg("t") = 2,
          py::arg("extend") = true,
          py::arg("shorten") = 0,
          py::arg("even") = false,
          py::arg("p_chase") = 5,
          py::arg("EsNo_dB_min") = 3.1,
          py::arg("EsNo_dB_max") = 3.1,
          "Generate a batch of BCH decoding samples as a NumPy array");
}