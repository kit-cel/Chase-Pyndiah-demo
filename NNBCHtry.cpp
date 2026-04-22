#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <iomanip>

struct SimpleNN {
    int input_dim = 0;
    int hidden_dim = 0;


    std::vector<std::vector<double>> fc1_weight; // [hidden_dim][input_dim]
    std::vector<double> fc1_bias;                // [hidden_dim]

    std::vector<double> fc2_weight;              // [hidden_dim]
    double fc2_bias = 0.0;

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
        if ((int)x_raw.size() != input_dim) {
            throw std::runtime_error("Input size mismatch in forward()");
        }

        std::vector<double> x = x_raw;

        // fc1 + identity activation
        std::vector<double> hidden(hidden_dim, 0.0);
        for (int h = 0; h < hidden_dim; ++h) {
            double sum = fc1_bias[h];
            for (int i = 0; i < input_dim; ++i) {
                sum += fc1_weight[h][i] * x[i];
            }
            hidden[h] = sum; // identity activation
        }

        // fc2
        double logit = fc2_bias;
        for (int h = 0; h < hidden_dim; ++h) {
            logit += fc2_weight[h] * hidden[h];
        }

        // sigmoid output
        return sigmoid(logit);
    }

    int predict(const std::vector<double>& x_raw, double threshold = 0.5) const {
        return forward(x_raw) >= threshold ? 1 : 0;
    }
};

static std::string read_tag(std::istream& in) {
    std::string tag;
    if (!(in >> tag)) {
        throw std::runtime_error("Failed to read tag");
    }
    return tag;
}

SimpleNN load_network(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open weight file: " + filename);
    }

    SimpleNN net;

    in >> net.input_dim >> net.hidden_dim;
    if (!in) {
        throw std::runtime_error("Failed to read input_dim and hidden_dim");
    }


    {
        std::string tag = read_tag(in);
        if (tag != "fc1_weight") throw std::runtime_error("Expected tag: fc1_weight");

        net.fc1_weight.assign(net.hidden_dim, std::vector<double>(net.input_dim));
        for (int h = 0; h < net.hidden_dim; ++h) {
            for (int i = 0; i < net.input_dim; ++i) {
                in >> net.fc1_weight[h][i];
            }
        }
    }

    {
        std::string tag = read_tag(in);
        if (tag != "fc1_bias") throw std::runtime_error("Expected tag: fc1_bias");

        net.fc1_bias.resize(net.hidden_dim);
        for (int h = 0; h < net.hidden_dim; ++h) {
            in >> net.fc1_bias[h];
        }
    }

    {
        std::string tag = read_tag(in);
        if (tag != "fc2_weight") throw std::runtime_error("Expected tag: fc2_weight");

        net.fc2_weight.resize(net.hidden_dim);
        for (int h = 0; h < net.hidden_dim; ++h) {
            in >> net.fc2_weight[h];
        }
    }

    {
        std::string tag = read_tag(in);
        if (tag != "fc2_bias") throw std::runtime_error("Expected tag: fc2_bias");

        in >> net.fc2_bias;
    }

    if (!in) {
        throw std::runtime_error("Failed while parsing weight file");
    }

    return net;
}

int main() {
    try {
        SimpleNN net = load_network("bch_nn_weights.txt");

        // Example input: 13 features
        // This must be the feature vector only, WITHOUT the label column.
        std::vector<double> x = {
            8.0000e+00,  2.5000e-01, -9.4513e-01, -9.4051e-01, -9.3764e-01,
         -9.3239e-01, -9.3197e-01,  4.0837e-03,  2.9878e+00,  4.5732e+00,
          2.6574e+00,  2.8009e+00
        };

        double prob = net.forward(x);
        int pred = net.predict(x);

        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Predicted probability: " << prob << "\n";
        std::cout << "Predicted label: " << pred << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}