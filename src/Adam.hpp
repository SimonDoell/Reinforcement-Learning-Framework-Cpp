#pragma once
#include <cmath>
#include "Matrix.hpp"

template<float betta_1 = 0.9f /*momentum*/, float betta_2 = 0.999f>
struct Adam {
    public:
        Adam(uint32_t _in, uint32_t _out) {
            mean_momentum_w     = Matrix::Matrix(_out, _in);
            variance_momentum_w = Matrix::Matrix(_out, _in);

            mean_momentum_b     = Matrix::Vector(_out);
            variance_momentum_b = Matrix::Vector(_out);
        }

        void update(Matrix& weights, Matrix& biases, const Matrix& activation, const Matrix& gradient, float lr)
        {
            biases.forEach([&](float& b, uint32_t r, uint32_t){
                float g = gradient(r);

                // First moment (mean) estimate:
                mean_momentum_b(r) *= betta_1;
                mean_momentum_b(r) += (1.0f - betta_1) * g;

                // Second moment (variance) estimate:
                variance_momentum_b(r) *= betta_2;
                variance_momentum_b(r) += (1.0f - betta_2) * (g * g);

                float betta_1_t = 1.0f - std::pow(betta_1, t);
                float betta_2_t = 1.0f - std::pow(betta_2, t);

                // Bias correction:
                float corrected_mean_momentum_b     = mean_momentum_b(r)     / betta_1_t;
                float corrected_variance_momentum_b = variance_momentum_b(r) / betta_2_t;

                // Final weight update:
                b -= corrected_mean_momentum_b / std::sqrtf(corrected_variance_momentum_b + eps) * lr;
            });
    
            weights.forEach([&](float& w, uint32_t r, uint32_t c){
                float g = gradient(r) * activation(c);

                // First moment (mean) estimate:
                mean_momentum_w(r, c) *= betta_1;
                mean_momentum_w(r, c) += (1.0f - betta_1) * g;

                // Second moment (variance) estimate:
                variance_momentum_w(r, c) *= betta_2;
                variance_momentum_w(r, c) += (1.0f - betta_2) * (g * g);

                float betta_1_t = 1.0f - std::pow(betta_1, t);
                float betta_2_t = 1.0f - std::pow(betta_2, t);

                // Bias correction:
                float corrected_mean_momentum_w     = mean_momentum_w(r, c)     / betta_1_t;
                float corrected_variance_momentum_w = variance_momentum_w(r, c) / betta_2_t;

                // Final weight update:
                w -= corrected_mean_momentum_w / std::sqrtf(corrected_variance_momentum_w + eps) * lr;
            });

            t++;
        }

        
    private:
        Matrix mean_momentum_w;
        Matrix mean_momentum_b;
        Matrix variance_momentum_w;
        Matrix variance_momentum_b;
        uint32_t t = 1;
        
        static constexpr float eps = 1e-6f;
};