#pragma once
#include "Matrix.hpp"

struct SGD {
    SGD(uint32_t, uint32_t) {}

    void update(
        Matrix& weights,
        Matrix& biases,
        const Matrix& activation,
        const Matrix& gradient,
        float lr
    ) {
        biases.forEach([&](float& b, uint32_t r, uint32_t c){
            b -= lr * gradient(r);
        });

        weights.forEach([&](float& w, uint32_t r, uint32_t c){
            float g = gradient(r) * activation(c);
            w -= lr * g;
        });
    }
};