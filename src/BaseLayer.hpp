#pragma once
#include "Matrix.hpp"

struct Layer {
    virtual ~Layer() = default;
    virtual void forward(Matrix& x) = 0;
    virtual void backward(Matrix& gradient, float lr) = 0;
};