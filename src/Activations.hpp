#pragma once
#include <cmath>

template<float leakage = 0.15f>
struct ReLU {
    static constexpr float forward   (float x) noexcept {return std::max(x, x * leakage);}
    static constexpr float derivative(float x) noexcept {return (x >= 0.0f ? 1.0f : leakage);}
};

template<float relaxation = 1.15f>
struct Tanh {
    static constexpr float forward   (float x) noexcept {return std::tanhf(x) * relaxation;}
    static constexpr float derivative(float x) noexcept {float t = std::tanhf(x); return (1.0f - t * t) * relaxation;}
};