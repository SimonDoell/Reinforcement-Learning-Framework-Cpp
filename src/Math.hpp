#pragma once
#include <random>

constexpr float PI     = 3.14159265358979f;
constexpr float TWO_PI = PI * 2.0f;

constexpr float randFloat(float min, float max) {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (max - min) + min;
}