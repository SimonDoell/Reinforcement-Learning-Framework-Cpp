#pragma once
#include <cstdint>
#include <cmath>

struct HeInit {
    static constexpr float range(uint32_t in, uint32_t) noexcept {
        return std::sqrtf(2.0f / static_cast<float>(in));
    }
};

struct XavierInit {
    static constexpr float range(uint32_t in, uint32_t out) noexcept {
        return std::sqrtf(6.0f / static_cast<float>(in + out));
    }
};