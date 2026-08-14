#pragma once
#include "BaseLayer.hpp"
#include "Activations.hpp"

template<typename Actv>
struct ActivationLayer : public Layer {
    public:
        void forward(Matrix& x) {
            activation_cache = x;
            
            x.forEach([&](float& activation){
                activation = Actv::forward(activation);
            });
        }
        
        void backward(Matrix& gradient, float lr) {
            gradient.forEach([&](float& g, size_t r, size_t c){
                g *= Actv::derivative(activation_cache(r, c));
            });
        }

    private:
        Matrix activation_cache;
};