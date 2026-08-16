#pragma once
#include "BaseLayer.hpp"
#include "Activations.hpp"

template<typename Actv>
struct ActivationLayer : public Layer {
    public:
        void forward(Matrix& x) override {
            activation_cache = x;
            
            x.forEach([&](float& activation){
                activation = Actv::forward(activation);
            });
        }
        
        void backward(Matrix& gradient, float lr) override {
            gradient.forEach([&](float& g, size_t r, size_t c){
                g *= Actv::derivative(activation_cache(r, c));
            });
        }

        std::unique_ptr<Layer> Clone() override {
            return std::make_unique<ActivationLayer<Actv>>(*this);
        }

    private:
        Matrix activation_cache;
};