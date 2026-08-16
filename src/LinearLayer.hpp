#pragma once
#include "Math.hpp"
#include "BaseLayer.hpp"
#include "Initilizations.hpp"
#include "SGD.hpp"

template<typename Initilization = XavierInit, typename Optimizer = SGD>
struct LinearLayer : public Layer {
    public:
        LinearLayer(size_t _in, size_t _out) : optimizer(_in, _out) {
            assert(_in  > 0);
            assert(_out > 0);

            weights = Matrix::Matrix(_out, _in);
            biases  = Matrix::Vector(_out);

            float range = Initilization::range(_in, _out);

            weights.forEach([&](float& w){
                w = randFloat(-range, range);
            });
        }
        
        void forward(Matrix& x) override {
            activation_cache = x;

            // z = W * x + b
            x = matmul(weights, x) + biases;
        }

        void backward(Matrix& gradient, float lr) override {
            Matrix prev_gradient = matmul(weights.transposed(), gradient);

            optimizer.update(weights, biases, activation_cache, gradient, lr);

            gradient = prev_gradient;
        }

        std::unique_ptr<Layer> Clone() override {
            return std::make_unique<LinearLayer<Initilization, Optimizer>>(*this);
        }

    private:
        Optimizer optimizer;
        Matrix activation_cache;
        Matrix weights;
        Matrix biases;
};