#pragma once
#include <memory>
#include <vector>
#include "Adam.hpp"
#include "Matrix.hpp"
#include "Activations.hpp"
#include "LinearLayer.hpp"
#include "ActivationLayer.hpp"



struct NeuralNetwork {
    public:
        float learning_rate = 0.01f;
        
        template<typename... Layers>
        NeuralNetwork(const Layers&... _layers) {
            (layers.emplace_back(std::make_unique<Layers>(std::move(_layers))), ...);
        }

        Matrix forward(Matrix activations) {
            for (size_t i = 0; i < layers.size(); ++i) {
                layers[i]->forward(activations);
            }

            return activations;
        }
        
        void train(const Matrix& activations, const Matrix& desired_activations) {
            Matrix result = forward(activations);
            result -= desired_activations;

            for (int i = layers.size()-1; i >= 0; --i) {
                layers[i]->backward(result, learning_rate);
            }
        }
    
    private:
        std::vector<std::unique_ptr<Layer>> layers;
};