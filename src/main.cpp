#include <iostream>
#include <SFML/Graphics.hpp>
#include "Config.hpp"
#include "NeuralNetwork.hpp"

constexpr float lerp(float percentage, float from, float to) {
    return from + (to - from) * percentage;
}

constexpr float invLerp(float value, float from, float to) {
    return (value - from) / (to - from);
}

constexpr float map(float value, float fromMin, float fromMax, float toMin, float toMax) {
    float percentage = invLerp(value, fromMin, fromMax);
    return lerp(percentage, toMin, toMax);
}


constexpr float learnFunc(float x) {
    return std::sinf(x * TWO_PI) * 0.5f + 0.5f;
}


struct Line : public sf::Drawable {
    public:
        sf::Vector2f posA, posB;
        sf::Color color = sf::Color::Red;
        float lineWidth = 2.0f;

        Line(const sf::Vector2f& _posA = sf::Vector2f(0, 0), const sf::Vector2f& _posB = sf::Vector2f(0, 0), sf::Color _color = sf::Color::Red, float _lineWidth = 2.0f)
        : posA(_posA), posB(_posB), color(_color), lineWidth(_lineWidth) {}

        void draw(sf::RenderTarget& target, sf::RenderStates states) const {
            float dx = posA.x - posB.x;
            float dy = posA.y - posB.y;
            float length = (posA - posB).length();
            float rotation = atan2(dy, dx);

            sf::RectangleShape obj;
            obj.setSize({length, lineWidth});
            obj.setOrigin({length, lineWidth/2.0f});
            obj.setRotation(sf::radians(rotation));
            obj.setPosition(posA);
            obj.setFillColor(color);

            target.draw(obj);
        }
};



size_t cells = 25;

struct RewardReturn {
    float reward;
    bool done;
};

struct State {
    sf::Vector2u finalPos;
    sf::Vector2u agentPos;

    static constexpr uint32_t input_dims = 4;

    Matrix networkInput() const {
        Matrix input = Matrix::Vector(input_dims);
        input = {
            (float)finalPos.x/(float)cells,
            (float)finalPos.y/(float)cells,
            (float)agentPos.x/(float)cells,
            (float)agentPos.y/(float)cells
        };
        return input;
    }
};


struct Action {
    static constexpr uint32_t output_dims = 4;
};

using LinearType = LinearLayer<XavierInit, SGD>;


struct Environment : public sf::Drawable {
    public:
        sf::Vector2f pos  = sf::Vector2f(0, 0);
        sf::Vector2f size = sf::Vector2f(100, 100);
        State state;
        sf::Vector2u lastAgentPos = {1, 1};

        NeuralNetwork network;

        float discount = 0.95f;
        float epsilon  = 0.95f;
        float epsilon_decay = 0.99f;
        
        Environment() : network(
            LinearType(State::input_dims, 32),
            ActivationLayer<ReLU<>>(),
            LinearType(32, 16),
            ActivationLayer<ReLU<>>(),
            LinearType(16, 8),
            ActivationLayer<Tanh<>>(),
            LinearType(8, Action::output_dims)
        ) {}

        void stepAgent() {
            uint32_t action = 0;
            
            if (randFloat(0, 1) <= epsilon) {
                action = rand() % 4;
            } else {
                Matrix output = network.forward(state.networkInput());

                float highest = 1e-20f;
                uint32_t highest_index = -1;

                for (size_t i = 0; i < Action::output_dims; ++i) {
                    
                }
            }


            epsilon *= epsilon_decay;
        }

        RewardReturn getReward() const {
            float reward = 0.0f;

            if (state.finalPos == state.agentPos)
                reward += 10.0f;
            
            
            return RewardReturn{
                .reward = reward,
                .done   = (state.finalPos == state.agentPos)
            };
        }

        void draw(sf::RenderTarget& target, sf::RenderStates states) const {
            sf::RectangleShape obj;
            obj.setSize(sf::Vector2f(
                size.x / static_cast<float>(cells),
                size.y / static_cast<float>(cells)
            ));
            obj.setFillColor(sf::Color::Red);
            obj.setOutlineColor(sf::Color(31, 31, 31));
            obj.setOutlineThickness(2);

            for (size_t i = 0; i < cells; ++i) {
                for (size_t j = 0; j < cells; ++j) {
                    obj.setPosition(pos + sf::Vector2f(
                        obj.getSize().x * i,
                        obj.getSize().y * j
                    ));

                    if (i == state.finalPos.x && j == state.finalPos.y) obj.setFillColor(sf::Color::Green);
                    else if (i == state.agentPos.x && j == state.agentPos.y) obj.setFillColor(sf::Color::Red);
                    else obj.setFillColor(sf::Color(50, 50, 50));

                    target.draw(obj);
                }
            }
        }

    private:
        std::vector<RewardReturn> episode;
};




int main() {
    srand(time(0));
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Boilerplate");
    sf::View view(sf::FloatRect({0, 0}, {WIDTH, HEIGHT}));
    window.setView(view);
    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);


    Environment env;
    env.pos  = sf::Vector2f(static_cast<float>(WIDTH) / 2.0f - 900.0f/2.0f, static_cast<float>(HEIGHT) / 2.0f - 900.0f/2.0f);
    env.size = sf::Vector2f(900, 900);



    while (window.isOpen()) {
        while (const std::optional<sf::Event> event = window.pollEvent()) {if (event->is<sf::Event::Closed>()) window.close();}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) window.close();

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space) && frame % 3 == 0) {
            env.stepAgent();
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::K)) {
            for (size_t i = 0; i < 100; ++i) {
                env.stepAgent();
            }
        }
       

        
        window.clear(sf::Color(31, 31, 31));
        window.draw(env);
        window.display();
        frame++;
    }
    

    return 0;
}