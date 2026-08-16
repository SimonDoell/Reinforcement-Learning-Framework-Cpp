#include <iostream>
#include <SFML/Graphics.hpp>
#include "Config.hpp"
#include "NeuralNetwork.hpp"
#include "DQN.hpp"



struct Action {
        enum class MoveDir {Up=0, Down=1, Left=2, Right=3};
    
    public:
        MoveDir direction;

        static Action random() {
            return Action{.direction = (MoveDir)(rand() % 4)};
        }
};

struct State {
    sf::Vector2i agent_pos;
    sf::Vector2i target_pos;
};


struct Instructions {
    static Matrix stateToInput(const State& state) {
        Matrix res = Matrix::Vector(4);

        res = {
            (float)state.agent_pos.x,
            (float)state.agent_pos.y,
            (float)state.target_pos.x,
            (float)state.target_pos.y
        };

        return res;
    }

    static Action outputToAction(const Matrix& matrix) {
        Action action;

        float max_val = -1e16f;
        size_t max_index = -1;

        for (size_t i = 0; i < 4; ++i) {
            if (matrix(i) >= max_val) {
                max_index = i;
                max_val = matrix(i);
            }
        }

        return Action{.direction = (Action::MoveDir)max_index};
    }

    static uint32_t actionToIndex(const Action& action) {
        return (uint32_t)action.direction;
    }
};


struct Environment {
        using StepTp = Step<State, Action>;
    
    public:
        void reset() {}

        StepTp step(const Action& action) {
            return StepTp{};
        }

        State state() {
            return State{};
        }
};




int main() {
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Boilerplate");
    sf::View view(sf::FloatRect({0, 0}, {WIDTH, HEIGHT}));
    window.setView(view);
    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    using LinearType = LinearLayer<XavierInit, Adam<>>;

    DQN<State, Action, Instructions, Environment> dqn(
        LinearType(4, 16),
        ActivationLayer<ReLU<>>(),
        LinearType(16, 8),
        ActivationLayer<Tanh<>>(),
        LinearType(8, 4)
    );


    while (window.isOpen()) {
        while (const std::optional<sf::Event> event = window.pollEvent()) {if (event->is<sf::Event::Closed>()) window.close();}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) window.close();

        
        window.clear(sf::Color(31, 31, 31));
        
        window.display();
    }
    

    return 0;
}