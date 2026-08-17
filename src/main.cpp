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
        Environment(size_t _cells)
        : cells(_cells) {reset();}
        
        void reset() {
            target_pos.x = rand() % cells;
            target_pos.y = rand() % cells;

            do {
                agent_pos.x = rand() % cells;
                agent_pos.y = rand() % cells;
            } while (target_pos == agent_pos);
        }

        float reward(const State& previos, const State& current) {
            float rew = 0.0f;
            
            float prev_dist = static_cast<sf::Vector2f>(previos.target_pos - previos.agent_pos).length();
            float curr_dist = static_cast<sf::Vector2f>(current.target_pos - current.agent_pos).length();

            float delta_dist = prev_dist - curr_dist;
            delta_dist /= static_cast<float>(cells);

            rew += delta_dist;

            if (current.agent_pos == current.target_pos)
                rew += 10.0f;
            else
                rew -= 0.1f;

            return rew;
        }

        StepTp step(const Action& action) {
            StepTp step;
            step.action = action;
            step.reward = 0.0f;
            step.done   = false;
            step.state  = state();

            if (action.direction == Action::MoveDir::Up)         {agent_pos += sf::Vector2i(0, -1);}
            else if (action.direction == Action::MoveDir::Down)  {agent_pos += sf::Vector2i(0,  1);}
            else if (action.direction == Action::MoveDir::Left)  {agent_pos += sf::Vector2i(-1, 0);}
            else if (action.direction == Action::MoveDir::Right) {agent_pos += sf::Vector2i( 1, 0);}

            if (agent_pos.x < 0 || agent_pos.y < 0 || agent_pos.x >= (int)cells || agent_pos.y >= (int)cells) {
                step.reward -= 1.0f;
                agent_pos.x = std::clamp((int)agent_pos.x, (int)0, (int)cells-1);
                agent_pos.y = std::clamp((int)agent_pos.y, (int)0, (int)cells-1);
            }
            
            if (agent_pos == target_pos) {
                std::cout << "Finished!\n";
                step.done = true;
            }

            step.next_state = state();
            step.reward += reward(step.state, step.next_state);

            return step;
        }

        State state() {
            return State{};
        }

        friend void renderEnv(sf::RenderWindow& window, const Environment& environment, const sf::Vector2f& center, const sf::Vector2f& size);

    private:
        size_t cells;
        sf::Vector2i agent_pos  = {0, 0};
        sf::Vector2i target_pos = {0, 0};
};

void renderEnv(sf::RenderWindow& window, const Environment& env, const sf::Vector2f& center, const sf::Vector2f& size) {
    size_t cells = env.cells;
    sf::RectangleShape obj;
    sf::Vector2f cell_size = sf::Vector2f(size.x / static_cast<float>(cells), size.y / static_cast<float>(cells));
    obj.setSize(cell_size);
    obj.setFillColor(sf::Color::Green);
    obj.setOutlineThickness(2);
    obj.setOutlineColor(sf::Color(31, 31, 31));

    sf::Vector2f pos = center - size/2.0f;

    for (uint32_t i = 0; i < cells; ++i) {
        for (uint32_t j = 0; j < cells; ++j) {
            sf::Vector2f cell_pos = pos + sf::Vector2f(cell_size.x * i, cell_size.y * j);
            obj.setPosition(cell_pos);

            if ((int)i == env.agent_pos.x && (int)j == env.agent_pos.y) obj.setFillColor(sf::Color::Green);
            else if ((int)i == env.target_pos.x && (int)j == env.target_pos.y) obj.setFillColor(sf::Color::Red);
            else obj.setFillColor(sf::Color(50, 50, 50));

            window.draw(obj);
        }
    }
}




int main() {srand(time(0));
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Boilerplate");
    sf::View view(sf::FloatRect({0, 0}, {WIDTH, HEIGHT}));
    window.setView(view);
    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    size_t step = 0;
    using LinearType = LinearLayer<XavierInit, Adam<>>;

    DQN<State, Action, Instructions, Environment> dqn(Environment(25),
        LinearType(4, 8),
        ActivationLayer<Tanh<>>(),
        LinearType(8, 4)
    );

    dqn.buffer_size = 500;
    dqn.discount    = 0.95f;
    dqn.max_steps   = 50;
    dqn.target_update_freq = 32;


    while (window.isOpen()) {
        while (const std::optional<sf::Event> event = window.pollEvent()) {if (event->is<sf::Event::Closed>()) window.close();}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) window.close();

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::T)) {
            dqn.trainEpisode();
            std::cout << "Epsilon: " << dqn.epsilon << "\n";
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space)) {
            NeuralNetwork& target_network = dqn.TargetNetwork();
            Environment& environment      = dqn.Environment();

            State state = environment.state();
            Action action = Instructions::outputToAction(target_network.forward(Instructions::stateToInput(state)));
            environment.step(action);
        }
        
        window.clear(sf::Color(31, 31, 31));
        renderEnv(window, dqn.Environment(), {WIDTH/2.0f, HEIGHT/2.0f}, {1000, 1000});
        window.display();
    }
    

    return 0;
}