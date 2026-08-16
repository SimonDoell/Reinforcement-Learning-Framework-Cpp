#include <iostream>
#include <SFML/Graphics.hpp>
#include "Config.hpp"
#include "NeuralNetwork.hpp"
#include "RL.hpp"

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





struct State {
    sf::Vector2i agent_pos;
    sf::Vector2i target_pos;
};

struct Action {
        enum class Direction : uint8_t {Up = 0, Down = 1, Left = 2, Right = 3};

    public:
        Direction direction;

        static Action random() {
            return Action{.direction = (Direction)(rand() % 4)};
        }
};

struct Agent {
        using StepTp = Step<State, Action>;
        using LinearType = LinearLayer<XavierInit, Adam<>>;
    
    public:
        Agent() {
            actor.learning_rate = 0.0001f;
        }

        Matrix stateToMatrix(const State& state) const {
            Matrix mat = Matrix::Vector(4);
            
            mat = {
                (float)state.agent_pos.x  / static_cast<float>(25),
                (float)state.agent_pos.y  / static_cast<float>(25),
                (float)state.target_pos.x / static_cast<float>(25),
                (float)state.target_pos.y / static_cast<float>(25),
            };

            return mat;
        }
        
        Action act(const State& state) {
            Matrix input = stateToMatrix(state);
            Matrix res   = actor.forward(input);

            uint32_t max_index = -1;
            float max_value = -1e16f;

            for (size_t i = 0; i < 4; ++i) {
                if (res(i) > max_value) {
                    max_index = i;
                    max_value = res(i);
                }
            }
            
            return Action{
                .direction = (Action::Direction)(max_index)
            };
        }

        void train(const StepTp& step) {
            Matrix input  = stateToMatrix(step.state);
            Matrix output = Matrix::Vector(4);

            output = actor.forward(input);
            output.forEach([](float& o){
                o *= 0.9f;
            });

            output((uint32_t)step.action.direction) = step.reward;

            actor.train(input, output);
        }

        constexpr NeuralNetwork& Actor() {return actor;}

    private:
        // NeuralNetwork actor = NeuralNetwork(
        //     LinearType(4, 32),
        //     ActivationLayer<ReLU<>>(),
        //     LinearType(32, 16),
        //     ActivationLayer<ReLU<>>(),
        //     LinearType(16, 8),
        //     ActivationLayer<Tanh<>>(),
        //     LinearType(8, 4)
        // );

        NeuralNetwork actor = NeuralNetwork(
            LinearType(4, 16),
            ActivationLayer<ReLU<>>(),
            LinearType(16, 8),
            ActivationLayer<Tanh<>>(),
            LinearType(8, 4)
        );
};

template<uint32_t size = 25>
struct Environment {
        using StepTp = Step<State, Action>;
    
    public:
        void reset() {
            target_pos.x = rand() % size;
            target_pos.y = rand() % size;

            do {
                agent_pos.x = rand() % size;
                agent_pos.y = rand() % size;
            } while (target_pos == agent_pos);
        }

        bool isDone() const {return (agent_pos == target_pos);}

        float reward(const State& previos, const State& current) {
            float rew = 0.0f;
            
            float prev_dist = static_cast<sf::Vector2f>(previos.target_pos - previos.agent_pos).length();
            float curr_dist = static_cast<sf::Vector2f>(current.target_pos - current.agent_pos).length();

            float delta_dist = prev_dist - curr_dist;
            delta_dist /= static_cast<float>(size);

            rew += delta_dist;

            if (current.agent_pos == current.target_pos)
                rew += 10.0f;
            else
                rew -= 0.1f;

            return rew;
        }

        StepTp step(const Action& action) {
            StepTp step;
            step.done = false;
            step.action = action;
            step.reward = 0.0f;

            step.state = state();

            if (action.direction == Action::Direction::Up)         {agent_pos += sf::Vector2i(0, -1);}
            else if (action.direction == Action::Direction::Down)  {agent_pos += sf::Vector2i(0,  1);}
            else if (action.direction == Action::Direction::Left)  {agent_pos += sf::Vector2i(-1, 0);}
            else if (action.direction == Action::Direction::Right) {agent_pos += sf::Vector2i( 1, 0);}

            if (agent_pos.x < 0 || agent_pos.y < 0 || agent_pos.x >= (int)size || agent_pos.y >= (int)size) {
                step.reward -= 1.0f;
                agent_pos.x = std::clamp((int)agent_pos.x, (int)0, (int)size-1);
                agent_pos.y = std::clamp((int)agent_pos.y, (int)0, (int)size-1);
            }

            if (agent_pos == target_pos)
                step.done = true;

            step.next_state = state();
            step.reward += reward(step.state, step.next_state);

            return step;
        }

        State state() const {
            return State{
                .agent_pos  = agent_pos,
                .target_pos = target_pos
            };
        }

        void draw(sf::RenderTarget& target, sf::RenderStates states) const {
            sf::RectangleShape obj;
            obj.setFillColor(sf::Color(50, 50, 50));
            
        }

        template<uint32_t cells>
        friend void renderEnv(sf::RenderWindow& window, const Environment<cells>& environment, const sf::Vector2f&, const sf::Vector2f&);

    private:
        sf::Vector2i agent_pos  = {0, 0};
        sf::Vector2i target_pos = {0, 0};
};


template<uint32_t cells>
void renderEnv(sf::RenderWindow& window, const Environment<cells>& environment, const sf::Vector2f& center, const sf::Vector2f& size) {
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

            if ((int)i == environment.agent_pos.x && (int)j == environment.agent_pos.y) obj.setFillColor(sf::Color::Green);
            else if ((int)i == environment.target_pos.x && (int)j == environment.target_pos.y) obj.setFillColor(sf::Color::Red);
            else obj.setFillColor(sf::Color(50, 50, 50));

            window.draw(obj);
        }
    }
}


int main() {
    srand(time(0));
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Boilerplate");
    sf::View view(sf::FloatRect({0, 0}, {WIDTH, HEIGHT}));
    window.setView(view);
    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    RL<State, Action, Agent, Environment<>> rl;
    rl.epsilon = 0.7f;
    rl.max_steps = 45;
    rl.Environment().reset();
    
    size_t step = 0;
    uint32_t frame = 0;

    while (window.isOpen()) {
        while (const std::optional<sf::Event> event = window.pollEvent()) {if (event->is<sf::Event::Closed>()) window.close();}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) window.close();
       
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space) && frame % 3 == 0) {
            State state = rl.Environment().state();
            Action action = rl.Agent().act(state);
            rl.Environment().step(action);

            if (rl.Environment().isDone() || step > rl.max_steps) {rl.Environment().reset(); step = 0;}
            step++;
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::T)) {
            rl.trainEpisode();
            std::cout << "Epsilon: " << rl.epsilon << "\n";
        }
        
        window.clear(sf::Color(31, 31, 31));
        renderEnv(window, rl.Environment(), {WIDTH/2.0f, HEIGHT/2.0f}, {1000, 1000});
        window.display();
        frame++;
    }
    

    return 0;
}