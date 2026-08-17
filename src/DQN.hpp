#pragma once
#include <deque>
#include <vector>
#include <type_traits>
#include "NeuralNetwork.hpp"



template<typename StateType, typename ActionType>
struct Step {
    StateType state;
    ActionType action;
    float reward;
    StateType next_state;
    bool done;
};

template<typename Tp>
concept StateT = true;

template<typename Tp>
concept ActionT = requires (const Tp& a) {
    { Tp::random()   } -> std::same_as<Tp>;
};

template<typename Tp, typename StateType, typename ActionType>
concept InstructionT = requires (const StateType& s, const ActionType& a, const Matrix& m) {
    { Tp::stateToInput(s)   } -> std::same_as<Matrix>;
    { Tp::outputToAction(m) } -> std::same_as<ActionType>;
    { Tp::actionToIndex(a)  } -> std::same_as<uint32_t>;
};

template<typename Tp, typename StateType, typename ActionType>
concept EnvironmentT = requires (Tp& environment, const ActionType& a) {
    { environment.reset() } -> std::same_as<void>;
    { environment.step(a) } -> std::same_as<Step<StateType, ActionType>>;
    { environment.state() } -> std::same_as<StateType>;
};


template<
    StateT  StateType,
    ActionT ActionType,
    InstructionT<StateType, ActionType> InstructionType,
    EnvironmentT<StateType, ActionType> EnvironmentType
>
struct DQN {
        using StepTp = Step<StateType, ActionType>;
    
    public:
        // ----- Hyperparamaters -----
        size_t max_steps    = 128;       // Maximum steps in a episode before forcefully terminating
        float discount      = 0.99f;     // discount factor accounting for future (accumulative) rewards
        float epsilon       = 0.99f;     // initial epsilon factor / randomness in taking action / exploration rate
        float min_epsilon   = 0.05f;     // minimum chance of the agent taking a random action
        float epsilon_decay = 0.99f;     // the amount epsilon decays per episode
        size_t batch_size   = 32;        // how many transitions from the replay buffer are sampled each step in the episode
        size_t buffer_size  = 10'000;    // How big the deque replay buffer is, before discarding old transitions / steps from the replay buffer
        size_t target_update_freq = 64;  // After how many steps he weights from the q_network are copied to the weights from the target_network
        size_t learning_start     = 128; // After how many entries in the replay buffer the agent starts to learn from the replay buffer

        template<typename... Layers>
        DQN(const EnvironmentType& _environment, const Layers&... _layers)
        : q_network(_layers...), target_network(_layers...), environment(_environment) {}

        void trainEpisode() {
            environment.reset();

            for (size_t i = 0; i < max_steps; ++i) {
                StateType  state  = environment.state();
                ActionType action = epsilon_greedy(state);
                StepTp     step   = environment.step(action);

                replay_buffer.push_back(step);

                while (replay_buffer.size() > buffer_size)
                    replay_buffer.pop_front();

                if (replay_buffer.size() >= learning_start) {
                    // choose and train on mini batch
                    for (size_t b = 0; b < batch_size; ++b) {
                        StepTp step = replay_buffer[rand() % replay_buffer.size()];

                        float target = 0.0f;

                        if (step.done) {
                            target = step.reward;
                        } else {
                            Matrix next_q = target_network.forward(
                                InstructionType::stateToInput(step.next_state));

                            float max_next_q = -1e16f;
                            next_q.forEach([&](float& q){
                                max_next_q = std::max(max_next_q, q);
                            });

                            target = step.reward + discount * max_next_q;
                        }

                        Matrix online_q = q_network.forward(
                            InstructionType::stateToInput(step.state));

                        online_q(InstructionType::actionToIndex(step.action)) = target;

                        q_network.train(InstructionType::stateToInput(step.state), online_q);
                    }
                }

                if (++total_steps % target_update_freq == 0) {
                    target_network = q_network;
                }

                if (step.done) break;
            }

            epsilon = std::max(epsilon * epsilon_decay, min_epsilon);
        }

        void setLearningRate(float lr) {
            q_network.learning_rate      = lr;
            target_network.learning_rate = lr;
        }

        constexpr NeuralNetwork& TargetNetwork() {return target_network;}
        constexpr EnvironmentType& Environment() {return environment;}

    private:
        NeuralNetwork q_network;  // online network
        NeuralNetwork target_network;
        EnvironmentType environment;
        std::deque<StepTp> replay_buffer;
        size_t total_steps = 0;

        ActionType epsilon_greedy(const StateType& state) {
            if (randFloat(0, 1) < epsilon) {
                return ActionType::random();
            } else {
                Matrix output = q_network.forward(InstructionType::stateToInput(state));
                ActionType action = InstructionType::outputToAction(output);
                return action;
            }
        }
};




// ------- Pseudocode -------
//
// for each episode:
//     reset environment
//     while not done and steps < max_steps:
//         state = env.state()
//         action = ε-greedy(Q_network(state))
//         next_state, reward, done = env.step(action)
//         store (state, action, reward, next_state, done) in replay buffer
//
//         if buffer has enough samples:
//             sample a mini-batch
//             compute targets using TARGET network
//             train Q_network on the batch (with gradient clipping)
//
//         every N steps: copy Q_network weights → target_network
//
//     decay ε