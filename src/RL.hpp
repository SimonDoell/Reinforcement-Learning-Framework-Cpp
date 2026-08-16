#pragma once
#include <vector>
#include <utility>
#include <cstdint>
#include <concepts>
#include "NeuralNetwork.hpp"


template<typename Tp>
concept StateT = true;

template<typename Tp>
concept ActionT = requires (Tp& action) {
    { Tp::random() } -> std::same_as<Tp>;
};

template<typename StateType, typename ActionType>
struct Step {
    StateType state;
    ActionType action;
    StateType next_state;
    float reward;
    bool done;
};

template<typename Tp, typename StateType, typename ActionType>
concept AgentT = requires (Tp& agent, const StateType& state, const Step<StateType, ActionType>& step) {
    { agent.act(state)  } -> std::same_as<ActionType>;
    { agent.train(step) } -> std::same_as<void>;
};

template<typename Tp, typename StateType, typename ActionType>
concept EnvironmentT = requires (Tp& environment, const ActionType& a) {
    { environment.reset() } -> std::same_as<void>;
    { environment.step(a) } -> std::same_as<Step<StateType, ActionType>>;
    { environment.state() } -> std::same_as<StateType>;
};



template<
    StateT StateType,
    ActionT ActionType,
    AgentT<StateType, ActionType> AgentType,
    EnvironmentT<StateType, ActionType> EnvironmentType
>
struct RL {
        using StepTp = Step<StateType, ActionType>;
    
    public:
        size_t max_steps    = 100;    // Maximum steps in a episode before forcefully terminating
        float discount      = 0.99f;  // discount factor accounting for future (accumulative) rewards
        float epsilon       = 0.95f;  // initial epsilon factor / randomness in taking action
        float min_epsilon   = 0.05f;  // minimum chance of the agent taking a random action
        float epsilon_decay = 0.99f;  // the amount epsilon decays per episode

        RL(AgentType _agent = AgentType(), EnvironmentType _environment = EnvironmentType())
        : agent(std::move(_agent)), environment(std::move(_environment)) {}

        void trainEpisode() {
            environment.reset();

            std::vector<StepTp> steps;
            
            for (size_t i = 0; i < max_steps; ++i) {
                if (randFloat(0, 1) <= epsilon) {
                    // Take random action
                    ActionType action = ActionType::random();

                    steps.emplace_back(environment.step(action));
                } else {
                    // Let the agent decide the action
                    StateType state = environment.state();

                    ActionType action = agent.act(state);

                    steps.emplace_back(environment.step(action));
                }

                if (steps.back().done) break;
            }

            float G = 0.0f;
            
            for (int i = steps.size()-1; i >= 0; --i) {
                G = steps[i].reward + discount * G; // terminal rewards should be included in the reward function of the env itself
                steps[i].reward = G;
            }

            for (const StepTp& step : steps)
                agent.train(step);

            // Epsilon update
            epsilon *= epsilon_decay;
            epsilon  = std::max(epsilon, min_epsilon);
        }

        constexpr       AgentType& Agent()       {return agent;}
        constexpr const AgentType& Agent() const {return agent;}

        constexpr       EnvironmentType& Environment()       {return environment;}
        constexpr const EnvironmentType& Environment() const {return environment;}

    private:
        AgentType agent;
        EnvironmentType environment;
};