#pragma once
#include <vector>
#include <utility>
#include <cstdint>
#include <concepts>
#include "NeuralNetwork.hpp"


template<typename Tp>
concept State = true;

template<typename Tp>
concept Action = true;

template<typename StateType, typename ActionType>
struct Step {
    StateType state;
    ActionType action;
    float reward;
    bool done;
    // StateType next_state;
};

template<typename Tp, typename StateType, typename ActionType>
concept Agent = requires (Tp& agent, const StateType& state) {
    { agent.act(state) } -> std::same_as<ActionType>;
};

template<typename Tp, typename StateType, typename ActionType>
concept Environment = requires (Tp& environment, const ActionType& a) {
    { environment.reset() } -> std::same_as<void>;
    { environment.step(a) } -> std::same_as<Step<StateType, ActionType>>;
};



template<
    State StateType,
    Action ActionType,
    Agent<StateType, ActionType> AgentType,
    Environment<StateType, ActionType> EnvironmentType
>
struct RL {
    public:
        float discount = 0.99f;

        RL() = default;

        constexpr       AgentType& Agent()       {return agent;}
        constexpr const AgentType& Agent() const {return agent;}

        constexpr       EnvironmentType& Environment()       {return environment;}
        constexpr const EnvironmentType& Environment() const {return environment;}

    private:
        AgentType agent;
        EnvironmentType environment;
};