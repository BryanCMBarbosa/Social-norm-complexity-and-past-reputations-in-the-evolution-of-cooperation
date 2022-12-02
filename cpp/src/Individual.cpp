#include "Individual.h"

Individual::Individual(unsigned long id)
{
    this->id = id;
    this->strategy = strategy;
    this->fitness = 0.0;
    generate_strategy();
    generate_reputation();
}

short Individual::act(short repcomb_index, vector<float> epsilon)
{
    bool can_execute = bool(rand() % 2);

    if (can_execute)
        return strategy[repcomb_index];
    else
        return reverse_act(strategy[repcomb_index]);
}

bool Individual::reverse_act(bool action)
{
    if (action == 1)
        return 0;
    else
        return 1;
}

void Individual::add_payoff(float payoff)
{
    payoffs.push_back(payoff);
    fitness = average(payoffs);
}

float Individual::average(vector<float> const& vec)
{
    if(vec.empty())
        return 0;

    auto const count = static_cast<float>(vec.size());

    return reduce(vec.begin(), vec.end()) / count;
}

void Individual::generate_strategy()
{
    for(int i = 0; i < strategy.size(); i++)
        strategy[i] = bool(rand()%2);
}

void Individual::generate_reputation()
{
    reputation.clear();
    reputation.resize(2);
    for(int i = 0; i < reputation.size(); i++)
        reputation[i] = bool(rand()%2);
}

void Individual::reset()
{
    payoffs.clear();
    fitness = 0.0;
    generate_reputation();
}