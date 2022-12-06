#include "Individual.h"

Individual::Individual(unsigned long id) : mt((random_device())())
{
    this->id = id;
    this->fitness = 0.0;
    generate_strategy();
    generate_reputation();
}

short Individual::act(short repcomb_index, vector<float> epsilon)
{
    bernoulli_distribution dist(epsilon[0]);
    bool can_execute = dist(mt);

    if (can_execute)
        return strategy[repcomb_index];
    else
        return !strategy[repcomb_index];
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
    bernoulli_distribution dist(0.5);
    for(int i = 0; i < strategy.size(); i++)
        strategy[i] = dist(mt);
}

void Individual::generate_reputation()
{
    reputation.clear();
    reputation.resize(2);
    bernoulli_distribution dist(0.5);
    for(int i = 0; i < reputation.size(); i++)
        reputation[i] = dist(mt);
}

void Individual::reset()
{
    payoffs.clear();
    fitness = 0.0;
    generate_reputation();
}