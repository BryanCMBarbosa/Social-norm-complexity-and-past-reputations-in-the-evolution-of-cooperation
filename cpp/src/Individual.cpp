#include "Individual.h"

Individual::Individual(unsigned long id) : mt((random_device())())
{
    this->id = id;
    generate_strategy();
    generate_reputation();
    fitness = 0.0;
    payoffs_size = 0;
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

float Individual::get_fitness()
{
    return fitness;
}

void Individual::generate_strategy()
{
    bernoulli_distribution dist(0.5);
    for(int i = 0; i < strategy.size(); i++)
    {
        strategy[i] = dist(mt);
    }
}

void Individual::generate_reputation()
{
    while (!reputation.empty())
        reputation.pop();

    bernoulli_distribution dist(0.5);
    for(int i = 0; i < 2; i++)
        reputation.push(dist(mt));
}

void Individual::reset_payoff()
{
    fitness = 0.0;
    payoffs_size = 0;
}

void Individual::add_payoff(double value)
{
    if (payoffs_size > 0)
    {
        double a, b;
        payoffs_size++;
        a = 1.0 / (double)payoffs_size;
        b = 1.0 - a;
        fitness = (a*value) + (b*fitness);
    }
    else
    {
        fitness = value;
        payoffs_size++;
    }
}