#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <iostream>
#include <bitset>
#include <vector>
#include <queue>
#include <numeric>
#include <random>

#define strategy_length 8

using namespace std;

class Individual
{
    public:
        Individual(unsigned long id);
        short act(short repcomb_index, vector<float> epsilon);
        float get_fitness(unsigned long z);
        void generate_strategy();
        void reset_payoff();
        void add_payoff(double value);

        unsigned long id;
        bitset<strategy_length> strategy;
        double payoffs_sum;
        double fitness;
        queue<bool> reputation;

    private:
        mt19937 mt;
        float sum(vector<float> const& vec);
        float average(vector<float> const& vec);
        void generate_reputation();
};

#endif