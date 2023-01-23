#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <iostream>
#include <bitset>
#include <vector>
#include <numeric>
#include <random>

#define strategy_length 8

using namespace std;

class Individual
{
    public:
        Individual(unsigned long id);
        short act(short repcomb_index, vector<float> epsilon);
        float get_fitness();
        void generate_strategy();
        void reset_payoff();

        unsigned long id;
        bitset<strategy_length> strategy;
        vector<float> payoffs;
        vector<bool> reputation;

    private:
        mt19937 mt;
        float average(vector<float> const& vec);
        void generate_reputation();
};

#endif