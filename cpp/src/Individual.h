#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <iostream>
#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <vector>
#include <numeric>
#include <random>

#define strategy_length 8

using namespace std;
using namespace boost;

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
        dynamic_bitset<> reputation;

    private:
        mt19937 mt;
        float average(vector<float> const& vec);
        void generate_reputation();
};

#endif