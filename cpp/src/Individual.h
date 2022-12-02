#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <iostream>
#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <vector>
#include <numeric>
#include <algorithm>

#define strategy_length 8

using namespace std;
using namespace boost;

class Individual
{
    public:
        Individual(unsigned long id);
        short act(short repcomb_index, vector<float> epsilon);
        void add_payoff(float payoff);
        void generate_strategy();
        void reset();

        unsigned long id;
        bitset<strategy_length> strategy;
        vector<float> payoffs;
        float fitness;
        dynamic_bitset<> reputation;

    private:
        bool reverse_act(bool action);
        float average(vector<float> const& vec);
        void generate_reputation();
};

#endif