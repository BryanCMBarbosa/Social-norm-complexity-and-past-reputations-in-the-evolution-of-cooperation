#ifndef SIMULATION_H
#define SIMULATION_H

#include <iostream>
#include <vector>
#include <string>
#include <bitset>
#include <random>
#include <omp.h>
#include <map>
#include <ctime> 
#include <math.h>
#include <fstream>
#include "Individual.h"

using namespace std;

#define strategy_length 8

class Simulation
{
    public:
        Simulation(bitset<16> norm, string norm_name, unsigned long z, unsigned long long generations, float payoff_b = 5, float payoff_c = 1, float epsilon = 0.01, float alpha = 0.01, float chi = 0.01);
        vector<double> run_generations(unsigned long long run, unsigned long long runs);
        void run_n_runs(unsigned long long runs);

    private:
        void create_agents();
        short repcomb_to_index(Individual donor, Individual receptor, bool donor_action = false, bool use_donor_action = false);
        void judge(Individual& x, Individual& y, bool x_action, bool y_action);
        void match(Individual& x, Individual& y);
        void mutation(vector<unsigned long long>& indexes);
        void imitation(vector <unsigned long long>& indexes);
        vector<vector<unsigned long long>> divide_mutation_imitation();
        vector<Individual> sample_with_reposition(vector<Individual>& vec, unsigned long long sample_size);
        unsigned long get_sampled_individual(vector <Individual>& population, long exception_id = -1);
        void turn_to_csv(unsigned long long runs, vector<vector<double>> eta_each_run);

        bitset<16> norm;
        string norm_name;
        unsigned long z;
        vector<Individual> individuals;
        vector<float> mi;
        unsigned long long generations;
        float payoff_b;
        float payoff_c;
        vector<float> epsilon;
        vector<float> alpha;
        vector<float> chi;
        vector<float> eta;
        unsigned long long coops;
        unsigned long long total_acts;
        bool keep_track;
        int available_threads;
        mt19937 mt;
};

#endif