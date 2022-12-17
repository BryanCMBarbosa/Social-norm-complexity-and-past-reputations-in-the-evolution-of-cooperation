#include "src/Individual.h"
#include "src/Simulation.h"
#include <bitset>
#include <stdlib.h>
#include <chrono>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    auto t1 = chrono::high_resolution_clock::now();

    if (argc == 6)
    {
        string argv_1(argv[1]);
        reverse(argv_1.begin(), argv_1.end());
        bitset<16> norm(argv_1);

        string norm_name(argv[2]);

        char *p_end_z;
        unsigned long z = strtoul(argv[3], &p_end_z, 10);

        char *p_end_gen;
        unsigned long long generations = strtoull(argv[4], &p_end_gen, 10);

        char *p_end_runs;
        unsigned long long n = strtoull(argv[5], &p_end_runs, 10);

        Simulation s = Simulation(norm, norm_name, z, generations);
        s.run_n_runs(n);
    }
    else if (argc > 6)
    {
        string argv_1(argv[1]);
        reverse(argv_1.begin(), argv_1.end());
        bitset<16> norm(argv_1);
        
        string norm_name(argv[2]);

        char *p_end_z;
        unsigned long z = strtoul(argv[3], &p_end_z, 10);

        char *p_end_gen;
        unsigned long long generations = strtoull(argv[4], &p_end_gen, 10);

        char *p_end_runs;
        unsigned long long n = strtoull(argv[5], &p_end_runs, 10);

        string other_args_order = string(argv[6]);

        float payoff_b = 5.0, payoff_c = 1.0, epsilon = 0.01, alpha = 0.01, chi = 0.01;

        for (short i = 7; i < argc; i++)
        {
            if (other_args_order[i-7] == 'b')
                payoff_b = atof(argv[i]);
            else if (other_args_order[i-7] == 'c')
                payoff_c = atof(argv[i]);
            else if (other_args_order[i-7] == 'e')
                epsilon = atof(argv[i]);
            else if (other_args_order[i-7] == 'a')
                alpha = atof(argv[i]);
            else if (other_args_order[i-7] == 'x')
                chi = atof(argv[i]);
        }

        Simulation s = Simulation(norm, norm_name, z, generations, payoff_b, payoff_c, epsilon, alpha, chi);
        s.run_n_runs(n);
    }
    else
        cout << "Not enough arguments were passed." << endl;

    auto t2 = chrono::high_resolution_clock::now();

    std::cout << "Time: "
              << chrono::duration_cast<chrono::milliseconds>(t2-t1).count()
              << " milliseconds." << endl;
              
    return 0;
}