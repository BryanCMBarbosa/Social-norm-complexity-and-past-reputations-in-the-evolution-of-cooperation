#include "src/Individual.h"
#include "src/Simulation.h"
#include <bitset>
#include <stdlib.h>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    clock_t start, end;
    
    start = clock();

    if (argc == 6)
    {
        string argv_1(argv[1]);
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

    end = clock();
     // Calculating total time taken by the program.
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Time elapsed: " << fixed 
         << time_taken;
    cout << " sec " << endl;

    return 0;
}