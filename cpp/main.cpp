#include "src/Individual.h"
#include "src/Simulation.h"
#include <bitset>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    bitset<16> norm("1001100110011001");
    Simulation s = Simulation(norm, "Stern-judging", 120, 5000);
    s.run_n_runs(10);

    return 0;
}