#include "Simulation.h"

Simulation::Simulation(bitset<16> norm, string norm_name, unsigned long z, unsigned long long generations, float payoff_b, float payoff_c, float epsilon, float alpha, float chi) : mt((random_device())())
{
    this->norm = norm;
    this->norm_name = norm_name;
    this->z = z;
    this->mi = vector<float>{ 1/float(z), 1-(1/float(z))};
    this->generations = generations;
    this->payoff_b = payoff_b;
    this->payoff_c = payoff_c;
    this->epsilon = vector<float>{ epsilon, 1-epsilon };
    this->alpha = vector<float>{ alpha, 1-alpha };
    this->chi = vector<float>{ chi, 1-chi };
    this->coops = 0;
    this->total_acts = 0;
    this->keep_track = false;
    create_agents();
}

void Simulation::create_agents()
{
    for(unsigned long i = 0; i < z; i++)
       individuals.emplace_back(i);
}

short Simulation::repcomb_to_index(Individual donor, Individual receptor, bool donor_action, bool use_donor_action)
{
    if (use_donor_action)
        return (8 * receptor.reputation[receptor.reputation.size()-2]) + (4 * donor.reputation[donor.reputation.size()-1]) + (2 * receptor.reputation[receptor.reputation.size()-1]) + (1 * donor_action);
    else
    {    
        //random_device rd;
        //mt19937 mt(rd());
        bernoulli_distribution dist(chi[0]);
        bitset<2> gossip_error;
        for(short i=0; i<gossip_error.size(); i++)
            gossip_error[i] = dist(mt);

        bool receptor_rep_1 = gossip_error[0] ? !receptor.reputation[receptor.reputation.size()-1] : receptor.reputation[receptor.reputation.size()-1];
        bool receptor_rep_2 = gossip_error[1] ? !receptor.reputation[receptor.reputation.size()-2] : receptor.reputation[receptor.reputation.size()-2];

        return (4 * receptor_rep_2) + (2 * donor.reputation[donor.reputation.size()-1]) + (1 * receptor_rep_1);
    }
}

void Simulation::judge(Individual x, Individual y, bool x_action, bool y_action)
{
    //random_device rd;
    //mt19937 mt(rd());
    bernoulli_distribution dist(alpha[0]);
    bitset<2> can_assign;
    for(short i=0; i<can_assign.size(); i++)
        can_assign[i] = dist(mt);
    
    if (can_assign[0])
        x.reputation.push_back(norm[repcomb_to_index(x, y, x_action, true)]);
    else
        x.reputation.push_back(norm[!repcomb_to_index(x, y, x_action, true)]);

    if (can_assign[1])
        y.reputation.push_back(norm[repcomb_to_index(y, x, y_action, true)]);
    else
        y.reputation.push_back(norm[!repcomb_to_index(y, x, y_action, true)]);
}

void Simulation::match(Individual x, Individual y)
{
    bool x_act = x.act(repcomb_to_index(x, y), epsilon);
    bool y_act = y.act(repcomb_to_index(y, x), epsilon);
    total_acts += keep_track * 2;

    if (x_act)
    {
        coops += keep_track * 1;
        x.payoffs.push_back(-payoff_c);
        y.payoffs.push_back(payoff_b);
    }
        
    if (y_act)
    {
        coops += keep_track * 1;
        y.payoffs.push_back(-payoff_c);
        x.payoffs.push_back(payoff_b);
    }

    judge(x, y, x_act, y_act);
}

void Simulation::mutation(vector<Individual>& mut)
{
    for (unsigned long i = 0; i < mut.size(); i++)
        mut[i].generate_strategy();
}

vector<Individual> Simulation::sample_with_reposition(vector<Individual>& vec, unsigned long long sample_size)
{
    vector<Individual> sample;
    uniform_int_distribution<> dist(0, vec.size()-1);
    for (unsigned long long i = 0; i < sample_size; i++)
    {
        unsigned long long index = dist(mt);
        sample.push_back(vec[index]);
    }

    return sample;
}

void Simulation::imitation(vector <Individual>& imit)
{
    vector<Individual> y_individuals;
    y_individuals = sample_with_reposition(individuals, imit.size());
    
    #pragma omp parallel num_threads(8)                 
    {
        #pragma omp for
        for (unsigned long i = 0; i < imit.size(); i++)
        {
            vector<Individual> adversaries_x;
            vector<Individual> adversaries_y;

            adversaries_x = sample_with_reposition(individuals, 2*z);
            adversaries_y = sample_with_reposition(individuals, 2*z);

            Individual x_i = imit[i];
            Individual y_i = y_individuals[i];

            x_i.reset();
            y_i.reset();

            for(unsigned long j = 0; j < 2*z; j++)
            {
                match(x_i, adversaries_x[j]);
                match(y_i, adversaries_y[j]);
            }

            double prob_imitation = 1 / (1 + exp(x_i.fitness - y_i.fitness));
        
            bernoulli_distribution dist(prob_imitation);
            bool must_imit = dist(mt);
        
            if (must_imit)
                imit[i].strategy = y_i.strategy;
        }
    }
}

vector<vector<Individual>> Simulation::divide_mutation_imitation()
{
    vector<Individual> mutation_group;
    vector<Individual> imitation_group;
    vector<vector<Individual>> groups;
    
    uniform_real_distribution<> dist(0, 1.0);
    double decision_var;

    for(unsigned long long i = 0; i < z; i++)
    {
        decision_var = dist(mt);
        
        if (decision_var < mi[0])
            mutation_group.push_back(individuals[i]);
        else
            imitation_group.push_back(individuals[i]);
    }

    groups.push_back(mutation_group);
    groups.push_back(imitation_group);

    return groups;
}

vector<double> Simulation::run_generations()
{
    vector<vector<Individual>> groups;
    vector<double> eta_each_gen;
    double eta;

    for(unsigned long long i = 0; i < generations; i++)
    {
        cout << "Gen. " << i+1 << " of " << generations << " started." << endl;

        keep_track = i > 0.2*generations;

        groups = divide_mutation_imitation();

        mutation(groups[0]);
        cout << "Mutation done!" << endl;

        imitation(groups[1]);
        cout << "Imitation done!" << endl;

        individuals.clear();
        
        individuals.reserve(groups[0].size()+groups[1].size());
        individuals.insert(individuals.end(), groups[0].begin(), groups[0].end());
        individuals.insert(individuals.end(), groups[1].begin(), groups[1].end());
        //shuffle(individuals.begin(), individuals.end(), mt);

        groups.clear();

        if (keep_track)
        {
            eta = double(coops) / double(total_acts);
            eta_each_gen.push_back(eta);
        }

        coops = 0;
        total_acts = 0;
        cout << "Gen. " << i+1 << " of " << generations << " finished." << endl << endl;
    }

    return eta_each_gen;
}

void Simulation::run_n_runs(unsigned long long runs)
{
    vector<double> eta_each_gen;
    vector<vector<double>> eta_each_run;

    for(unsigned long long i = 0; i < runs; i++)
    {
        cout << endl << endl << endl << endl << "Run " << i+1 << " started." << endl;

        eta_each_gen = run_generations();
        eta_each_run.push_back(eta_each_gen);

        individuals.clear();
        create_agents();

        cout << "Run " << i+1 << " finished." << endl;
    }

    cout << "Turning to CSV..." << endl;
    turn_to_csv(runs, eta_each_run);
    cout << "Done!" << endl;
}

void Simulation::turn_to_csv(unsigned long long runs, vector<vector<double>> eta_each_run)
{
    fstream csv_file;
    string file_name = "z="+to_string(z)+"_norm="+norm_name+"_runs="+to_string(runs)+"_gen="+to_string(generations)+".csv";
    csv_file.open(file_name, ios::out | ios::app);

    for(unsigned long long i = 0; i < runs; i++)
    {
        csv_file << "run_" << i+1;
        if (!(i == runs-1))
            csv_file << ",";
    }
    csv_file << "\n";

    for(unsigned long long i = 0; i < eta_each_run[0].size(); i++)
    {
        for(unsigned long long j = 0; j < runs; j++)
        {
            csv_file << eta_each_run[j][i];
            if (!(j == runs-1))
                csv_file << ",";
        }
        csv_file << "\n";
    }
    csv_file.close();
}