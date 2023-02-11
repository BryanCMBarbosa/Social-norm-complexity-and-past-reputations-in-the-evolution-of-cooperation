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
    this->available_threads = omp_get_num_procs();
    omp_set_num_threads(available_threads);
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
        return (8 * receptor.reputation.front()) + (4 * donor.reputation.front()) + (2 * receptor.reputation.back()) + (1 * donor_action);
    else
    {    
        bernoulli_distribution dist(chi[0]);
        bitset<2> gossip_error;
        for(short i=0; i < gossip_error.size(); i++)
            gossip_error[i] = dist(mt);
        
        bool receptor_rep_1 = gossip_error[0] ? !receptor.reputation.back() : receptor.reputation.back();
        bool receptor_rep_2 = gossip_error[1] ? !receptor.reputation.front() : receptor.reputation.front();

        return (4 * receptor_rep_2) + (2 * donor.reputation.back()) + (1 * receptor_rep_1);
    }
}

void Simulation::judge(Individual& x, Individual& y, bool x_action, bool y_action)
{
    bernoulli_distribution dist(alpha[0]);
    bitset<2> can_assign;
    for(short i=0; i<can_assign.size(); i++)
        can_assign[i] = dist(mt);
    
    x.reputation.pop();
    y.reputation.pop();

    if (can_assign[0])
        x.reputation.push(norm[repcomb_to_index(x, y, x_action, true)]);
    else
        x.reputation.push(norm[!repcomb_to_index(x, y, x_action, true)]);

    if (can_assign[1])
        y.reputation.push(norm[repcomb_to_index(y, x, y_action, true)]);
    else
        y.reputation.push(norm[!repcomb_to_index(y, x, y_action, true)]);
}

void Simulation::match(Individual& x, Individual& y)
{
    bool x_act = x.act(repcomb_to_index(x, y), epsilon);
    bool y_act = y.act(repcomb_to_index(y, x), epsilon);
    total_acts += keep_track * 2;

    if (x_act)
    {
        coops += keep_track * 1;
        x.add_payoff(-payoff_c);
        y.add_payoff(payoff_b);
    }
        
    if (y_act)
    {
        coops += keep_track * 1;
        y.add_payoff(-payoff_c);
        x.add_payoff(payoff_b);
    }

    judge(x, y, x_act, y_act);
}

void Simulation::mutation(vector<unsigned long long>& indexes)
{
    for (unsigned long long i:indexes)
        individuals[i].generate_strategy();
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

unsigned long Simulation::get_sampled_individual(vector<Individual>& population, long exception_id)
{
    uniform_int_distribution<> dist(0, population.size()-1);
    long index, sampled_id;
    long i = 0;
    if (exception_id != -1)
    {
        do
        {
            index = dist(mt);
            sampled_id = population[index].id;
        } while (sampled_id == exception_id);
    }
    else
        index = dist(mt);

    return index;
}

void Simulation::imitation(vector <unsigned long long>& indexes)
{
    vector<Individual> y_individuals;
    y_individuals = sample_with_reposition(individuals, indexes.size());
    map<unsigned long, bitset<strategy_length>> imit_index_and_strat;
    int size = indexes.size();

    #pragma omp parallel private(individuals, indexes)
    { 
        #pragma omp for
        for (unsigned long long i = 0; i < size; i++)
        {
            Individual x_i = individuals[indexes[i]];
            Individual y_i = y_individuals[i];

            x_i.reset_payoff();
            y_i.reset_payoff();

            for(unsigned long j = 0; j < 2*z; j++)
            {
                match(x_i, individuals[get_sampled_individual(individuals, x_i.id)]);
                match(y_i, individuals[get_sampled_individual(individuals, y_i.id)]);
            }

            double prob_imitation = 1 / (1 + exp(x_i.get_fitness(z) - y_i.get_fitness(z)));

            bernoulli_distribution dist(prob_imitation);
            bool must_imit = dist(mt);
        
            if (must_imit)
                individuals[indexes[i]].strategy = y_i.strategy;   
        }
    }
    /*
    for (auto it = imit_index_and_strat.begin(); it != imit_index_and_strat.end(); it++)
    {
        cout << it->first << endl;
        cout << it->second << endl;
        individuals[it->first].strategy = it->second;
    }
    */
}

vector<vector<unsigned long long>> Simulation::divide_mutation_imitation()
{
    vector<unsigned long long> mutation_group;
    vector<unsigned long long> imitation_group;
    vector<vector<unsigned long long>> groups;
    
    uniform_real_distribution<> dist(0, 1.0);
    double decision_var;

    for(unsigned long long i = 0; i < z; i++)
    {
        decision_var = dist(mt);
        
        if (decision_var < mi[0])
            mutation_group.push_back(i);
        else
            imitation_group.push_back(i);
    }

    groups.push_back(mutation_group);
    groups.push_back(imitation_group);

    return groups;
}

vector<double> Simulation::run_generations(unsigned long long run, unsigned long long runs)
{
    vector<vector<unsigned long long>> groups;
    vector<double> eta_each_gen;
    double eta;

    for(unsigned long long i = 0; i < generations; i++)
    {
        keep_track = i > 0.2*generations;
        

        groups = divide_mutation_imitation();

        mutation(groups[0]);

        imitation(groups[1]);

        
        groups.clear();

        if (keep_track)
        {
            eta = double(coops) / double(total_acts);
            eta_each_gen.push_back(eta);
        }

        coops = 0;
        total_acts = 0;
    }

    return eta_each_gen;
}

void Simulation::run_n_runs(unsigned long long runs)
{
    vector<double> eta_each_gen;
    vector<vector<double>> eta_each_run;

    for(unsigned long long i = 0; i < runs; i++)
    {
        cout << endl << "Run " << i+1 << " started." << endl;

        eta_each_gen = run_generations(i, runs);
        eta_each_run.push_back(eta_each_gen);

        individuals.clear();
        create_agents();
        
        cout << "Run " << i+1 << " finished." << endl;
        cout << "::::::::::::::::::::::::::::::::::::::::::" << endl;
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