#include "Simulation.h"

Simulation::Simulation(bitset<16> norm, string norm_name, unsigned long z, unsigned long long generations = 1000, float payoff_b = 5, float payoff_c = 1, float epsilon = 0.01, float alpha = 0.01, float chi = 0.01) : mt((random_device())())
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
    this->coops_in_gen = 0;
    this->total_acts = 6*z*long(z*mi[1]);
}

void Simulation::create_agents()
{
    for(unsigned long i = 0; i < z; i++)
       individuals.emplace_back(i);
}

short Simulation::repcomb_to_index(Individual donor, Individual receptor, bool donor_action = false, bool use_donor_action = false)
{
    if (use_donor_action)
        return (8 * receptor.reputation[receptor.reputation.size()-2]) + (4 * donor.reputation[donor.reputation.size()-1]) + (2 * receptor.reputation[receptor.reputation.size()-1]) + (1 * donor_action);
    else
    {    
        //random_device rd;
        //mt19937 mt(rd());
        bernoulli_distribution dist(chi[0]);
        vector<bool> gossip_error(2);
        for(int i=0; i<gossip_error.size(); i++)
            gossip_error.push_back(dist(mt));

        bool receptor_rep_1 = gossip_error[0] ? !receptor.reputation[receptor.reputation.size()-1] : receptor.reputation[receptor.reputation.size()-1];
        bool receptor_rep_2 = gossip_error[1] ? !receptor.reputation[receptor.reputation.size()-2] : receptor.reputation[receptor.reputation.size()-2];

        return 4 * receptor_rep_2 + 2 * donor.reputation[donor.reputation.size()-1] + 1 * receptor_rep_1;
    }
}

void Simulation::judge(Individual x, Individual y, bool x_action, bool y_action)
{
    //random_device rd;
    //mt19937 mt(rd());
    bernoulli_distribution dist(alpha[0]);
    vector<bool> can_assign(2);
    for(int i=0; i<can_assign.size(); i++)
        can_assign.push_back(dist(mt));
    
    if (can_assign[0])
        x.reputation.push_back(norm[repcomb_to_index(x, y, x_action, true)]);
    else
        x.reputation.push_back(norm[!repcomb_to_index(x, y, x_action, true)]);

    if (can_assign[1])
        y.reputation.push_back(norm[repcomb_to_index(y, x, y_action, true)]);
    else
        x.reputation.push_back(norm[!repcomb_to_index(y, x, y_action, true)]);
}

void Simulation::match(Individual x, Individual y)
{
    bool x_act = x.act(repcomb_to_index(x, y), epsilon = epsilon);
    bool y_act = y.act(repcomb_to_index(y, x), epsilon = epsilon);

    if (x_act)
    {
        coops_in_gen++;
        x.payoffs.push_back(-payoff_c);
        y.payoffs.push_back(payoff_b);
    }
        
    if (y_act)
    {
        coops_in_gen++;
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

void Simulation::imitation(vector <Individual>& imit)
{
    vector<Individual> y_individuals;
    sample(individuals.begin(), individuals.end(), back_inserter(y_individuals), imit.size(), mt);

    for (unsigned long i = 0; i < imit.size(); i++)
    {
        vector<Individual> adversaries_x;
        vector<Individual> adversaries_y;

        sample(individuals.begin(), individuals.end(), back_inserter(adversaries_x), 2*z, mt);
        sample(individuals.begin(), individuals.end(), back_inserter(adversaries_y), 2*z, mt);

        Individual x_i = imit[i];
        Individual y_i = y_individuals[i];
        x_i.reset();
        y_i.reset();

        for(unsigned long j; j < 2*z; j++)
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

vector<vector<Individual>> Simulation::divide_mutation_imitation()
{
    vector<Individual> mutation_group;
    vector<Individual> imitation_group;
    vector<vector<Individual>> groups;
    
    bernoulli_distribution dist(mi[0]);
    bool decision_var;

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

void Simulation::run_generations()
{
    vector<vector<Individual>> groups;

    for(unsigned long long i = 0; i < generations; i++)
    {
        groups = divide_mutation_imitation();
        mutation(groups[0]);
        imitation(groups[1]);

        individuals.clear();
        individuals.reserve(groups[0].size()+groups[1].size());
        individuals.insert(individuals.end(), groups[0].begin(), groups[0].end());
        individuals.insert(individuals.end(), groups[1].begin(), groups[1].end());
        shuffle(individuals.begin(), individuals.end(), mt);

        coops_per_gen.push_back(coops_in_gen);
        coops_in_gen = 0;
        groups.clear();
    }
}

void Simulation::run_n_runs(unsigned long long runs)
{
    vector<double> eta_each_gen;
    vector<vector<double>> eta_each_run;

    double total_acts_inverse = 1/total_acts;
    
    for(unsigned long long i = 0; i < runs; i++)
    {
        run_generations();
        transform(coops_per_gen.begin(), coops_per_gen.end(), eta_each_gen.begin(), [total_acts_inverse](double val) {return val * total_acts_inverse;});
        eta_each_run.push_back(eta_each_gen);

        coops_per_gen.clear();
        eta_each_gen.clear();
        //clear_agents();
        individuals.clear();
        create_agents();
        cout << "Run " << i+1 << "finished." << endl;
    }

    turn_to_csv(runs, eta_each_run);
}

void Simulation::turn_to_csv(unsigned long long runs, vector<vector<double>> eta_each_run)
{
    fstream csv_file;
    string file_name = "z="+to_string(z)+"_norm="+norm_name+"_runs="+to_string(runs)+"_gen="+to_string(generations)+".csv";
    csv_file.open(file_name, ios::out | ios::app);

    for(unsigned long long i = 0; i < runs; i++)
        csv_file << "run_" << i+1 << ",";
    csv_file << "\n";

    for(unsigned long long i = 0; i < generations; i++)
    {
        for(unsigned long long j = 0; j < runs; j++)
        {
            csv_file << "run_" << eta_each_run[j][i] << ",";
        }
        csv_file << "\n";
    }
    csv_file.close();
}