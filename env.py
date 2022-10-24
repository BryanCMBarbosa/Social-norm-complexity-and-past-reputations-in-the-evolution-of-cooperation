import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

class individual():
    def __init__(self, id, strategy = np.array([])):
        self.id = id
        self.strategy = strategy
        self.payoffs = np.array([])
        self.fitness = 0.0
        self.reputation = np.random.randint(2, size=2)

    def act(self, repcomb_index, epsilon):
        can_execute = np.random.choice([False, True], 1, p = epsilon)[0]
        if can_execute:
            return self.strategy[repcomb_index]
        else:
            return self.reverse_act(self.strategy[repcomb_index])

    def reverse_act(self, action):
        if action == 1:
            return 0
        else:
            return 1

    def add_payoff(self, p):
        self.payoffs = np.append(self.payoffs, p)
        self.fitness = self.payoffs.mean()

    def reset(self):
        self.payoffs = np.array([])
        self.fitness = 0.0
        self.reputation = np.random.randint(2, size=2)

class env():
    def __init__(self, norm, z = 60, mi = [], gen = 1000, payoff_b = 5, payoff_c = 1, epsilon = 0.01, alpha = 0.01, chi = 0.01):
        self.reputation_layer = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        self.reputation_with_donor_action = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
        self.z = z

        if not mi:
            self.mi = [1/z, 1-(1/z)]
        else:
            self.mi = mi

        self.gen = int(gen)
        self.social_norms = {"Stern-judging": np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]), 
                            "Judging": np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]), 
                            "Strict-standing": np.array([1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0]), 
                            "SS+SJ": np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1]), 
                            "Simple-standing": np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]), 
                            "Score-judging": np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]), 
                            "Standing": np.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0]), 
                            "Image-score": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]), 
                            "SJ+SS": np.array([1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1]), #SS+SS only appears 1 time in the paper 
                            "All good": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
        self.norm = self.social_norms[norm]
        self.norm_name = norm
        self.individuals = np.array([])
        self.create_agents()
        self.coops = []
        self.total_coops = []
        self.payoff_b = payoff_b
        self.payoff_c = payoff_c
        self.epsilon = [epsilon, 1-epsilon]
        self.alpha = [alpha, 1-alpha]
        self.chi = [chi, 1-chi]
        self.eta = np.array([])
        self.coops_in_g = 0
        self.coops_per_generation = np.array([])
        self.total_acts = 2*2*(2*self.z)*(self.z*self.mi[1]) #2 matches (x vs o_i and y vs o_i) TIMES 2 acts (one for x, one for o_i) TIMES the size of other indiv TIMES the size of imit

    def create_agents(self):  
        for i in range(self.z):
            s = np.random.randint(2, size=8)
            indiv = individual(i, s)
            self.individuals = np.append(self.individuals, indiv)

    def bin_to_dec(self, bin_array):
        return int(bin_array.dot(1 << np.arange(bin_array.size)[::-1]))

    def repcomb_to_index(self, donor, receptor, donor_action = -1):
        if donor_action != -1:
            return self.bin_to_dec(np.array([receptor.reputation[-2], donor.reputation[-1], receptor.reputation[-1], donor_action]))
        else:
            gossip_error1 = np.random.choice([False, True], 1, p=self.chi)
            gossip_error2 = np.random.choice([False, True], 1, p=self.chi)

            if gossip_error1:
                receptor_rep_1 = self.reverse_value(receptor.reputation[-1])
            else:
                receptor_rep_1 = receptor.reputation[-1]

            if gossip_error2:
                receptor_rep_2 = self.reverse_value(receptor.reputation[-2])
            else:
                receptor_rep_2 = receptor.reputation[-2]

            return self.bin_to_dec(np.array([receptor_rep_2, donor.reputation[-1], receptor_rep_1]))
        
    def judge(self, x, y, x_action, y_action):
        can_assign = np.random.choice([False, True], 2, p=self.alpha)
        if can_assign[0]:
            x.reputation = np.append(x.reputation, self.norm[self.repcomb_to_index(x, y, x_action)])
        else:
            x.reputation = np.append(x.reputation, self.reverse_value(self.norm[self.repcomb_to_index(x, y, x_action)]))

        if can_assign[1]:
            y.reputation = np.append(y.reputation, self.norm[self.repcomb_to_index(y, x, y_action)])
        else:
            y.reputation = np.append(y.reputation, self.reverse_value(self.norm[self.repcomb_to_index(y, x, y_action)]))

    def reverse_value(self, value):
        if value == 1:
            return 0
        else:
            return 1

    def match(self, x, y):
        x_act = x.act(self.repcomb_to_index(x, y), epsilon = self.epsilon)
        y_act = y.act(self.repcomb_to_index(y, x), epsilon = self.epsilon)
        
        if x_act == 1:
            self.coops_in_g += 1
            x.payoffs = np.append(x.payoffs, -self.payoff_c)
            y.payoffs = np.append(y.payoffs, self.payoff_b) #Decrement from donor?

        if y_act == 1:
            self.coops_in_g += 1
            y.payoffs = np.append(y.payoffs, -self.payoff_c)
            x.payoffs = np.append(x.payoffs, self.payoff_b) #Decrement from donor?

        self.judge(x, y, x_act, y_act)

    def plot_eta(self, eta):
        for i in range(len(eta)):
            l = "Run " + str(i+1)
            plt.plot(range(len(eta[i])), eta[i], label = l, lw=0.5)
        plt.legend()
        plt.show()

    def mutation(self, mut):
        for i in range(mut.shape[0]):
            mut[i].strategy = np.random.randint(2, size=8)
        return mut

    def imitation(self, whole_population, imit):
        y_indiv = np.random.choice(whole_population, imit.shape[0], replace=False)
        indexes = np.arange(y_indiv.shape[0])

        imit = Parallel(n_jobs=-1, backend='threading')(delayed(self.imit_operation)(x, y) for x, y in zip(imit, y_indiv))
        
        return imit

        '''
        for index, x, y in zip(indexes, imit, y_indiv):
            i_x = x
            i_y = y
            i_x.reset()
            i_y.reset()
            for o_i_x, o_i_y in other_indiv:
                self.match(i_x, o_i_x)
                self.match(i_y, o_i_y)

            prob_imitation = 1 / (1 + np.exp(i_x.fitness - i_y.fitness))
            must_imit = np.random.choice([True, False], 1, p = [prob_imitation, 1-prob_imitation])[0]
            if must_imit:
                i_x.strategy = i_y.strategy
                imit[index] = i_x
        return imit
        '''

    def imit_operation(self, x, y):
        other_indiv_x = np.random.choice(self.individuals, 2*self.z, replace=True)
        other_indiv_y = np.random.choice(self.individuals, 2*self.z, replace=True)
        i_x = x
        i_y = y
        i_x.reset()
        i_y.reset()
        for o_i_x, o_i_y in zip(other_indiv_x, other_indiv_y):
            self.match(i_x, o_i_x)
            self.match(i_y, o_i_y)

        prob_imitation = 1 / (1 + np.exp(i_x.fitness - i_y.fitness))
        must_imit = np.random.choice([True, False], 1, p = [prob_imitation, 1-prob_imitation])[0]

        if must_imit:
            i_x.strategy = i_y.strategy

        return i_x

    def run_gens(self):
        for g in range(self.gen):
            #create mask, choose random indices from x according to pdf, set chosen indices to True:
            indexes = np.full(self.z, False, bool)
            indexes[np.random.choice(np.arange(indexes.shape[0]), self.z, replace = False)] = True
            revising_strat_indiv = self.individuals[indexes]
            not_revising_strat_indiv = self.individuals[~indexes]

            mut_individuals = revising_strat_indiv[0:int(self.z*self.mi[0])]
            imit_individuals = revising_strat_indiv[int(self.z*self.mi[0]):]
            
            mut_individuals = self.mutation(mut_individuals)
            imit_individuals = self.imitation(self.individuals, imit_individuals)
            self.individuals = np.concatenate([not_revising_strat_indiv, mut_individuals, imit_individuals])
             
            if g >= int(0.2*self.gen):
                self.coops_per_generation = np.append(self.coops_per_generation, self.coops_in_g)
            self.coops_in_g = 0
    
    def clear_agents(self):
        for a in self.individuals:
            a.reset()
            a.strategy = np.random.randint(2, size=8)

    def turn_to_dataframe(self, eta_runs_dict):
        return pd.DataFrame(eta_runs_dict)

    def turn_to_csv(self, df):
        df.to_csv("z="+str(self.z)+"_norm="+self.norm_name+"_runs="+str(self.runs)+"_gen="+str(self.gen)+".csv")

    def n_runs(self, runs):
        self.runs = runs
        eta_n_runs = []
        eta_runs_dict = dict()
        for i in range(runs):
            self.run_gens()
            self.eta = self.coops_per_generation / self.total_acts
            eta_n_runs.append(self.eta)
            eta_runs_dict["run_"+str(i)] = self.eta
            self.eta = np.array([])
            self.coops_per_generation = np.array([])
            self.clear_agents()
            self.individuals = []
            self.create_agents()
            print(i+1, "run(s).")
        df = self.turn_to_dataframe(eta_runs_dict)
        self.turn_to_csv(df)
        print(df)
        self.plot_eta(eta_n_runs)


if __name__ == '__main__':
    e = env(z = 16, norm = "Stern-judging", gen=20)
    e.n_runs(1)