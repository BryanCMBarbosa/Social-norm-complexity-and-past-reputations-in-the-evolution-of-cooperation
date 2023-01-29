import sys

norms = {
    "All_bad": "0000000000000000",
    "All-good": "1111111111111111",
    "Shunning": "1000100010001000",
    "Judging": "1001100010011000",
    "Stern-judging": "1001100110011001",
    "Score-judging": "1001101010011010",
    "SJ+SS": "1001101110011011",
    "Image-score": "1010101010101010",
    "Strict-standing": "1011100010111000",
    "SS+SJ": "1011100110111001",
    "Standing": "1011101010111010",
    "Simple-standing": "1011101110111011",
    }
population_inf = sys.argv[1]
population_sup = sys.argv[2]
generations = sys.argv[3]
runs = sys.argv[4]

scripts = []

for name, code in zip(norms.keys(), norms.values()):
    file = open(f"{name}_{code}.sh", "w")
    file.write("#!/bin/bash")
    file.write('\n')
    scripts.append(f"{name}_{code}.sh")   
    file.write(f"srun ./simulation {code} {name} $SLURM_ARRAY_TASK_ID {generations} {runs}")
    file.write('\n')
file.close()

file = open("a_script_to_rule_them_all.sh", "w")
file.write("#!/bin/bash")
file.write('\n')
for s in scripts:
    file.write(f"sbatch --array={population_inf}-{population_sup}:10 {s}")
    file.write('\n')
file.close()