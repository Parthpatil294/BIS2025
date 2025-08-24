import random
import pandas as pd

def to_binary(x, bits=5):
    return format(x, f'0{bits}b')

def to_decimal(b):
    return int(b, 2)

def fitness(x):
    return x**2

def initial_population():
    print("Enter 4 numbers between 0 and 31 (inclusive) for initial population:")
    pop = []
    for i in range(4):
        while True:
            try:
                val = int(input(f"Chromosome {i+1}: "))
                if 0 <= val <= 31:
                    pop.append(val)
                    break
                else:
                    print("Value must be between 0 and 31.")
            except ValueError:
                print("Please enter an integer.")
    return pop

population = initial_population()

data = []
fit_vals = [fitness(x) for x in population]
total_fit = sum(fit_vals)

for i, x in enumerate(population):
    chrom = to_binary(x)
    f = fitness(x)
    prob = f/total_fit if total_fit != 0 else 0
    data.append([i+1, chrom, x, f, prob, prob*100, prob*4, None])

probs = [row[4] for row in data]
selected = random.choices(population, weights=probs, k=4) if total_fit != 0 else random.choices(population, k=4)

for i in range(4):
    data[i][-1] = selected.count(population[i])

initial_df = pd.DataFrame(data, columns=["String No","Chromosome","X","Fitness","Prob","%Prob","Expected Value","Actual Count"])

print("\n=== Initial Population Table ===")
print(initial_df.to_string(index=False))

mating_pool = [to_binary(x) for x in selected]
sel_data = []

for i in range(0, len(mating_pool)-1, 2):
    cp = random.randint(1,4)
    mate1, mate2 = mating_pool[i], mating_pool[i+1]
    off1 = mate1[:cp] + mate2[cp:]
    off2 = mate2[:cp] + mate1[cp:]
    sel_data.append([i+1, mate1, cp, off1, to_decimal(off1), fitness(to_decimal(off1))])
    sel_data.append([i+2, mate2, cp, off2, to_decimal(off2), fitness(to_decimal(off2))])

sel_df = pd.DataFrame(sel_data, columns=["String No","Mating Pool","Crossover Point","Offspring","X","Fitness"])

print("\n=== Selection & Crossover Table ===")
print(sel_df.to_string(index=False))

mut_data = []
offspring = sel_df["Offspring"].tolist()

for i, chrom in enumerate(offspring):
    mp = random.randint(0,4)
    mutated = list(chrom)
    mutated[mp] = '1' if mutated[mp]=='0' else '0'
    mutated = ''.join(mutated)
    mut_data.append([i+1, chrom, mp, mutated, to_decimal(mutated), fitness(to_decimal(mutated))])

mut_df = pd.DataFrame(mut_data, columns=["String No","Offspring Before Mutation","Mutation Point","Mutated Chromosome","X","Fitness"])

print("\n=== Mutation Table ===")
print(mut_df.to_string(index=False))

best_solution = mut_df.loc[mut_df["Fitness"].idxmax()]
print("\n=== Best Solution Found ===")
print(best_solution.to_string())
