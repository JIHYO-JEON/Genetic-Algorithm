import random
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    capacity = int(lines[2].split()[-1])
    print ('Total weight capacity:', capacity)
    items = []
    for line in lines[5:]:
        ctxt = line.split()
        item = {'item':ctxt[0], 'weight':ctxt[1], 'profit':ctxt[2]}
        items.append(item)
    # print (items)
    return items, capacity

# selection type
# START ) Generate random population!
def generate(data, size):
    l = len(data)
    pop = [{'gene':[random.randint(0, 1) for i in range(l)]} for j in range(size)]
    # [{'gene': ~~}, {'gene': ~~}, ...]
    return pop

# Fitness ) Evaluate the fitness of each chromosome x in population
def fit_cal(data, pop, capacity):
    for i in range(len(pop)):
        gene = pop[i]['gene']
        weight = 0
        profit = 0
        for j in range(len(gene)):
            if gene[j] == 1:
                weight += int(data[j]['weight'])
                profit += int(data[j]['profit'])
        if weight > capacity:
            profit = 0
        pop[i]['weight'] = weight
        pop[i]['profit'] = profit
    return pop

def crossover(pop, pc):
    kids_gen = []
    for i in range(len(pop)):
        mom = pop[i]
        dad = pop[-(i + 1)]
        if random.random() <= pc:
            point1 = random.randint(0, len(mom['gene'])-1)
            point2 = random.randint(point1, len(mom['gene']) - 1)
            point3 = random.randint(point2, len(mom['gene']) - 1)
            kid = {}
            kid['gene'] = mom['gene'][:point1] + dad['gene'][point1:point2] + mom['gene'][point2:point3] + dad['gene'][point3:]
            kids_gen.append(kid)
        else:
            kids_gen.append(mom)
            kids_gen.append(dad)
    #print(len(kids_gen))
    return kids_gen

# bit-wise mutation
def mutate(pop, pm):
    for i in range(len(pop)):
        mutant = pop[i]
        for j in mutant['gene']:
            if random.random() <= pm:
                j = abs(j-1)
    return pop

# roulette wheel selection
def RWS(pop):
    selection = []
    profit_sum = 0
    for i in range(len(pop)):
        profit_sum += pop[i]['profit']
    relative_profit = [pop[i]['profit']/profit_sum for i in range(len(pop))]
    prob = [sum(relative_profit[:i+1]) for i in range(len(relative_profit))]
    for i in range(len(pop)):
        x = random.random()
        for j in range(len(pop)):
            if x <= prob[j]:
                selection.append(pop[j])
                break
    # print(len(selection))
    return selection

# tournament selection
def TS(pop):
    selection = []
    for i in range(len(pop)):
        versus = pop[random.randint(0, len(pop) - 1 )]
        if pop[i]['profit'] >= versus['profit']:
            selection.append(pop[i])
        else:
            selection.append(versus)
    # print(len(selection))
    return selection

def test(pop):
    best = 0
    avg = 0
    for i in range(len(pop)):
        avg += pop[i]['profit']/len(pop)
        if pop[i]['profit'] >= best:
            best = pop[i]['profit']
            best_individual = pop[i]
    return best, avg, best_individual

pc = 0.9 # crossover prob.
pm = 0.01 # mutation prob.
N = 100 # population size
NoG = 100 # number of generations

bests = []
avgs = []

#0. Load data
data, capacity = load_data('testdata.txt')

#1. Generate individuals
population = generate(data, N)

for i in range(NoG):
    # 2. Calculate weight and profit
    population = fit_cal(data, population, capacity)

    # 3. Test
    best, avg, best_individual = test(population)
    bests.append(best)
    avgs.append(avg)
    print('generation', i)
    print('best profit:', best)
    print('avg profit:', avg)
    print('gene:', best_individual['gene'])

    # 4. New population
    # 4.1. selection
    algo = 'rws'
    if algo == 'rws':
        selected_pop = RWS(population)
    else:
        selected_pop = TS(population)

    # 4.2. crossover
    # 4.3. mutate
    new_generation = mutate(crossover(selected_pop, pc), pm)

    #5. Accept and replace
    population = new_generation

x = np.arange(0, NoG, 1)
plt.plot(x, bests, 'r', x, avgs, 'b')
if algo == 'rws':
    plt.suptitle('Roulette wheel selection')
else:
    plt.suptitle('Tournament selection')
plt.xlabel('generation')
plt.ylabel('profit')
plt.show()
print(best_individual)
