import random
import matplotlib.pyplot as plt
import numpy as np

def generate(l, size):
    pop = [{'gene':[random.randint(0, 1) for i in range(l)]} for j in range(size)] #100
    # [{'gene': ~~}, {'gene': ~~}, ...]
    return pop

def crossover(mate1, mate2):
    gene = []
    point = random.randint(0, len(mate1['gene']) - 1)
    gene = mate1['gene'][:point] + mate2['gene'][point:]
    return gene

# tournament selection
def TS(pop):
    selection = []
    for i in range(len(pop)):
        versus = pop[random.randint(0, len(pop) - 1 )]
        if pop[i]['ham'] >= versus['ham']:
            selection.append(pop[i])
        else:
            selection.append(versus)
    return selection

def mutate(gene, pm):
    for i in gene:
        if random.random() <= pm:
            i = abs(i-1)
    return gene

def takeham(individual):
    return individual['ham']

def takefit(individual):
    return individual['fit']

# overlap selection
def OS(pop, tau):
    num = int(len(pop) * tau)
    pop.sort(key=takeham)
    pop = pop[num:]
    return pop

def hamming(D, ele1, ele2):
    d = 0
    for i in range(len(ele1['gene'])):
        if ele1['gene'][i] != ele2['gene'][i]:
            d += 1
    if d < D:
        d = 1-(d/D)
    else:
        d = 0
    return d

def fit(gene):
    s = sum(gene)
    if s >= int(len(gene)/2):
        return s
    else:
        return len(gene)-s

def fit_hamming(pop, D):
    for i in range(len(pop)):
        z = -1
        for j in range(len(pop)):
            h = hamming(D, pop[i], pop[j])
            z += h
        f = fit(pop[i]['gene'])
        z=z/2.5
        if z == 0:
            pop[i]['ham'] = f
        else:
            pop[i]['ham'] = f/z
    return pop

def test(pop):
    best = 0
    avg = 0
    for i in range(len(pop)):
        avg += pop[i]['fit']/len(pop)
        if pop[i]['fit'] >= best:
            best = pop[i]['fit']
            best_individual = pop[i]
    return best, avg, best_individual

# 0. initial setting
D = 30
pop_size = 100
NoG = 300
length = 50
pm = 0.01
tau = 0.5
bests = []
avgs = []

# 1. initial generation
pop = generate(length, pop_size)

for i in range(NoG):
    for j in pop:
        j['fit'] = fit(j['gene'])
    print(pop)
    # test
    best, avg, best_individual = test(pop)
    bests.append(best)
    avgs.append(avg)
    print('generation', i)
    print('best profit:', best)
    print('avg profit:', avg)
    print('gene:', best_individual['gene'])

    # 2. Overlap selection + TS selection
    pop = fit_hamming(pop, D)
    selected = TS(pop)
    pop1 = []
    # crossover
    for j in range(int(len(selected)/2)):
        individual = {}
        individual['gene'] = crossover(selected[j], selected[-(j+1)])
        pop1.append(individual)

    for j in pop:
        j['fit'] = fit(j['gene'])
    pop2 = OS(pop, tau)
    # mutate
    for individual in pop1:
        individual['gene'] = mutate(individual['gene'], pm)
    pop = pop1 + pop2

x = np.arange(0, NoG, 1)
plt.plot(x, bests, 'r', x, avgs, 'b')
plt.suptitle('Two_Max Problem')
plt.xlabel('generation')
plt.ylabel('quality')
plt.show()

for i in pop:
    i['fit'] = fit(i['gene'])
pop.sort(key=takefit)
print(pop[90:])