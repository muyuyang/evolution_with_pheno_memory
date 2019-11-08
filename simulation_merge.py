import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os

# Parameters 
N = 100000
num_allele = 4
num_steps = 100000
mutation_rate = 1e-5
recomb_rate = 0.5
period = 15
s_mid = 0.4
p = 0    # Phenotypic memory
var = 0.3

class VarPheno(object):
    """docstring for VarPheno"""
    def __init__(self):
        self.freq = np.array([0.0,0.0])
        self.p = [0.6,0.4]

    def mutate(self):
        j = random.random()
        if j < 0.5:
            self.freq[0] = 1/N
        else:
            self.freq[1] = 1/N

    def select(self,s):
        f = np.array([1-(s-var),1-(s+var)])
        self.freq = np.multiply(self.freq,f)
        return self.sum()

    def normalize(self,Z):
        self.freq = self.freq / Z

    def sum(self):
        return np.sum(self.freq)

    def reproduce(self):
        total = self.sum()
        if total != 0:
            newGen = np.zeros(2)
            newGen[0] = self.freq[0] * p + self.freq[0] * (1-p) * self.p[0] +\
                        self.freq[1] * (1-p) * self.p[0]
            newGen[1] = self.freq[1] * p + self.freq[0] * (1-p) * self.p[1] +\
                        self.freq[1] * (1-p) * self.p[1]
            self.freq = newGen


    def frac(self):
        return self.freq / self.sum()

        

# A = VarPheno()
# A.freq = np.array([0.5,0.5])
# A.normalize(10)
# print(A.freq)


# Cyclic selection using sine function
def s(t):
    pt = t // period
    if pt % 2 == 0:
        return s_mid
    else:
        return -s_mid


# Returns a noise with distribution from N(1,1)
def eps():
    return np.random.normal(loc=1)


def simulation():
    # Initialization
    AH = np.zeros(num_steps) # Record heterozygosity at every time step
    BH = np.zeros(num_steps)
    ah = 0
    bh = 0
    ah_cumu = 0  # Record expected cumulative heterozygosity
    bh_cumu = 0


    freq = np.array([1.0,0.0,0.0,0,0])  # ab, aB, Ab, AB
    aB_freq = VarPheno()
    AB_freq = VarPheno()
    freq_history = []

    # Introduce a mutation in either locus
    i = random.random()
    if i < 0.5:   # Introduce at A
        freq[2] = 1/N
        freq[0] -= 1/N
        mutated = 0
    else:
        freq[1] = 1/N
        aB_freq.mutate()
        freq[0] -= 1/N
        mutated = 1



    for t in range(num_steps):
        # Mutation
        # If mutation happens at A
        freq = freq / np.sum(freq)
        if mutated == 0:
            i = random.random()
            if i < mutation_rate * N:
                freq[1] = 1/N
                aB_freq.mutate()
                freq[0] -= 1/N
                mutated = 2
        elif mutated == 1:
            i = random.random()
            if i < mutation_rate * N:
                freq[2] += 1/N
                freq[0] -= 1/N
                mutated = 2

        # print(freq)
        # print('freq',freq)
        # print('aB',aB_freq.freq)
        # print('AB',AB_freq.freq)

        # Selection
        f = s(t) # mean selection force
        newFreq = np.array([freq[0]*(1-f),aB_freq.select(f),freq[2]*(1+f),AB_freq.select(-f)])

        # print('selection',selection)
        summ = np.sum(newFreq)
        freq = newFreq / summ
        # print('freq1',newFreq)
        # print('aB1',aB_freq.freq)
        # print('AB1',AB_freq.freq)

        aB_freq.normalize(summ)
        AB_freq.normalize(summ)
        # print('freq2',freq)
        # print('aB2',aB_freq.freq)
        # print('AB2',AB_freq.freq)



        # Recombination
        if (freq[0]+freq[2])*(freq[0]+freq[1])*(freq[1]+freq[3])*(freq[2]+freq[3]) > 0:
            D = (freq[0] * freq[3] - freq[1] * freq[2]) * recomb_rate
            recomb = np.array([-D,D,D,-D])
            freq += recomb
            if aB_freq.sum() == 0:
                aB_freq.freq += D * AB_freq.frac()
            elif AB_freq.sum() == 0:
                AB_freq.freq -= D * aB_freq.frac()
            else:
                (aB_freq.freq,aB_freq.freq) = (aB_freq.freq + D * AB_freq.frac(),\
                                                AB_freq.freq - D * aB_freq.frac())
        # print('freq3',freq)
        # print('aB3',aB_freq.freq)
        # print('AB3',AB_freq.freq)

        # Reproduction: Sample next generation from multinomial distribution
        # print('freq',freq)
        aB_freq.reproduce()
        AB_freq.reproduce()
        # print('freq4',freq)
        # print('aB4',aB_freq.freq)
        # print('AB4',AB_freq.freq)


        freq_expand = [freq[0],aB_freq.freq[0],aB_freq.freq[1],freq[2],AB_freq.freq[0],AB_freq.freq[1]]
        # print(freq_expand)
        next_N = np.random.multinomial(N,pvals=freq_expand/np.sum(freq_expand))
        next_freq = next_N / np.sum(next_N)
        freq = [next_freq[0],next_freq[1]+next_freq[2],next_freq[3],next_freq[4]+next_freq[5]]
        aB_freq.freq = next_freq[1:3]
        AB_freq.freq = next_freq[4:]

        # Calculate heterozygosity
        ah = (freq[0] + freq[1]) * (1 - (freq[0] + freq[1]))
        bh = (freq[0] + freq[2]) * (1 - (freq[0] + freq[2]))
        ah_cumu += ah
        bh_cumu += bh
        AH[t] = ah
        BH[t] = bh
        freq_history.append(list(freq))

        # Summary
        # If with mutation at both sites one allele still dominates
        if mutated == 2 and ((freq[0] + freq[1] + freq[2] <= 1e-5) \
            or (freq[0] + freq[1] + freq[3] <= 1e-5)\
            or (freq[0] + freq[2] + freq[3] <= 1e-5)
            or (freq[1] + freq[2] + freq[3] <= 1e-5)):
                freq_history = np.array(freq_history)
                return (freq_history,AH,BH,ah_cumu,bh_cumu,t)
        if t >= N / 10 and ah_cumu >= 100:
            freq_history = np.array(freq_history)
                # print(t)
            return (freq_history,AH,BH,ah_cumu,bh_cumu,t)

    freq_history = np.array(freq_history)
    # print(t)
    return (freq_history,AH,BH,ah_cumu,bh_cumu,t)



def main():
    num_trial = 10000
    max_t = 0
    max_freq = None
    max_cumu = 0
    ave_AH = np.zeros(num_steps)
    ave_BH = np.zeros(num_steps)
    ave_ah_cumu = 0
    ave_bh_cumu = 0
    for i in range(num_trial):
        print(i)
        (freq_history,AH,BH,ah_cumu,bh_cumu,t) = simulation()
        print('exit',t)
        ave_ah_cumu += ah_cumu
        ave_bh_cumu += bh_cumu
        ave_AH += AH
        ave_BH += BH

        if t >= max_t and max_cumu < ah_cumu:
            max_t = t
            max_freq = freq_history
            max_cumu = ah_cumu

    print('max time', max_t)

    print('locus A cumulative heterozygosity', ave_ah_cumu / num_trial)
    print('locus B cumulative heterozygosity', ave_bh_cumu / num_trial)

    output_path = 'results/l%d_p%d_s%d_var%d/' % (period,int(p*10),int(s_mid*10),int(var*10))
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ave_AH = ave_AH / num_trial
    ave_BH = ave_BH / num_trial

    plt.figure()

    t = list(range(max_t+1))
    t = t[::100]
    plt.plot(t,max_freq[:,0][::100])
    plt.plot(t,max_freq[:,1][::100])
    plt.plot(t,max_freq[:,2][::100])
    plt.plot(t,max_freq[:,3][::100])
    plt.xlabel('t')
    plt.ylabel('frequency')
    plt.legend(['ab','aB','Ab','AB'])
    plt.title('Allele frequency (N=%d)' % N)
    plt.savefig(output_path+'allele_freq.png')




    plt.figure()

    plt.plot(t,(max_freq[:,0] + max_freq[:,1])[::100])
    plt.plot(t,(max_freq[:,2] + max_freq[:,3])[::100])
    print(max_freq[-1])
    plt.legend(['a','A'])
    plt.xlabel('t')
    plt.ylabel('frequency')
    plt.savefig(output_path+'a_freq.png')



    plt.figure()

    plt.plot(t,(max_freq[:,0] + max_freq[:,2])[::100])
    plt.plot(t,(max_freq[:,1] + max_freq[:,3])[::100])
    plt.legend(['b','B'])
    plt.xlabel('t')
    plt.ylabel('frequency')
    plt.savefig(output_path+'b_freq.png')




    t = list(range(max_t))
    t = t[::100]
    plt.figure()
    plt.plot(t,ave_AH[:max_t][::100])
    plt.title('Average locus A heterozygosity')
    plt.xlabel('t')
    plt.ylabel('h')
    plt.savefig(output_path+'a_hetero.png')


    plt.figure()
    plt.plot(t,ave_BH[:max_t][::100])
    plt.title('Average locus B heterozygosity')
    plt.xlabel('t')
    plt.ylabel('h')
    plt.savefig(output_path+'b_hetero.png')

    plt.show()



if __name__ == '__main__':
    main()
