import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Parameters 
N = 10000
num_allele = 4
num_steps = N * 100
mutation_rate = 1e-5
recomb_rate = 0.5
period = 20
smax = 0.25
p = 1    # Effect of second locus

# cyclic selection using sine function 
def selection(t):
    pt = t % period / period
    s = smax * math.sin(pt*2*math.pi)
    return np.array([1-s,1-s*(1-p),1+s,1+s*(1-p)])



# modified cyclic selection using sine function 
# def selection(t):
#     pt = t % period / period * 2
#     s = smax * math.sin(pt*2*math.pi)
#     if pt <= 1:
#         return np.array([1-s,1-s*(1-p),1+s,1+s*(1-p)])
#     else:
#         return np.array([1-s*(1-p),1-s,1+s*(1-p),1+s])


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


    freq = np.array([1.0,0.0,0.0,0.0])  # ab, aB, Ab, AB

    freq_history = []


    # Introduce a mutation in either locus
    i = random.random()
    if i < 0.5:   # Introduce at A
        freq[2] = 1/N
        freq[0] -= freq[2]
        mutated = 0
    else:
        freq[1] = 1/N
        freq[0] -= freq[1]
        mutated = 1
    # print(freq)

    for t in range(num_steps):

        # Mutation
        # If mutation happens at A
        if mutated == 0:
            i = random.random()
            if i < mutation_rate * N:
                freq[1] += 1/N
                freq[0] -= 1/N
                mutated = 2
        elif mutated == 1:
            i = random.random()
            if i < mutation_rate * N:
                freq[2] += 1/N
                freq[0] -= 1/N
                mutated = 2

        # Selection

        sele = selection(t)
        # print('selection',selection)
        freq = np.multiply(freq,sele)
        freq = freq / np.sum(freq)

        # print(freq)

        # Recombination
        if (freq[0]+freq[2])*(freq[0]+freq[1])*(freq[1]+freq[3])*(freq[2]+freq[3]) > 0:
            D = (freq[0] * freq[3] - freq[1] * freq[2]) * recomb_rate
            recomb = np.array([-D,D,D,-D])
            freq += recomb

        # Reproduction: Sample next generation from multinomial distribution
        # print('freq',freq)
        new_P = np.random.multinomial(N,pvals=freq)
        freq = new_P / N

        # Calculate heterozygosity
        ah = (freq[0] + freq[1]) * (1 - (freq[0] + freq[1]))
        bh = (freq[0] + freq[2]) * (1 - (freq[0] + freq[2]))
        ah_cumu += ah
        bh_cumu += bh
        AH[t] = ah
        BH[t] = bh
        freq_history.append(list(freq))

        # Summary
        # If with mutation at both sites the frequency of any allele still goes to 0
        if mutated == 2 and ((freq[0] + freq[1] <= 1e-10) or (freq[0] + freq[2] <= 1e-10)\
            or (freq[1] + freq[3] <= 1e-10) or (freq[2] + freq[3] <= 1e-10)):
            freq_history = np.array(freq_history)
            return (freq_history,AH,BH,ah_cumu,bh_cumu,t)

    freq_history = np.array(freq_history)
    return (freq_history,AH,BH,ah_cumu,bh_cumu,t)



def main():
    num_trial = N * 10
    max_t = 0
    max_freq = None
    ave_AH = np.zeros(num_steps)
    ave_BH = np.zeros(num_steps)
    ave_ah_cumu = 0
    ave_bh_cumu = 0
    for i in range(num_trial):
        print(i)
        (freq_history,AH,BH,ah_cumu,bh_cumu,t) = simulation()
        ave_ah_cumu += ah_cumu
        ave_bh_cumu += bh_cumu
        ave_AH += AH
        ave_BH += BH

        if t > max_t:
            max_t = t
            max_freq = freq_history

    print('average locus A heterozygosity: ', ave_ah_cumu / num_trial)
    print('average locus B heterozygosity: ', ave_bh_cumu / num_trial)

    ave_AH = ave_AH / num_trial
    ave_BH = ave_BH / num_trial

    plt.figure()

    t = list(range(max_t))
    t = t[::100]
    plt.plot(t,max_freq[:,0][::100])
    plt.plot(t,max_freq[:,1][::100])
    plt.plot(t,max_freq[:,2][::100])
    plt.plot(t,max_freq[:,3][::100])
    plt.xlabel('t')
    plt.ylabel('frequency')
    plt.legend(['ab','aB','Ab','AB'])
    plt.title('Allele frequency (N=%d)' % N)


    plt.figure()

    plt.plot(t,(max_freq[:,0] + max_freq[:,1])[::100])
    plt.plot(t,(max_freq[:,2] + max_freq[:,3])[::100])
    print(max_freq[-1])
    plt.legend(['a','A'])
    plt.xlabel('t')
    plt.ylabel('frequency')



    t = list(range(num_steps))
    t = t[::100]
    plt.figure()
    plt.plot(t,ave_AH[::100])
    plt.title('Average heterozygosity (h) at locus A')
    plt.xlabel('t')
    plt.ylabel('h')

    plt.figure()
    plt.plot(t,ave_BH[::100])
    plt.title('Average heterozygosity (h) at locus B')
    plt.xlabel('t')
    plt.ylabel('h')
    

    plt.show()



if __name__ == '__main__':
    main()
