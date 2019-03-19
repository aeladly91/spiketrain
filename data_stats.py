import numpy as np
import sys

# dataset
a = np.load("spikeTimes_1980_sec.npy")
# beginning time step
start = 19600127

# this takes in a neuron and the dataset
# this returns an array of times where
# gaps[i] = spike time i - spike time (i-1)
def getGaps(a, neur,start):
    prev = start
    gaps = []
    for i in a:
        if i[0] == neur:
            gaps.append(i[1]-prev)
            prev = i[1]
    # because start to spike 1 is not significant
    return gaps[1:]

# progress printing
print()
print("0" + "%\n\n", end="\r")
sys.stdout.write("\033[F")
sys.stdout.write("\033[F")

gaps = []
for i in range(1000):
    # progress printing
    if (i+1)%50 == 0:
        print(str((i+1)//10) + "%\n\n", end="\r")
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[F")
    # gaps[i] is the array of gaps for neuron i
    toappend = getGaps(a,i,start)
    gaps.append(toappend)

# index i is the variance of the spikes for neuron i
variances = [np.around(np.var(i), decimals=2) for i in gaps]

# index i is the average spike gap for neuron i
avggaps = [np.around(np.mean(i), decimals=2) for i in gaps]

# neuron with the min and max average gap
print("Min avg gap:\n", min(avggaps))
print("Max avg gap:\n", max(avggaps))

# normalized gaps
gap_norm = []
for i in range(len(gaps)):
    # take gaps[i] -= mean of gaps[i]
    gap_norm.append([gaps[i][j]-avggaps[i] for j in range(1,len(gaps[i]))])

# index i of avg_err_p is abs value of gap_norm[i]/gap_mean[i] for neuron i
# intuitively, entry i is average percent deviation from pattern of neuron i
avg_err_p = [np.mean(np.abs(gap_norm[i]))/avggaps[i] for i in range(len(gap_norm))]

# this is average across neurons of average percent deviation of each neuron
print("avg % err\n", np.mean(avg_err_p))

# index i is average number of time steps off neuron i is from it's pattern
avg_err = [np.mean(np.abs(gap_norm[i])) for i in range(len(gap_norm))]
print("avg err\n", np.mean(avg_err))

# avg across neurons of standard deviation of the times for that neuron
print("avg std of times:\n", np.mean(np.sqrt(variances)))
# avg across neurons of average deviation time from pattern
print("std of avg diffs:\n", np.sqrt(np.var(avg_err)))

print()