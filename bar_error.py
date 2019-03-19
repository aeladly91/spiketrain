import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Plots a bar graph of the errors between
# matrices A and B
def bar_plot(A, B):
    A = np.array(A)
    B = np.array(B)
    errors = np.abs(A-B).flatten()
    bins = np.arange(0, errors.max() + 1.5) - 0.5
    xticks = range(0,errors.max()+2)
    ymax = int(max(np.bincount(errors)+1)*1.1)
    yticks = range(0,ymax)
    plt.grid(True, axis='y',zorder=0)
    plt.hist(errors,bins,rwidth=.9,zorder=3)
    plt.yticks(yticks)
    plt.xticks(xticks)
    plt.show()