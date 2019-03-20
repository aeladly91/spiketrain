import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Plots a bar graph of the errors between
# matrices A and B
def bar_plot(A, B, y_ax_percent=False):
    A = np.array(A)
    B = np.array(B)

    errors = np.abs(A-B).flatten()
    bins = np.arange(0, errors.max() + 1.5) - 0.5

    ymax = max(np.bincount(errors)+1)*1.1
    ymax = (int(ymax/100)+1)*100
    yticks = np.arange(0,ymax,101)

    xticks = range(0,errors.max()+2)
    plt.xticks(xticks)

    if y_ax_percent:
        plt.title("Error Distance Percentages")
        plt.ylabel("Percent of Errors")
    else:
        plt.title("Number of Errors by Distance")
        plt.ylabel("Number of Errors")
    plt.xlabel("Error Distance")
    plt.grid(True, axis='y',zorder=0)
    plt.hist(errors,bins,rwidth=.9,zorder=3,density=y_ax_percent,color="cornflowerblue")
    plt.show()

# Example use:
A = np.random.randint(low=0, high=4, size=(1000,1000))
B = np.random.randint(low=0, high=4, size=(1000,1000))
bar_plot(A,B,y_ax_percent=True)