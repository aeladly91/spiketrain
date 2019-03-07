import numpy as np
from matplotlib import pyplot as plt
import random
import neuron_module as nm
import time

start_time = time.time()

# inject current here
I_0 = 10 #mV

#neuronal dynamics variables
a = 0.02
b = 0.2
c = -61
d = 2
spikethresh = 30

#initial conditions
t0 = 0 
tf = 100 #msec
v0 = c #mV
u0 = 0
g0 = 0
n =200
v_e = -70 #reversal potential of presynaptic neuron (mV)
tau = 5 #characteristic time of PSP decay (msec)
tvec = np.linspace(t0,tf,n)
tvec = tvec.reshape(1,n)


#step size
dt = (tf-t0)/(n-1)

#coupling matrix
neurons = 1;
w = np.random.rand(neurons, neurons)
for i in range (0,neurons-1):
	w[i,i] = 0


#value mat: [number of neurons]x3 dim matrix, where col0 = v, col1 = u, 
#col2 = g for the ith neuron
valmat = np.zeros([neurons, 3])
phase = np.random.randint(1, 20, size =neurons)

#spike catalog
oldspikevec = np.zeros([neurons, 1])
newspikevec = np.zeros([neurons, 1])

spikecount = 0

catalog = np.array([])

#####################################################################
# Calculate spike times for n firing neurons #
#####################################################################

# plt.figure()
# plt.axis([0, n, -5, spikethresh+5])
# plt.xlabel("Time (ms)")
# plt.ylabel("Membrane potential (mV)")
# plt.title("Two Coupled Neurons") 
# plt.ion()

for t in range (1, n):
	for i in range (0, neurons-1):
		if t == phase[i]:
			valmat[i,0]=v0 
			valmat[i,1]=u0
			valmat[i,2]=g0
		valvec = valmat[i,:]
		spiked = nm.neurstep(dt, valvec, I_0, oldspikevec, newspikevec, w, i, neurons, tau, v_e)
		if spiked:
			spikevec = np.array ([i, (n-1)*dt])
			catalog = np.vstack([catalog, spikevec]) if catalog.size else spikevec
			spikecount += 1
		valmat[i,:] = valvec
	oldspikevec = newspikevec;
	newspikevec = np.zeros([neurons, 1])
	tvec = np.full([neurons,1], t)
	try:
		plt.plot(tvec, valmat[:,0])
		plt.draw()
		plt.pause(0.001)
	except KeyboardInterrupt:
		plt.close('all')
		sys.exit('0')
		


print("Spike count =  %2d" % spikecount)
print("--- %s seconds ---" % (time.time() - start_time))

# name = str(neurons) + 'neurspikes'
# np.save(name,catalog)


