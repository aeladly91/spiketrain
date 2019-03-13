import numpy as np
from matplotlib import pyplot as plt


#neuronal dynamics variables
a = 0.02
b = 0.2
c = -61
d = 2
spikethresh = 30



def v_step(dt, v_last, u_last, I):
	v =  dt * ((0.04*v_last*v_last) + 5*v_last + 140 - u_last + I) + v_last
	return v

def u_step(dt, v_last, u_last, I):
	u = dt * (a*(b*v_last - u_last)) + u_last
	return u

def neurstep(dt, valvec, I_0, oldspikevec, newspikevec, w, index, neurons, tau, v_e):
	#Euler method to solve coupled DEs
	g_i = sc_step(dt, valvec[2], w, index, oldspikevec, neurons, tau)
	valvec[2] = g_i
	#reset after spiking event
	if (valvec[0] >= spikethresh):
		valvec[0] = c
		valvec[1] = valvec[1] + d
		return False
	else:
		I_syn = g_i*(v_e - valvec[0])
		I = I_0 + I_syn
		valvec[0] = v_step(dt, valvec[0], valvec[1], I)
		valvec[1] = u_step(dt, valvec[0], valvec[1], I)
		if (valvec[0] >=spikethresh):
			valvec[0] = spikethresh
			newspikevec[index] = 1
			return True
	return False

#################################
# Step in synaptic conductance #
#################################

def sc_step(dt, g_last, w, index, spikevec, neurons, tau):
	coupling = w[index, :]
	coupling = coupling.reshape(1,neurons)
	PSP= np.dot(coupling, spikevec) # set one as row, other as column
	g = g_last + (dt * ((1/tau) * ((-g_last)+(1/neurons)*PSP)))
	return g


def single_neuron():
	#time and value vectors
	t = np.linspace(t0,tf,n)

	#initial conditions
	t0 = 0
	tf = 1000 #msec
	v0 = c #mV
	u0 = 0
	n =2000 

	#step size
	dt = (tf-t0)/(n-1)

	#time and value vectors
	v = np.zeros([n]) 
	u = np.zeros([n]) 
	v[0] = v0
	u[0] = u0

	# inject current here
	I = 10 #mV
	# if (spikevec==0 && w ==0 && index && 0  && sc_step==0 && g==0):
	# 	neuron_generator(n, 1, v, u ,v_step, u_step, I, 0)
	# 	plot_activity(t, v)

def plot_activity(t, v):
	plt.plot(t, v)
	plt.xlabel("Time (ms)")
	plt.ylabel("Membrane potential (mV)")
	plt.title("Single Neuron Model-- Constant Current") 
	plt.show()
	plt.close()