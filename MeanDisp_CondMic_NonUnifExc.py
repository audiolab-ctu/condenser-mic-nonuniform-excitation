import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps
from math import sqrt, pi
from alive_progress import alive_bar
import os
from functions_CondMic_NonUnifExc import plot_ximean_mn_Full_NONunifload_frqParallel

#Membrane properties
R = 0.018 #membrane radius [m]
hm = 25e-6 #membrane thickness [m]
rho_m = 1944 #membrane density [kg/m^3]
ms = hm*rho_m #membrane mass per unit area [kg/m^2]
fres = 1040 #membrane resonance frequency [Hz]
J0zeros = sps.jn_zeros(0, 1)
J0zeros = J0zeros[-1]
T = ms*(2*pi*R*fres/J0zeros)**2 #membrane tension [N/m]

#fluid parameters
rho0=1.18 #air density [kg/m^3] thermoviscous fluid
c0 = 345.9 #adiabatic sound speed [m/s]
mu=1.83e-5 #shear dynamic viscosity [Pa.s]
gamma=1.4 #ratio of specific heats [-]
Cp = 1010 #specific heat at constant pressure [J/(kg.K)]
lamh = 24.4e-3 #thermal conductivity [W/(m.K)]

#Other dimmensions
hg = 230e-6 #gap thickness [m]
hc = 7.6e-3 #back cavity thickness [m]
Rc = R #back cavity radius [m]

No_parts = 4 #number of electrode segments

#input parameters
p_inc = 1 #incident acoustic pressure [Pa]
freq1 = np.logspace(1, 1.99, 6) #frequency [Hz] 20-OK
freq2 = np.logspace(2, 2.99, 20) #frequency [Hz] 100-OK
freq3 = np.logspace(3, 4, 50) #frequency [Hz] 500-OK
freq = np.concatenate((freq1, freq2, freq3))

#ATTENTION: higher values of mode numbers increase calculation time
R_modes = [8, 24] #number of radial modes [membrane, gap], optimal for given dimensions: [8, 32]
Theta_modes = [8, 24] #number of angular modes [membrane, gap], optimal for given dimensions: [8, 32]

alpha = 30*pi/180 #incident angle [rad]

#Symmetrical perforation pattern
Nh = 4 #number of holes
Rh = 0.5e-3*np.ones(Nh) #hole radius [m]
Lh = 1.6e-3*np.ones(Nh) #hole length [m]
r_h = 0.0084853 #distance of the holes from the center of the electrode [m]
R_h_coord = r_h*np.ones(Nh)#np.array([R_hn, R_hn, R_hn, R_hn]) #distance of the holes from the center of the electrode [m]
theta_h_coord = np.linspace(0, 3*pi/2, Nh) + pi/4 #angles of the holes [rad]

#Asymmetrical perforation pattern
'''
Rh = 0.5e-3*np.array([1/2, 1, 1.5, 2]) #hole radius [m]
Nh = len(Rh) #number of holes
Lh = 1.6e-3*np.ones(Nh) #hole length [m]
x_h_coord = 6e-3*np.array([1/2, -1/2, -1.5, 2]) #x-coordinates of the holes [m]
y_h_coord = 6e-3*np.array([1/3, 2, -1/2, -1]) #y-coordinates of the holes [m]
R_h_coord = np.sqrt(x_h_coord**2 + y_h_coord**2) #distance of the holes from the center of the electrode [m]
theta_h_coord = np.atan2(y_h_coord, x_h_coord) #angles of the holes [rad]
'''

#Comsol result filename 
filenames = [f for f in os.listdir() if f.startswith('Ximean')]
filenameComs = ' '.join(filenames)
#filenameComs = '' # uncomment if no Comsol result should be displayed

#Formulation with cavity impedance (III E): Yes - True / No - False
ImpCav = False

if __name__ == "__main__":
    plot_ximean_mn_Full_NONunifload_frqParallel(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, No_parts, filenameComs, ImpCav, freq)

