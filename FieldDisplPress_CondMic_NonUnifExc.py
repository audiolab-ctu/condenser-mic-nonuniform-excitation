import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps
from math import sqrt, pi
from functions_CondMic_NonUnifExc import plot_disp_Full_NONunifload_frq

#Membrane properties
R = 0.018 #membrane radius [m]
hm = 25e-6 #membrane thickness [m]
rho_m = 1944 #membrane density [kg/m^3]
ms = hm*rho_m #membrane mass per unit area [kg/m^2]
fres = 1040 #membrane resonance frequency [Hz]
J0zeros = sps.jn_zeros(0, 1)
J0zero1 = J0zeros[-1]
T = ms*(2*pi*R*fres/J0zero1)**2 #membrane tension [N/m]

#fluid parameters
rho0=1.18 #air density [kg/m^3] thermoviscous fluid
c0 = 345.9 #adiabatic sound speed [m/s]
mu = 1.83e-5 #shear dynamic viscosity [Pa.s]
gamma = 1.4 #ratio of specific heats [-]
Cp = 1010 #specific heat at constant pressure [J/(kg.K)]
lamh = 24.4e-3 #thermal conductivity [W/(m.K)]

#Other dimmensions
hg = 230e-6 #gap thickness [m]
hc = 7.6e-3 #back cavity thickness [m]
Rc = R #back cavity radius [m]

#input parameters
p_inc = 1 #incident acoustic pressure [Pa]
freq = 100 #frequency [Hz]

Disp_r_points = 100 #number of points in radial direction for dispacement plot
Disp_th_points = 180 #number of points in angular direction for dispacement plot

#ATTENTION: higher values of Press_r_points and Press_th_points increase calculation time
Press_r_points = 100 #number of points in radial direction for pressure plot
Press_th_points = 180 #number of points in angular direction for pressure plot

#ATTENTION: higher values of mode numbers increase calculation time
R_modes = [8, 24] #number of radial modes [membrane, gap], optimal for given dimensions: [8, 32], for higher frequencies [16, 32]
Theta_modes = [8, 24] #number of angular modes [membrane, gap], optimal for given dimensions: [8, 32], for higher frequencies [16, 32]

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

#Formulation with cavity impedance (III E): Yes - True / No - False
ImpCav = False

#dispacement + pressure field plot
plot_disp_Full_NONunifload_frq(R_modes, Theta_modes, Disp_r_points, Disp_th_points, Press_r_points, Press_th_points, R, ms, T, p_inc,c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, ImpCav, freq)

plt.show()