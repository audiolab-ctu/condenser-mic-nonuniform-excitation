import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import scipy as sp
import scipy.special as sps
from scipy.integrate import nquad, quad
import mpmath as mp
import math
from alive_progress import alive_bar
from math import pi
from sys import exit
from datetime import datetime
import time
from joblib import Parallel, delayed
import warnings
import re
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import transforms
from typing import Sequence, Union, Iterable

Int_precision = 1e-6

#Kronecker delta
def KronDelta(x,y):
    if x == y:
        return 1
    else:
        return 0

#Integral of combinations of sin and cos from 0 to 2*Pi
def Int_SinCos(m,q,sigma,s):
    if sigma == s and m == q != 0:
        Int = pi
    elif sigma == s == 1 and m == q == 0:
        Int = 2*pi
    else:
        Int = 0
    return Int
    
#Membrane eigennumbers
def kz_mn(m, n, R):
    z_mn = sps.jn_zeros(m, n)
    k_mn = z_mn[n-1]/R
    return k_mn, z_mn[n-1]

#Membrane norm
def Norm_Psi(m, n, R):
    k_mn, z_mn = kz_mn(m, n, R)
    if m == 0:
        Norm = 1/(np.sqrt(pi)*R*sps.jn(1, k_mn*R))
    else:
        Norm = 1/(np.sqrt(pi/2)*R*sps.jn(m-1, k_mn*R))
    return Norm

#Membrane eigenfunctions
def Psi_smn(r, theta, s, m, n, R, kmn_pre, NormPsi_pre):
    #k_mn, z_mn = kz_mn(m, n, R)
    k_mn = kmn_pre[m, n]
    Norm = NormPsi_pre[m, n]#Norm_Psi(m, n, R)
    if s == 1:
        Psi_mn = Norm*sps.jn(m, k_mn*r)*np.cos(m*theta)
    elif s == 2:
        Psi_mn = Norm*sps.jn(m, k_mn*r)*np.sin(m*theta)
    return Psi_mn

#Membrane mean value of eigenfunction
def Int_Psi_smn(s, m, n, R, kmn_pre):
    #k_mn, z_mn = kz_mn(m, n, R)
    k_mn = kmn_pre[m, n]
    if s == 1 and m == 0:
        IntPsi_mn = 2*np.sqrt(pi)/(k_mn)
    else:
        IntPsi_mn = 0
    return IntPsi_mn

#Membrane mean values of eigenfunction over [No_parts] parts
def IntQuart_Psi_smn(R_modes, Theta_modes, R, No_parts, kmn_pre, NormPsi_pre):
    def Integrand(r, theta):
        return Psi_smn(r, theta, s, m, n, R, kmn_pre, NormPsi_pre)*r
    Int_Psi_Qparts_smn = np.zeros((2*R_modes[0]*Theta_modes[0],No_parts))
    Int_lims = np.linspace(0,2*pi,No_parts+1)
    count = 0
    opts = {'epsabs': Int_precision, 'epsrel': Int_precision}
    for s in [1, 2]:
        for m in range(0, Theta_modes[0]):
            for n in range(1, R_modes[0]+1):
                #integrals over [No_parts] quadrants
                for part in range(0,No_parts):
                    Int_Psi_Qparts_smn[count, part] = nquad(Integrand, [[0, R], [Int_lims[part], Int_lims[part+1]]], opts=[opts, opts])[0]
                count = count + 1
    return Int_Psi_Qparts_smn, Int_lims

#Gap eigennumbers
def kappaz_mn(m, n, R):
    if m == 0:
        if n == 1:
            z_mn = 0
            kappa_mn = 0
        else:
            z_mn1 = sps.jnp_zeros(m, n)
            z_mn = z_mn1[n-2]
            kappa_mn = z_mn/R
    else:
        z_mn1 = sps.jnp_zeros(m, n)
        z_mn = z_mn1[n-1]
        kappa_mn = z_mn/R
    return kappa_mn, z_mn
    
#Gap norm
def Norm_Phi(m, n, R):
    kappa_mn, z_mn = kappaz_mn(m, n, R)
    if z_mn == 0:
        Norm = np.sqrt(2)/(R*np.sqrt((1+KronDelta(m, 0))*pi))
    else:
        Norm = np.sqrt(2)/(R*np.sqrt((1+KronDelta(m, 0))*pi)*np.sqrt(1-(m/z_mn)**2)*sps.jn(m, z_mn))
    return Norm

#Gap eigenfunctions
def Phi_smn(r, theta, s, m, n, kappamn_pre, NormPhi_pre):
    kappa_mn = kappamn_pre[m, n]
    Norm = NormPhi_pre[m, n]#Norm_Phi(m, n, R)
    if s == 1:
        Phi_mn = Norm*sps.jn(m, kappa_mn*r)*np.cos(m*theta)
    elif s == 2:
        Phi_mn = Norm*sps.jn(m, kappa_mn*r)*np.sin(m*theta)
    return Phi_mn

#Green's function G(r,r_0;theta,theta_0)
def G(r, r_0, theta, theta_0, R_modes, Theta_modes, R, Chi, kappamn_pre, NormPhi_pre):
    G = 0
    for sig in [1,2]:
        for mu in range(0,Theta_modes[1]):
            for nu in range(1,R_modes[1]+1):
                kappa_munu = kappamn_pre[mu, nu]#kappaz_mn(mu, nu, R)[0]
                G = G + Phi_smn(r, theta, sig, mu, nu, kappamn_pre, NormPhi_pre)*Phi_smn(r_0, theta_0, sig, mu, nu, kappamn_pre, NormPhi_pre)/(kappa_munu**2 - Chi**2)
    return G

#Integral of both eigenfunctions product over membrane area
def Int_Psi_smn_Phi_sigmunu(s, m, n, sigma, mu, nu, R, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre):
    k_mn = kmn_pre[m, n]
    kappa_munu = kappamn_pre[mu, nu]
    int_sincos = Int_SinCos(m, mu, s, sigma)
    if int_sincos == 0:
        Int = 0
    else:
        Int = int_sincos*NormPsi_pre[m, n]*NormPhi_pre[mu, nu]*R*k_mn*sps.jv(m,kappa_munu*R)*sps.jv(m+1,k_mn*R)/(k_mn**2 - kappa_munu**2)
    return Int

#Auxiliary variable a_mn
def a_mn(m, n, R, Chi, kmn_pre, NormPsi_pre):
    k_mn = kmn_pre[m, n]
    N_mn = NormPsi_pre[m, n]
    a = N_mn*k_mn*(sps.jv(m-1,k_mn*R)-sps.jv(m+1,k_mn*R))/(Chi*(sps.jv(m-1,Chi*R)-sps.jv(m+1,Chi*R)))
    return a

#Auxiliary variable X_mns(r,theta)
def X_mns(r, theta, s, m, n, R, Chi, kmn_pre, NormPsi_pre):
    k_mn = kmn_pre[m, n]
    if s == 1:
        X1 = np.cos(m*theta)*sps.jv(m,Chi*r)*a_mn(m, n, R, Chi, kmn_pre, NormPsi_pre)
    else:
        X1 = np.sin(m*theta)*sps.jv(m,Chi*r)*a_mn(m, n, R, Chi, kmn_pre, NormPsi_pre)
    X = (Psi_smn(r, theta, s, m, n, R, kmn_pre, NormPsi_pre) - X1)/(k_mn**2 - Chi**2)
    return X

#Auxiliary variable Gamma_h(r,theta) above each hole
def Gamma_h(r, theta, r_h, theta_h, R_modes, Theta_modes, R, Chi, zeta_g, kappamn_pre, NormPhi_pre):
    Gamma = zeta_g*G(r, r_h, theta, theta_h, R_modes, Theta_modes, R, Chi, kappamn_pre, NormPhi_pre)
    return Gamma

#Integral of X_mns*Psi_qlz over the membrane area
def Int_X_mns_Psi_qlz(s, m, n, z, q, l, R, Chi, zeta_xi, kmn_pre, NormPsi_pre):
    k_mn = kmn_pre[m, n]
    k_ql = kmn_pre[q, l]
    N_mn = NormPsi_pre[m, n]
    N_ql = NormPsi_pre[q, l]
   
    int_sincos = Int_SinCos(m, q, s, z)
    if int_sincos == 0:
        IntIn = 0
    else:
        IntIn = int_sincos*N_ql*R*(k_ql*sps.jv(q,Chi*R)*sps.jv(q-1,k_ql*R))/(Chi**2 - k_ql**2)
 
    Int = zeta_xi*(- IntIn*a_mn(m, n, R, Chi, kmn_pre, NormPsi_pre))/(k_mn**2 - Chi**2)
    
    return Int

#Integral of Gamma_h*Psi_qlz over the membrane area
def Int_Gamma_h_Psi_qlz(r_h, theta_h, z, q, l, R_modes, Theta_modes, R, Chi, zeta_g, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre):
    Int = 0
    for sig in [1,2]:
        for mu in range(0,Theta_modes[1]):
            for nu in range(1,R_modes[1]+1):
                kappa_munu = kappamn_pre[mu,nu]
                Int = Int + Phi_smn(r_h, theta_h, sig, mu, nu, kappamn_pre, NormPhi_pre)*Int_Psi_smn_Phi_sigmunu(z, q, l, sig, mu, nu, R, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre)/(kappa_munu**2 - Chi**2)
    return Int*zeta_g

#Membrane diagonal matrix HH with T*(k_mn**2-K**2)- zeta_xi/( k_mn**2 - Chi**2) and Nh ones
def HH_matrixFull_xi_p(R_modes, Theta_modes, R_h_coord, T, K, k_mn_vect, zeta_xi, Chi):
    Nh = len(R_h_coord) #number of holes
    UU = np.eye(2*R_modes[0]*Theta_modes[0]+Nh, dtype=complex) #eye matrix - preallocation
    count_row = 0
    count_col = 0
    for s in [1,2]:
        for ii in range(0, R_modes[0]*Theta_modes[0]):
            for ss in [1,2]:
                for jj in range(0, R_modes[0]*Theta_modes[0]):
                    k_mn = k_mn_vect[jj]
                
                    if count_row == count_col:
                        UU[count_row, count_col] = T*(k_mn**2-K**2) - zeta_xi/( k_mn**2 - Chi**2)


                    count_col = count_col + 1
            count_col = 0    
            count_row = count_row + 1
    return UU

#Vector BB of <p_f(r,theta)*Psi_smn(r,theta)> right side
def E_vect_an_xi_p(R_modes, Theta_modes, R, R_h_coord, p_inc, c0, alpha, kmn_pre, NormPsi_pre, freq):
    omega = 2*pi*freq #angular frequency [rad/s]
    k0 = omega/c0 #acoustic wave number [1/m]
    Nh = len(R_h_coord) #number of holes
    E = np.zeros((2*R_modes[0]*Theta_modes[0]+Nh,1),dtype=complex) #right side - preallocation
    smn_vect = np.zeros((2*R_modes[0]*Theta_modes[0],3)) #smn vector - preallocation
    k_mn_vect = np.zeros((2*R_modes[0]*Theta_modes[0],1)) #k_mn vector - preallocation
    MeanPsi = np.zeros((2*R_modes[0]*Theta_modes[0])) #MeanPsi vector - preallocation
    count_row = 0
    
    for s in [1,2]:
        for m in range(0,Theta_modes[0]):
            for n in range(1,R_modes[0]+1):
                smn_vect[count_row] = [s, m, n]
                k_mn_vect[count_row] = kmn_pre[m, n]
                MeanPsi[count_row] = Int_Psi_smn(s, m, n, R, kmn_pre)/(pi*R**2)
                IntR_m_1 = k_mn_vect[count_row]*R*sps.jn(m, -k0*R)*sps.jn(m-1, k_mn_vect[count_row]*R)/(k0**2 - k_mn_vect[count_row]**2)
                
                if s == 1:
                    IntTh_sm = 2*pi*np.cos(m*alpha)
                elif s == 2:
                    IntTh_sm = 2*pi*np.sin(m*alpha)

                E[count_row] = -p_inc*NormPsi_pre[m, n]*((1j)**m)*IntR_m_1*IntTh_sm
                count_row = count_row + 1

    return E, MeanPsi, k_mn_vect, smn_vect

#Admittance of the hole
def func_Admitance_hole(R, Rh, Lh, Nh, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, freq):
    omega = 2 * np.pi * freq  # angular frequency [rad/s]
    Sm = np.pi * R**2  # membrane/gap surface [m^2]
    Sh = np.pi * Rh**2  # hole cross-section [m^2]
    
    # Acoustic impedance of the hole
    lvp=mu/(rho0*c0); #viscous characteristic length [m]
    k0 = omega/c0; #adiabatic sound wave number [1/m]
    kv=np.sqrt(k0/lvp)*(1-1j)/np.sqrt(2) #complex wavenumb.
    Fvh= 1-2*sps.jn(1,kv*Rh)/(kv*Rh*sps.jn(0,kv*Rh))
    Zha = 1j*omega*rho0*Lh/(Fvh*Sh)
    
    #Hole edges correction - not used here
    X0 = np.sqrt(Sm/(Nh*pi)) #gap surface around one hole
    alph = Rh/X0
    Z_edgeGap = 1j*omega*rho0*(0.26164-0.353*alph+0.0809*alph**3)/X0 #added mass
    Z_mech_edge_Cav = (rho0*c0*pi*Rh**2)*(1-sps.jv(1,2*k0*Rh)/(k0*Rh) + 1j*sps.struve(1,2*k0*Rh)/(k0*Rh))
    Z_edge_Cav = Z_mech_edge_Cav/Sh**2
    
    Ytota = 1/(Zha)  # total acoustic admittance
    #Ytota = 1/(Zha + Z_edgeGap + Z_edge_Cav)  # total mechanical admittance with hole edges correction
    Ytots = Ytota / Sm  # total specific admittance
    Ytotm = Ytots / Sm  # total mechanical admittance
    
    return Ytotm, Ytots, Ytota

#Acoustic impedance of the back cavity
def func_Impedance_Cavity(Vc, Sc, c0, rho0, gamma, Cp, lamh, freq):
    omega = 2 * np.pi * freq  # angular frequency [rad/s]
    lh = lamh / (rho0 * c0 * Cp)  # thermal characteristic length [m]
    Vcplx = Vc * (1 + (1 - 1j) * (gamma - 1) * Sc * np.sqrt(c0 * lh / omega) / (Vc * np.sqrt(2)))  # equivalent complex volume
     
    Cva = Vc / (rho0 * c0**2) # Acoustic compliance of back cavity
    #Cva = Vcplx / (rho0 * c0**2) # Acoustic compliance of back cavity taking into accout thermal boundary layers effects
    ZVca = 1 / (1j * omega * Cva)  # acoustic impedance of back cavity
    return ZVca

#Matrix CC - influence of the air-filled system behind the membrane
def CC_matrix_Full_xi_p(R_modes, Theta_modes, R, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, Chi, Fv, Chic, Fvc, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre, ImpCav, freq):
    omega = 2*pi*freq
    Nh = len(R_h_coord) #number of holes
    CC = np.zeros((2*R_modes[0]*Theta_modes[0]+Nh,2*R_modes[0]*Theta_modes[0]+Nh),dtype=complex) #preallocation
    Yh = np.zeros((Nh,1),dtype=complex)
    for h in range(0,Nh):
        Yhm, Yhs, Yha = func_Admitance_hole(R, Rh[h], Lh[h], Nh, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, freq)
        Yh[h] = Yha
    Vc = np.pi * Rc**2 * hc  # cavity volume [m^3]
    Sc = 2 * np.pi * Rc * (Rc + hc)  # cavity inner surface [m^2]
    Zca = func_Impedance_Cavity(Vc, Sc, c0, rho0, gamma, Cp, lamh, freq)
    zeta_xi = omega**2*rho0/(Fv*hg)
    zeta_g = 1j*omega*rho0/(Fv*hg)
    zeta_c = 1j*omega*rho0/(Fvc*hc)
    count_row = 0
    count_col = 0
    for z in [1,2]:
        for q in range(0,Theta_modes[0]):
            for l in range(1, R_modes[0]+1):

                for s in [1,2]:
                    for m in range(0,Theta_modes[0]):
                        for n in range(1, R_modes[0]+1):
                            CC[count_row,count_col] = Int_X_mns_Psi_qlz(s, m, n, z, q, l, R, Chi, zeta_xi, kmn_pre, NormPsi_pre)
                            count_col = count_col + 1
                count_col = 0    
                count_row = count_row + 1

    count_row_mid = count_row
    count_col_mid = count_row
    count_col = count_col_mid
    count_row = 0

    for z in [1,2]:
        for q in range(0,Theta_modes[0]):
            for l in range(1, R_modes[0]+1):
                for hc in range(0, Nh):
                    CC[count_row,count_col] = Yh[hc]*Int_Gamma_h_Psi_qlz(R_h_coord[hc], theta_h_coord[hc], z, q, l, R_modes, Theta_modes, R, Chi, zeta_g, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre)
                    count_col = count_col + 1
                count_col = count_col_mid
                count_row = count_row + 1

    count_row = count_row_mid
    count_col = 0

    for hr in range(0, Nh):
        for s in [1,2]:
            for m in range(0,Theta_modes[0]):
                for n in range(1, R_modes[0]+1):
                    CC[count_row,count_col] = -zeta_xi*X_mns(R_h_coord[hr], theta_h_coord[hr], s, m, n, R, Chi, kmn_pre, NormPsi_pre)
                    count_col = count_col + 1
        count_col = 0
        count_row = count_row + 1

    count_row = count_row_mid
    count_col = count_row_mid

    for hr in range(0, Nh):
        for hc in range(0, Nh):
            if ImpCav:
                CC[count_row,count_col] = -Yh[hc]*(Gamma_h(R_h_coord[hc], theta_h_coord[hc], R_h_coord[hr], theta_h_coord[hr], R_modes, Theta_modes, R, Chi, zeta_g, kappamn_pre, NormPhi_pre) 
                                               + Zca)
            else:
                CC[count_row,count_col] = -Yh[hc]*(Gamma_h(R_h_coord[hc], theta_h_coord[hc], R_h_coord[hr], theta_h_coord[hr], R_modes, Theta_modes, R, Chi, zeta_g, kappamn_pre, NormPhi_pre) 
                                               + Gamma_h(R_h_coord[hc], theta_h_coord[hc], R_h_coord[hr], theta_h_coord[hr], R_modes, Theta_modes, R, Chic, zeta_c, kappamn_pre, NormPhi_pre))
            count_col = count_col + 1
        count_col = count_row_mid
        count_row = count_row + 1

    return CC

#Mean disp. and modal coeffs of the memb. in the AIR, LEM model, NONunif load, frequency dependent
def xi_mean_mn_Full(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre, ImpCav, freq):
    
    omega = 2*pi*freq #angular frequency [rad/s]
    K = omega*np.sqrt(ms/T) #membrane wave number [1/m]
    k0 = omega/c0; #adiabatic sound wave number [1/m]

    Nh = len(R_h_coord) #number of holes
    #characteristic lengths
    lh=lamh/(rho0*c0*Cp) #thermal characteristic length [m]
    lvp=mu/(rho0*c0); #viscous characteristic length [m]

    #complex wavenumb.
    kv=np.sqrt(k0/lvp)*(1-1j)/np.sqrt(2)
    kh=np.sqrt(k0/lh)*(1-1j)/np.sqrt(2)

    #Air gap
    Fv = 1-(np.tan(kv*hg/2)/(kv*hg/2))
    Fh = 1-(np.tan(kh*hg/2)/(kh*hg/2))

    Chi=np.sqrt(k0**2*(gamma-(gamma-1)*Fh)/Fv) # complex wavenumber of the air gap
    zeta_xi = omega**2*rho0/(Fv*hg)

    #Cavity
    Fvc = 1-(np.tan(kv*hc/2)/(kv*hc/2))
    Fhc = 1-(np.tan(kh*hc/2)/(kh*hc/2))
    Chic=np.sqrt(k0**2*(gamma-(gamma-1)*Fhc)/Fvc) # complex wavenumber of the cavity
    
    #Matrices and vectors
    xi_mn = np.zeros((2*R_modes[0]*Theta_modes[0],1),dtype=complex) #modal coefficients - preallocation

    E, MeanPsi, k_mn_vect, smn_vect = E_vect_an_xi_p(R_modes, Theta_modes, R, R_h_coord, p_inc, c0, alpha, kmn_pre, NormPsi_pre, freq)

    HH = HH_matrixFull_xi_p(R_modes, Theta_modes, R_h_coord, T, K, k_mn_vect, zeta_xi, Chi)
  
    CC = CC_matrix_Full_xi_p(R_modes, Theta_modes, R, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, Chi, Fv, Chic, Fvc, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre, ImpCav, freq)
    
    #Matrix equation solution
    xi_mn_p = np.linalg.solve((HH - CC), E)
    xi_mn = np.copy(xi_mn_p[0:2*R_modes[0]*Theta_modes[0]])
    dp_holes = np.copy(xi_mn_p[-Nh:])
  
    #Mean displacement
    xi_mean = MeanPsi @ xi_mn

    return xi_mean.squeeze(), xi_mn.squeeze(), smn_vect, dp_holes.squeeze()

#Mean displacement over [No_parts] parts
def Qmean_disp(R_modes, Theta_modes, R, xi_mn, No_parts, kmn_pre, NormPsi_pre):
    IntQuart_Psi, Int_lims = IntQuart_Psi_smn(R_modes, Theta_modes, R, No_parts, kmn_pre, NormPsi_pre)
    MeanQuart_Psi = IntQuart_Psi/(pi*R**2/No_parts)
    Parts_means = xi_mn.T @ MeanQuart_Psi
    return Parts_means, Int_lims

#Frequency loop, returns mean disp., means over [No_parts] parts, and xi_mn for all freqs
def ximean_Qmean_mn_Full_NONunifload_frq(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, No_parts, ImpCav, freq):
    #Preallocation
    xi_mean = np.zeros(len(freq),dtype=complex)
    xi_mn = np.zeros((2*R_modes[0]*Theta_modes[0],len(freq)),dtype=complex)
    Nh = len(R_h_coord) #number of holes
    dp_holes = np.zeros((Nh,len(freq)),dtype=complex)
    Parts_means = np.zeros((No_parts,len(freq)),dtype=complex)
    kmn_pre = np.zeros([Theta_modes[0],R_modes[0]+1])
    kappamn_pre = np.zeros([Theta_modes[1],R_modes[1]+1])
    NormPsi_pre = np.zeros([Theta_modes[0],R_modes[0]+1])
    NormPhi_pre = np.zeros([Theta_modes[1],R_modes[1]+1])

    for m in range(0,Theta_modes[0]):
        for n in range(1,R_modes[0]+1):
            kmn_pre[m,n], z_mn = kz_mn(m, n, R)
            NormPsi_pre[m,n] = Norm_Psi(m, n, R)
            
    for m in range(0,Theta_modes[1]):
        for n in range(1,R_modes[1]+1):
            kappamn_pre[m,n], z_mn = kappaz_mn(m, n, R)
            NormPhi_pre[m,n] = Norm_Phi(m, n, R)

    #frequency loop
    with alive_bar(len(freq)) as bar:   
        for ind_freq in range(len(freq)):
            bar()
            xi_mean[ind_freq], xi_mn[:,ind_freq], smn_vect, dp_holes[:,ind_freq] = xi_mean_mn_Full(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre, ImpCav, freq[ind_freq])
    Parts_means, Int_lims = Qmean_disp(R_modes, Theta_modes, R, xi_mn, No_parts, kmn_pre, NormPsi_pre)
    return xi_mean, Parts_means.T, xi_mn, smn_vect, Int_lims, dp_holes

#Parallelized frequency loop
def ximean_Qmean_mn_Full_NONunifload_frqParallel(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, No_parts, ImpCav, frequencies):
    #Preallocation
    xi_mean = np.zeros(len(frequencies),dtype=complex)
    xi_mn = np.zeros((2*R_modes[0]*Theta_modes[0],len(frequencies)),dtype=complex)
    Nh = len(R_h_coord) #number of holes
    dp_holes = np.zeros((Nh,len(frequencies)),dtype=complex)
    Parts_means = np.zeros((No_parts,len(frequencies)),dtype=complex)
    kmn_pre = np.zeros([Theta_modes[0],R_modes[0]+1])
    kappamn_pre = np.zeros([Theta_modes[1],R_modes[1]+1])
    NormPsi_pre = np.zeros([Theta_modes[0],R_modes[0]+1])
    NormPhi_pre = np.zeros([Theta_modes[1],R_modes[1]+1])

    for m in range(0,Theta_modes[0]):
        for n in range(1,R_modes[0]+1):
            kmn_pre[m,n], z_mn = kz_mn(m, n, R)
            NormPsi_pre[m,n] = Norm_Psi(m, n, R)
            
    for m in range(0,Theta_modes[1]):
        for n in range(1,R_modes[1]+1):
            kappamn_pre[m,n], z_mn = kappaz_mn(m, n, R)
            NormPhi_pre[m,n] = Norm_Phi(m, n, R)

    #frequency loop
    with alive_bar(len(frequencies)) as bar: 
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit each frequency computation as a separate task
            future_to_freq = {executor.submit(xi_mean_mn_Full, R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre, ImpCav, freq): ind_freq for ind_freq, freq in enumerate(frequencies)}
            
            for future in concurrent.futures.as_completed(future_to_freq):
                ind_freq = future_to_freq[future]
                bar()
                try:
                    # Unpack the multiple results from the function
                    xi_mean[ind_freq], xi_mn[:,ind_freq], smn_vect, dp_holes[:,ind_freq] = future.result()  # Store all results
                except Exception as exc:
                    print(f"Frequency {frequencies[ind_freq]} generated an exception: {exc}")
  
    Parts_means, Int_lims = Qmean_disp(R_modes, Theta_modes, R, xi_mn, No_parts, kmn_pre, NormPsi_pre)
    return xi_mean, Parts_means.T, xi_mn, smn_vect, Int_lims, dp_holes

def ximean_Qmean_mn_Full_NONunifload_angleParallel(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha_vec, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, No_parts, ImpCav, freq):
    #Mean dispacement
    xi_mean = np.zeros(len(alpha_vec),dtype=complex)
    xi_mn = np.zeros((2*R_modes[0]*Theta_modes[0],len(alpha_vec)),dtype=complex)
    Nh = len(R_h_coord) #number of holes
    dp_holes = np.zeros((Nh,len(alpha_vec)),dtype=complex)
    Parts_means = np.zeros((No_parts,len(alpha_vec)),dtype=complex)

    kmn_pre = np.zeros([Theta_modes[0],R_modes[0]+1])
    kappamn_pre = np.zeros([Theta_modes[1],R_modes[1]+1])
    NormPsi_pre = np.zeros([Theta_modes[0],R_modes[0]+1])
    NormPhi_pre = np.zeros([Theta_modes[1],R_modes[1]+1])
    for m in range(0,Theta_modes[0]):
        for n in range(1,R_modes[0]+1):
            kmn_pre[m,n], z_mn = kz_mn(m, n, R)
            NormPsi_pre[m,n] = Norm_Psi(m, n, R)
            
    
    for m in range(0,Theta_modes[1]):
        for n in range(1,R_modes[1]+1):
            kappamn_pre[m,n], z_mn = kappaz_mn(m, n, R)
            NormPhi_pre[m,n] = Norm_Phi(m, n, R)

    #frequency loop
    with alive_bar(len(alpha_vec)) as bar: 
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit each frequency computation as a separate task
            future_to_alpha = {executor.submit(xi_mean_mn_Full, R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre, ImpCav, freq): ind_alpha for ind_alpha, alpha in enumerate(alpha_vec)}
            
            for future in concurrent.futures.as_completed(future_to_alpha):
                ind_alpha = future_to_alpha[future]
                bar()
                try:
                    # Unpack the multiple results from the function
                    xi_mean[ind_alpha], xi_mn[:,ind_alpha], smn_vect, dp_holes[:,ind_alpha] = future.result()  # Store all results
                except Exception as exc:
                    print(f"Frequency {alpha_vec[ind_alpha]} generated an exception: {exc}")
   
    Parts_means, Int_lims = Qmean_disp(R_modes, Theta_modes, R, xi_mn, No_parts, kmn_pre, NormPsi_pre)
    return xi_mean, Parts_means.T, xi_mn, smn_vect, Int_lims, dp_holes

#Displacement space dependent
def disp_space(R_modes, Theta_modes, Disp_r_points, Disp_th_points, R, xi_mn, kmn_pre, NormPsi_pre):
    r_vect = np.linspace(0, R, Disp_r_points)
    theta_vect = np.linspace(0, 2*pi, Disp_th_points)
    Rr, Theta = np.meshgrid(r_vect, theta_vect)

    Psi_matrix = np.zeros((len(theta_vect), len(r_vect)))
    count = 0
    for s in [1,2]:
        for m in range(0,Theta_modes[0]):
            for n in range(1,R_modes[0]+1):
                Psi_matrix = Psi_matrix + xi_mn[count]*Psi_smn(Rr, Theta, s, m, n, R, kmn_pre, NormPsi_pre)
                count = count + 1

    # Express the mesh in the cartesian system.
    X, Y = Rr*np.cos(Theta), Rr*np.sin(Theta)

    return Psi_matrix, X, Y, Rr, Theta

#Auxiliary functions for parallelized pressure field
def compute_one_term(index, s, m, n, xi_mn, Rr, Theta, R_modes, Theta_modes, R, Chi, zeta_xi, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre):
    xmn_val = zeta_xi*X_mns(Rr, Theta, s, m, n, R, Chi, kmn_pre, NormPsi_pre)
    return xi_mn[index] * xmn_val

def compute_sum2_term(h, Yh, Rr, Theta, R_h_coord, theta_h_coord, R_modes, Theta_modes, R, Chi, zeta_g, kappamn_pre, NormPhi_pre, dp_holes):
    gamma_val = Gamma_h(Rr, Theta, R_h_coord[h], theta_h_coord[h], R_modes, Theta_modes, R, Chi, zeta_g, kappamn_pre, NormPhi_pre)
    return Yh[h] * gamma_val * dp_holes[h]

def compute_cav_term(h, Yh, Rr, Theta, R_h_coord, theta_h_coord, R_modes, Theta_modes, R, Chic, zeta_c, kappamn_pre, NormPhi_pre, dp_holes):
    gamma_mat = Gamma_h(Rr, Theta, R_h_coord[h], theta_h_coord[h], R_modes, Theta_modes, R, Chic, zeta_c, kappamn_pre, NormPhi_pre)
    return -Yh[h] * gamma_mat * dp_holes[h]

def compute_cav_term(h, Yh, Rr, Theta, R_h_coord, theta_h_coord, R_modes, Theta_modes, R, Chic, zeta_c, kappamn_pre, NormPhi_pre, dp_holes):
    gamma_mat = Gamma_h(Rr, Theta, R_h_coord[h], theta_h_coord[h], R_modes, Theta_modes, R, Chic, zeta_c, kappamn_pre, NormPhi_pre)
    return -Yh[h] * gamma_mat * dp_holes[h]

#Pressure space dependent
def press_space(R_modes, Theta_modes, Press_r_points, Press_th_points, R, hg, hc, Rc, Rh, Lh, R_h_coord, theta_h_coord, xi_mn, dp_holes, kmn_pre, NormPsi_pre, kappamn_pre, NormPhi_pre, c0, rho0, mu, Cp, lamh, gamma, ImpCav, freq):
    r_vect = np.linspace(0, R, Press_r_points)
    theta_vect = np.linspace(0, 2*pi, Press_th_points)
    Rr, Theta = np.meshgrid(r_vect, theta_vect)

    PressGap_matrix = np.zeros((len(theta_vect), len(r_vect)))
    PressCav_matrix = np.ones((len(theta_vect), len(r_vect)))
    Sum1 = np.zeros((len(theta_vect), len(r_vect)))
    Sum2 = np.zeros((len(theta_vect), len(r_vect)))

    #frequency dependent variables
    omega = 2*pi*freq #angular frequency [rad/s]
    k0 = omega/c0; #adiabatic sound wave number [1/m]

    Nh = len(R_h_coord) #number of holes
    #characteristic lengths
    lh=lamh/(rho0*c0*Cp) #thermal characteristic length [m]
    lvp=mu/(rho0*c0); #viscous characteristic length [m]

    #complex wavenumb.
    kv=np.sqrt(k0/lvp)*(1-1j)/np.sqrt(2)
    kh=np.sqrt(k0/lh)*(1-1j)/np.sqrt(2)

    #Gap
    Fv = 1-(np.tan(kv*hg/2)/(kv*hg/2))
    Fh = 1-(np.tan(kh*hg/2)/(kh*hg/2))

    Chi=np.sqrt(k0**2*(gamma-(gamma-1)*Fh)/Fv) # complex wavenumber of the air gap

    #Cavity
    Fvc = 1-(np.tan(kv*hc/2)/(kv*hc/2))
    Fhc = 1-(np.tan(kh*hc/2)/(kh*hc/2))
    Chic=np.sqrt(k0**2*(gamma-(gamma-1)*Fhc)/Fvc) # complex wavenumber of the cavity

    Yh = np.zeros((Nh,1),dtype=complex)
    for h in range(0,Nh):
        Yhm, Yhs, Yha = func_Admitance_hole(R, Rh[h], Lh[h], Nh, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, freq)
        Yh[h] = Yha
    zeta_xi = omega**2*rho0/(Fv*hg)
    zeta_g = 1j*omega*rho0/(Fv*hg)
    zeta_c = 1j*omega*rho0/(Fvc*hc)

    count = 0
    
    start1 = time.time() 

    tasks = []
    count = 0
    for s in [1, 2]:
        for m in range(Theta_modes[0]):
            for n in range(1, R_modes[0] + 1):
                tasks.append((count, s, m, n))
                count += 1

    # Run in parallel
    results = Parallel(n_jobs=-1)(
        delayed(compute_one_term)(index, s, m, n, xi_mn, Rr, Theta, R_modes, Theta_modes, R, Chi, zeta_xi, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre)
        for index, s, m, n in tasks
    )            
    Sum1 = sum(results)

    print('smn time: ',time.time() - start1)

    start1 = time.time() 

    results_gap = Parallel(n_jobs=-1)(
    delayed(compute_sum2_term)(h, Yh, Rr, Theta, R_h_coord, theta_h_coord, R_modes, Theta_modes, R, Chi, zeta_g, kappamn_pre, NormPhi_pre, dp_holes)
    for h in range(Nh)
    )
    Sum2 = sum(results_gap)

    print('h gap time: ',time.time() - start1)
    PressGap_matrix = Sum1 + Sum2

    start1 = time.time() 

    Vc = np.pi * Rc**2 * hc  # cavity volume [m^3]
    Sc = 2 * np.pi * Rc * (Rc + hc)  # cavity inner surface [m^2]
    Zca = func_Impedance_Cavity(Vc, Sc, c0, rho0, gamma, Cp, lamh, freq)

    if ImpCav:
        sum_cav = 0
        for h in range(Nh):
            sum_cav += Yh[h] * dp_holes[h]
        PressCav_matrix = -Zca*sum_cav*PressCav_matrix
    else:
        results_cav = Parallel(n_jobs=-1)(
        delayed(compute_cav_term)(h, Yh, Rr, Theta, R_h_coord, theta_h_coord, R_modes, Theta_modes, R, Chic, zeta_c, kappamn_pre, NormPhi_pre, dp_holes)
        for h in range(Nh)
        )
        PressCav_matrix = sum(results_cav)
    print('h cav time: ',time.time() - start1)

    # Express the mesh in the cartesian system.
    X, Y = Rr*np.cos(Theta), Rr*np.sin(Theta)

    return PressGap_matrix, PressCav_matrix, X, Y, Rr, Theta

#Plot displacement
def plot_disp(xi_matrix, X, Y):
    
    fig = plt.figure()
    fig.suptitle('Membrane displacement')
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.view_init(elev=90, azim=-90, roll=0)
    sc = ax.plot_surface(X, Y, xi_matrix.real, cmap='jet', rstride=1, cstride=1, linewidth=0, antialiased=False) #cmap='plasma'
    cbar = fig.colorbar(sc, orientation='horizontal', label='[m]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    #ax.set_zlabel(r"$\Re[\xi(x,y)]$")
    ax.set_title(r"Re$[\xi(x,y)]$")

    ax = fig.add_subplot(1,2,2,projection='3d')
    ax.view_init(elev=90, azim=-90, roll=0)
    sc = ax.plot_surface(X, Y, xi_matrix.imag, cmap='jet', rstride=1, cstride=1, linewidth=0, antialiased=False, ) #cmap='plasma'
    cbar = fig.colorbar(sc, orientation='horizontal', label='[m]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    #ax.set_zlabel(r"$\Im[\xi(x,y)]$")
    ax.set_title(r"Im$[\xi(x,y)]$")

    #plt.show(block=False)

#Plot pressure
def plot_press(P_matrix, X, Y, ReTitleStr, ImTitleStr, SupTitle):
    
    fig = plt.figure()
    fig.suptitle(SupTitle)
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.view_init(elev=90, azim=-90, roll=0)
    sc = ax.plot_surface(X, Y, P_matrix.real, cmap='jet', rstride=1, cstride=1, linewidth=0, antialiased=False) #cmap='plasma'
    cbar = fig.colorbar(sc, orientation='horizontal', label='[Pa]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    #ax.set_zlabel(r"$\Re[p(x,y)]$")
    ax.set_title(ReTitleStr)

    ax = fig.add_subplot(1,2,2,projection='3d')
    ax.view_init(elev=90, azim=-90, roll=0)
    sc = ax.plot_surface(X, Y, P_matrix.imag, cmap='jet', rstride=1, cstride=1, linewidth=0, antialiased=False, ) #cmap='plasma'
    cbar = fig.colorbar(sc, orientation='horizontal', label='[Pa]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    #ax.set_zlabel(r"$\Im[p(x,y)]$")
    ax.set_title(ImTitleStr)

    #plt.show(block=False)

#Plot displacement with given function
def plot_disp_Full_NONunifload_frq(R_modes, Theta_modes, Disp_r_points, Disp_th_points, Press_r_points, Press_th_points, R, ms, T, p_inc,c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, ImpCav, freq):
    start = time.time()
    kmn_pre = np.zeros([Theta_modes[0],R_modes[0]+1])
    kappamn_pre = np.zeros([Theta_modes[1],R_modes[1]+1])
    NormPsi_pre = np.zeros([Theta_modes[0],R_modes[0]+1])
    NormPhi_pre = np.zeros([Theta_modes[1],R_modes[1]+1])
    for m in range(0,Theta_modes[0]):
        for n in range(1,R_modes[0]+1):
            kmn_pre[m,n], z_mn = kz_mn(m, n, R)
            NormPsi_pre[m,n] = Norm_Psi(m, n, R)
            
    for m in range(0,Theta_modes[1]):
        for n in range(1,R_modes[1]+1):
            kappamn_pre[m,n], z_mn = kappaz_mn(m, n, R)
            NormPhi_pre[m,n] = Norm_Phi(m, n, R)

    xi_mean, xi_mn, smn_vect, dp_holes = xi_mean_mn_Full(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, kmn_pre, kappamn_pre, NormPsi_pre, NormPhi_pre, ImpCav, freq)
    
    Disp_matrix, X_xi, Y_xi, Rr, Theta = disp_space(R_modes, Theta_modes, Disp_r_points, Disp_th_points, R, xi_mn, kmn_pre, NormPsi_pre)
    plot_disp(Disp_matrix, X_xi, Y_xi)

    elapsed_time = time.time() - start

    # Convert the elapsed time to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Format the time into a string for filename
    time_str = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
    print('Displacement time: '+time_str)

    start = time.time()
    PressGap_matrix, PressCav_matrix, X_p, Y_p, Rr, Theta = press_space(R_modes, Theta_modes, Press_r_points, Press_th_points, R, hg, hc, Rc, Rh, Lh, R_h_coord, theta_h_coord, xi_mn, dp_holes, kmn_pre, NormPsi_pre, kappamn_pre, NormPhi_pre, c0, rho0, mu, Cp, lamh, gamma, ImpCav, freq)
    elapsed_time = time.time() - start

   # Convert the elapsed time to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Format the time into a string for filename
    time_str = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
    print('Press time: '+time_str) 

    plot_press(PressGap_matrix, X_p, Y_p, r"Re$[p_g(x,y)]$",r"Im$[p_g(x,y)]$", 'Air gap pressure')

    plot_press(PressCav_matrix, X_p, Y_p, r"Re$[p_c(x,y)]$", r"Im$[p_c(x,y)]$", 'Cavity pressure')

    plt.show(block=False)
    plt.pause(0.001) # Pause for interval seconds.
    input("hit[enter] to end.")
    plt.close('all') # all open plots are correctly closed after each run

#Plot Mean displacement Part means and xi_mnvs frequency
def plot_ximean_mn(xi_mean, Parts_means, xi_mn, dp_holes, smn_vect, Int_lims, No_parts, filenameComs, freq):

    freq_num, xi_num, xi_mean_num = ComsolImport(filenameComs)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    plt.loglog(freq_num, abs(xi_mean_num), 'k.', label = 'Reference FEM')
    plt.loglog(freq, abs(xi_mean), 'r', label = 'Present model')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Abs($\xi_{mean}$) [m]')
    plt.title('Total mean displacement')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best')
    ax = fig.add_subplot(2,1,2)
    plt.semilogx(freq_num, np.angle(xi_mean_num), 'k.')
    plt.semilogx(freq, np.unwrap(np.angle(xi_mean)), 'r')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Phase($\xi_{mean}$) [rad]')

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    for i in range(Parts_means.shape[0]):
        plt.loglog(freq, abs(Parts_means[i,:]), label = str(round(Int_lims[i],2)) + " - " + str(round(Int_lims[i+1],2)))
    plt.loglog(freq_num, abs(xi_num), '.', label = 'Comsol')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Abs($\xi_{mean}$) [m]')
    plt.title('Mean displacements over ' + str(No_parts) + ' parts')
    plt.legend(title="Mean limits [rad]")
    ax = fig.add_subplot(2,1,2)
    for i in range(Parts_means.shape[0]):
        plt.semilogx(freq, np.unwrap(np.angle(Parts_means[i,:])))
    plt.semilogx(freq_num, np.angle(xi_num), '.') #np.unwrap(np.angle(xi_num), period=2*np.pi), '.')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Phase($\xi_{mean}$) [rad]')

    #Differences of mean displacements between opposite segments
    diff_1_3 = Parts_means[0, :] - Parts_means[2, :] # \xi_1 - \xi_3 Analytical 
    diff_2_4 = Parts_means[1, :] - Parts_means[3, :]  # \xi_2 - \xi_4 Analytical 

    if np.any(np.isnan(xi_num)):
        xi_num_1_3 = (np.nan + 1j*np.nan)*np.ones(5)
        xi_num_2_4 = (np.nan + 1j*np.nan)*np.ones(5)
        freq_num = np.ones(5)
    else:
        xi_num_1_3 =xi_num[:, 0] - xi_num[:, 2] # \xi_1 - \xi_3 Numerical
        xi_num_2_4 =xi_num[:, 1] - xi_num[:, 3] # \xi_2 - \xi_4 Numerical
    
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    plt.loglog(freq_num, abs(xi_num_1_3) , 'k.', label =r'$ \bar{\xi}_{1} -\bar{\xi}_{3} $ Reference FEM model')
    plt.loglog(freq_num, abs(xi_num_2_4), '.', color='0.6', label=r'$ \bar{\xi}_{2} - \bar{\xi}_{4} $ Reference FEM model')
    plt.loglog(freq, abs(diff_1_3),'r', label=r'$ \bar{\xi}_{1} - \bar{\xi}_{3} $ Present analytical model')
    plt.loglog(freq, abs(diff_2_4),'b--', label=r'$ \bar{\xi}_{2} - \bar{\xi}_{4} $ Present analytical model')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Abs($\Delta\xi_{mean}$) [m]')
    plt.title('Differences of mean displacements between opposite segments')    
    ax = fig.add_subplot(2,1,2)
    ticks = np.arange(-3*pi, 1.1*pi, pi/2)  # from -2π to 2π, step π/2
    labels_phase = [r"$-3\pi$", r"$-5\pi/2$", r"$-2\pi$", r"$-3\pi/2$", r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels_phase)
    plt.semilogx(freq_num, np.unwrap(np.angle(xi_num_1_3)), 'k.') #, period=2*np.pi
    plt.semilogx(freq_num, np.unwrap(np.angle(xi_num_2_4), discont=3*pi/2, period=2*pi), '.', color='0.6')
    plt.semilogx(freq, np.unwrap(np.angle(diff_1_3)), 'r')
    plt.semilogx(freq, np.unwrap(np.angle(diff_2_4), discont=3*pi/2, period=2*pi), 'b--')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Phase($\Delta\xi_{mean}$) [rad]')
    handles, labels = ax1.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], labelspacing = 0.05)
    plt.ylim(bottom=-3.5*pi)


    plt.ion()
    fig = plt.figure()
    for i in range(xi_mn.shape[0]):
        plt.loglog(freq, abs(xi_mn[i,:]), label = str(smn_vect[i,:]))
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Abs value of Modal coefficients [m]')
    plt.title('Modal coefficients')
    plt.legend(title="modes: [s,m,n]")

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    for i in range(dp_holes.shape[0]):
        plt.semilogx(freq, np.abs(dp_holes[i,:]))
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Abs($\Delta p_h$) [Pa]')
    plt.title('Acoustic pressure difference between both sides of the holes')
    ax = fig.add_subplot(2,1,2)
    for i in range(dp_holes.shape[0]):
        plt.semilogx(freq, np.unwrap(np.angle(dp_holes[i,:])))
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Phase($\Delta p_h$) [rad]')

    plt.show(block=False)
    plt.pause(0.001) # Pause for interval seconds.
    input("hit[enter] to end.")
    plt.close('all') # all open plots are correctly closed after each run

#Polar plot
def plot_ximean_angle_mn(xi_mean, Parts_means, xi_mn, dp_holes, smn_vect, Int_lims, No_parts, filenameComs, freq, alpha, alpha_num,dB_num_1_3,dB_num_2_4):  
    dBpolarLowLim = -40

    No_parts = np.shape(Parts_means)[0]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, polar=True)

    Diff_ximean = np.zeros((No_parts, Parts_means.shape[1]), dtype=complex)
    dB_Diff_ximean = np.zeros((No_parts, Parts_means.shape[1]))
    dB_Diff_ximean_Norm = np.zeros((No_parts, Parts_means.shape[1]))

    if dB_num_1_3 is not np.nan and dB_num_2_4 is not np.nan:
        dB_num_1_3=dB_num_1_3-np.max(dB_num_1_3)
        dB_num_2_4=dB_num_2_4-np.max(dB_num_2_4)
    
        index_1_3 = [9, 10, 11, 12, 14,15, 16, 17,61,62,63,64,66,67,68,69]
        new_dB_num_1_3 = np.delete(dB_num_1_3, index_1_3)
    
        index_2_4 = [35,36,37,38,40,41,42,43,87,88,89,90,92,93,94,95]
        new_dB_num_2_4 = np.delete(dB_num_2_4, index_2_4)

        new_alpha_num13=np.delete(alpha_num, index_1_3)
        new_alpha_num24=np.delete(alpha_num, index_2_4)
    else:
        new_dB_num_1_3=dB_num_1_3
        new_dB_num_2_4=dB_num_2_4
        new_alpha_num13=alpha_num
        new_alpha_num24=alpha_num

    for i in range(int(No_parts/2)):
        Diff_ximean[i,:] = Parts_means[i,:]-Parts_means[i+int(No_parts/2),:]
        dB_Diff_ximean[i,:] = 20*np.log10(abs(Diff_ximean[i,:]))
        dB_Diff_ximean_Norm[i,:] = dB_Diff_ximean[i,:] - np.max(dB_Diff_ximean[i,:])


        if i == 0:
            ax1.plot(alpha, dB_Diff_ximean_Norm[i, :], color='r',
                 label=r'$ \bar{\xi}_{1} - \bar{\xi}_{3}$', linewidth=2.5)
        else:
            ax1.plot(alpha, dB_Diff_ximean_Norm[i, :],'--', color='b', label=r'$ \bar{\xi}_{2} - \bar{\xi}_{4}$',linewidth=2.5) 
        
    marker_style = dict(markersize=8.5, markeredgewidth=1.5)  
    ax1.plot(new_alpha_num13,new_dB_num_1_3,'.', color='k', label =r'$ \bar{\xi}_{1_{num}} -\bar{\xi}_{3_{num}}$', **marker_style)
    ax1.plot(new_alpha_num24,new_dB_num_2_4,'.',color='darkgrey',label =r'$ \bar{\xi}_{2_{num}} -\bar{\xi}_{4_{num}}$', **marker_style) 
    ax1.set_title(str(freq)+"Hz", fontsize=25, fontweight='bold')
    fig1.legend(loc='outside lower center', ncols=4)

    # Hide default radial labels
    ax1.set_yticklabels([])
    ax1.xaxis.grid(True, linestyle='-', linewidth=0.8) 
    ax1.tick_params(axis='x', labelsize=22, pad=15)

    # Radial tick values
    r_ticks = [-30, -20, -10, 0]
    ax1.set_yticks(r_ticks) 
    # Base angle (radians)
    theta = np.radians(30)  
    # Hide default radial labels if you want manual labels
    ax1.set_yticklabels([])

    # Turn off the default polar grid
    ax1.yaxis.grid(False)

    for r in r_ticks:
        ax1.plot(np.linspace(0, 2*np.pi, 360), np.ones(360)*r, 
             color='gray', linestyle='-', linewidth=0.8)

    # How far from center to place each label (radial distance)
    r_offset = 0.1  # increase to move labels farther outward

    for r in r_ticks:
        ax1.text(
            theta,           # angle along circle
         r + r_offset,    # shift outward
         f"{r}",
         ha='left',
         va='center',
         fontsize=20
     )

    plt.tight_layout()
    ax1.set_ylim(bottom=dBpolarLowLim)

    plt.show(block=False)
    plt.pause(0.001) # Pause for interval seconds.
    input("hit[enter] to end.")
    plt.close('all') # all open plots are correctly closed after each run


#Plot Mean displacement NONunif. load in AIR, LEM model,vs frequency
def plot_ximean_mn_Full_NONunifload_frq(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, No_parts, filenameComs, ImpCav, freq, save):
    xi_mean, Parts_means, xi_mn, smn_vect, Int_lims, dp_holes = ximean_Qmean_mn_Full_NONunifload_frq(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, No_parts, ImpCav, freq)
    
    plot_ximean_mn(xi_mean, Parts_means, xi_mn, dp_holes, smn_vect, Int_lims, No_parts, filenameComs, freq)

#Plot Mean displacement NONunif. load in AIR, LEM model,vs frequency - PARALLEL version  
def plot_ximean_mn_Full_NONunifload_frqParallel(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, No_parts, filenameComs, ImpCav, freq):
    start = time.time()
    xi_mean, Parts_means, xi_mn, smn_vect, Int_lims, dp_holes = ximean_Qmean_mn_Full_NONunifload_frqParallel(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, No_parts, ImpCav, freq)
    elapsed_time = time.time() - start
    # Convert the elapsed time to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Format the time into a string for filename
    time_str = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
    print('Total time: '+time_str)
    print('Time per frequency point: ',elapsed_time/len(freq), 's')

    plot_ximean_mn(xi_mean, Parts_means, xi_mn, dp_holes, smn_vect, Int_lims, No_parts, filenameComs, freq)

#Polar plot call
def plot_ximean_mn_Full_NONunifload_angleParallel(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, No_parts, filenameComs, ImpCav, freq):
    start = time.time()
    xi_mean, Parts_means, xi_mn, smn_vect, Int_lims, dp_holes = ximean_Qmean_mn_Full_NONunifload_angleParallel(R_modes, Theta_modes, R, ms, T, p_inc, c0, alpha, Rh, Lh, R_h_coord, theta_h_coord, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, No_parts, ImpCav, freq)
    elapsed_time = time.time() - start
    # Convert the elapsed time to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Format the time into a string for filename
    time_str = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
    print('Total time: '+time_str)
    print('Time per angle point: ',elapsed_time/len(alpha), 's')

    alpha_num, dB_num_1_3, dB_num_2_4 = read_angle_data(filenameComs)

    plot_ximean_angle_mn(xi_mean, Parts_means, xi_mn, dp_holes, smn_vect, Int_lims, No_parts, filenameComs, freq, alpha, alpha_num,dB_num_1_3,dB_num_2_4)

# Function to read the data from a file
def read_data(filename):
    try:
        data = np.loadtxt(filename, skiprows=8)  # Skip 9 header lines
        freq = data[:, 0]  # First column is the frequency
        real_parts = data[:, 1:5]  # Next 4 columns are the real parts
        imag_parts = data[:, 5:9]  # Next 4 columns are the imaginary parts
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        freq = np.nan
        real_parts = np.nan
        imag_parts = np.nan
    return freq, real_parts, imag_parts

def read_angle_data(filename):
    try:
        data = np.loadtxt(filename, skiprows=8)  # Skip 9 header lines
        alpha_num = data[:, 0]  # First column is the frequency
        dB_num_1_3 = data[:, 1]  # Next 4 columns are the real parts
        dB_num_2_4  = data[:, 2]  # Next 4 columns are the imaginary parts
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        alpha_num = np.nan
        dB_num_1_3  = np.nan
        dB_num_2_4  = np.nan
    return alpha_num, dB_num_1_3 , dB_num_2_4

#Import of Comsol data
def ComsolImport(filenameComs):

    filenames = filenameComs.split()
    
    # Lists to accumulate all data
    all_freq = []
    all_real = []
    all_imag = []
    freq_num = np.nan
    
    # Read and store data from each file
    for filename in filenames:
        freq_num, real, imag = read_data(filename)
        all_freq.extend(freq_num)
        all_real.extend(real)
        all_imag.extend(imag)
    
    # Convert to NumPy arrays for sorting
    all_freq = np.array(all_freq)
    all_real = np.array(all_real)
    all_imag = np.array(all_imag)
    
    # Sort by freq_num
    sorted_indices = np.argsort(all_freq)
    sorted_freq = all_freq[sorted_indices]
    sorted_real = all_real[sorted_indices]
    sorted_imag = all_imag[sorted_indices]

    # if no data
    if np.any(np.isnan(freq_num)):
        xi_num = np.nan*np.ones(5)
        xi_mean_num = np.nan*np.ones(5)
        sorted_freq = np.nan*np.ones(5)
    else:
        xi_num = sorted_real + 1j * sorted_imag
        xi_mean_num = np.mean(xi_num, axis=1)

    return sorted_freq, xi_num, xi_mean_num