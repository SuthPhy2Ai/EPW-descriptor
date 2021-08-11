import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.special import assoc_laguerre

def get_wave_r1(n,l,m,Zm,r):
        x = r
        y = 0 
        z = 0
        X, Z = np.meshgrid(x, z)

        rho = np.linalg.norm((X,y,Z), axis=0) *Zm / n
        Lag = assoc_laguerre(2 * rho, n - l - 1, 2 * l + 1)
        Ylm  = sph_harm(m, l, np.arctan2(y,X), np.arctan2(np.linalg.norm((X,y), axis=0), Z))
        Psi = np.exp(-rho) * np.power((2*rho),l) * Lag * Ylm

        density = np.conjugate(Psi) * Psi
        density = density.real
        return density[0]
    
def get_wave_r2(n,l,m,r,Zm):
        x = 0
        y = 0
        z = r
        X, Z = np.meshgrid(x, z)

        rho = np.linalg.norm((X,y,Z), axis=0)*Zm  / n
        Lag = assoc_laguerre(2 * rho, n - l - 1, 2 * l + 1)
        Ylm  = sph_harm(m, l, np.arctan2(y,X), np.arctan2(np.linalg.norm((X,y), axis=0), Z))
        Psi = np.exp(-rho) * np.power((2*rho),l) * Lag * Ylm

        density = np.conjugate(Psi) * Psi
        density = density.real
        return density[0]
    
def get_wave_r3(n,l,m,r,Zm):
        x = 0
        y = r
        z = 0
        X, Z = np.meshgrid(x, z)

        rho = np.linalg.norm((X,y,Z), axis=0)*Zm  / n
        Lag = assoc_laguerre(2 * rho, n - l - 1, 2 * l + 1)
        Ylm  = sph_harm(m, l, np.arctan2(y,X), np.arctan2(np.linalg.norm((X,y), axis=0), Z))
        Psi = np.exp(-rho) * np.power((2*rho),l) * Lag * Ylm

        density = np.conjugate(Psi) * Psi
        density = density.real
        return density[0]
        
def get_wave_mean(n,l,m,r,Zm):
    n = n
    l = l
    m = m
    Zm = Zm
    wave1 =get_wave_r1(n,l,m,r,Zm)[0]
    wave2 =get_wave_r2(n,l,m,r,Zm)[0]
    wave3 =get_wave_r3(n,l,m,r,Zm)[0]
    mean_wave = (wave1+ wave2 + wave3)/3
    #return mean_wave+wave1+wave2+wave3
    return np.array([mean_wave, wave1, wave2, wave3])


if __name__ == '__main__':
