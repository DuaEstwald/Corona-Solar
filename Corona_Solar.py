# AUTHOR: ELENA ARJONA GALVEZ


import numpy as np
import matplotlib.pyplot as plt

# ========================
# ==== READ THE DATA =====
# ========================

fil = "bifrost.dat" 
z, n_el, p, T, rho = np.loadtxt(fil,comments='#', skiprows=1, unpack=True)

# ========================
# ====== TASK 1A =========
# ========================

plt.close("all")
plt.ion()

figa, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(6,9),sharex=True)

ax1.semilogy(z, T, label='Temperature')

ax1.axvline(x=0.0,lw=0.5,ls='--',color='k')
ax1.axvline(x=z[T==min(T)],lw=0.5,ls='--',color='k')
ax1.axvline(x=z[T<20e3][-1], lw=0.5, ls='--',color='k')
ax1.axvline(x=z[T>1e5][0],lw=0.5,ls='--',color='k')

ax1.fill_between(z, T, where=z<0.0, color='green',alpha=0.2, transform=ax1.get_xaxis_transform())
ax1.fill_between(z, T, where=(z>0.0)&(z<z[T==min(T)]), color='red', alpha=0.2, transform=ax1.get_xaxis_transform())
ax1.fill_between(z, T, where=(z>z[T==min(T)])&(z<z[T<20e3][-1]),color='pink',alpha=0.2,transform=ax1.get_xaxis_transform())
ax1.fill_between(z, T, where=(z>z[T<20e3][-1])&(z<z[T>1e5][0]),color='blue',alpha=0.2,transform=ax1.get_xaxis_transform())
ax1.fill_between(z, T, where=(z>z[T>1e5][0]),color='orange',alpha=0.2,transform=ax1.get_xaxis_transform())

ax1.invert_xaxis()
ax1.set_ylabel(r'Temperature [K]')


ax2.semilogy(z, rho, label='Gas pressure')

ax2.axvline(x=0.0,lw=0.5,ls='--',color='k')
ax2.axvline(x=z[T==min(T)],lw=0.5,ls='--',color='k')
ax2.axvline(x=z[T<20e3][-1], lw=0.5, ls='--',color='k')
ax2.axvline(x=z[T>1e5][0],lw=0.5,ls='--',color='k')

ax2.fill_between(z, min(rho),max(rho), where=z<0.0, color='green',alpha=0.2)
ax2.fill_between(z, min(rho),max(rho), where=(z>0.0)&(z<z[T==min(T)]), color='red', alpha=0.2)
ax2.fill_between(z, min(rho),max(rho), where=(z>z[T==min(T)])&(z<z[T<20e3][-1]),color='pink',alpha=0.2)
ax2.fill_between(z, min(rho),max(rho), where=(z>z[T<20e3][-1])&(z<z[T>1e5][0]),color='blue',alpha=0.2)
ax2.fill_between(z, min(rho),max(rho), where=(z>z[T>1e5][0]),color='orange',alpha=0.2)

ax2.invert_xaxis()
ax2.set_ylabel(r'Pressure [Pa]')


ax3.semilogy(z, p, label='Mass density')

ax3.axvline(x=0.0,lw=0.5,ls='--',color='k')
ax3.axvline(x=z[T==min(T)],lw=0.5,ls='--',color='k')
ax3.axvline(x=z[T<20e3][-1], lw=0.5, ls='--',color='k')
ax3.axvline(x=z[T>1e5][0],lw=0.5,ls='--',color='k')

ax3.fill_between(z, min(p),max(p), where=z<0.0, color='green',alpha=0.2, transform=ax3.get_xaxis_transform())
ax3.fill_between(z, min(p),max(p), where=(z>0.0)&(z<z[T==min(T)]), color='red', alpha=0.2, transform=ax3.get_xaxis_transform())
ax3.fill_between(z, min(p),max(p), where=(z>z[T==min(T)])&(z<z[T<20e3][-1]),color='pink',alpha=0.2,transform=ax3.get_xaxis_transform())
ax3.fill_between(z, min(p),max(p), where=(z>z[T<20e3][-1])&(z<z[T>1e5][0]),color='blue',alpha=0.2,transform=ax3.get_xaxis_transform())
ax3.fill_between(z, min(p),max(p), where=(z>z[T>1e5][0]),color='orange',alpha=0.2,transform=ax3.get_xaxis_transform())

ax3.invert_xaxis()
ax3.set_ylabel(r'Density $[kg/m^3]$')
ax3.set_xlabel('Height [Mm]')


plt.tight_layout()
plt.savefig('properties.png')

# =================================
# ======== TASK 1B ================
# =================================

mH = 1.6735e-27 # kg
mHe = 6.6465e-27

# PARA CALCULAR LAS POBLACIONES REALIZAMOS UN SISTEMA DE ECUACIONES:


coef = np.array([[mH, mHe],[1., -10.]])
sol = np.array([rho[:], np.zeros(len(rho))])
#   rho = mH*nH + mHe*nHe
#   0    = nH - 10*nHe
n_H, n_He = np.linalg.inv(coef).dot(sol)


plt.figure(size=(9,6))
plt.semilogy(z, n_el, label=r'$n_{el}$')
plt.semilogy(z, n_H, label = r'$n_H$')
plt.semilogy(z, n_He, label = r'$n_{He}$')

plt.axvline(x=0.0,lw=0.5,ls='--',color='k')
plt.axvline(x=z[T==min(T)],lw=0.5,ls='--',color='k')
plt.axvline(x=z[T<20e3][-1], lw=0.5, ls='--',color='k')
plt.axvline(x=z[T>1e5][0],lw=0.5,ls='--',color='k')


plt.xlabel('Height [Mm]')
plt.ylabel(r'Number density $[m^{-3}]$')
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig('numberdensity.png')

# =====================================
# ========== TASK 1C ==================
# =====================================


a1 = 1e-3 # H [kg/mol]
a1 = a1*1./(mH*6.022e23)


a2 = 4e-3 # He [kg/mol]
a1 = a2*1./(mHe*6.022e23)


a_i = np.array([a1, a2])
n_i = np.array([n_H, n_He])


mu = np.sum(n_i.T*a_i, axis = 1)/(n_el + np.sum(n_i, axis = 0))


X = 0.703 ; Y = 0.279 ; Z = 0.018

mu_ion_sim = 1./(2.*X+(3./4.)*Y + 0.5*Z)
mu_neu_sim = 1./(X+(Y/4.)+15.5*Z)

mu_ion = 1./(2*n_H*mH/rho + 3*n_He*mHe/rho)
mu_neu = 1./(n_H*mH/rho + n_He*mHe/rho)


plt.figure(figsize=(9,6))
plt.plot(z,mu_neu, label='Neutral gas')
plt.plot(z,mu_ion,label='Full ionization gas')
plt.plot(z,mu)
plt.axvline(x=0.0,lw=0.5,ls='--',color='k')
plt.axvline(x=z[T==min(T)],lw=0.5,ls='--',color='k')
plt.axvline(x=z[T<20e3][-1], lw=0.5, ls='--',color='k')
plt.axvline(x=z[T>1e5][0],lw=0.5,ls='--',color='k')


plt.legend()
plt.xlabel('Height [Mm]')
plt.ylabel(r'Atomic weight per particle, $\mu$')
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig('mu.png')

# ======================================
# =========== TASK 1D ==================
# ======================================


neHion = n_H
neHHeion = n_H+n_He


plt.figure(figsize=(9,6))
plt.plot(z, neHion/n_H, label = 'H ionized')
plt.plot(z, neHHeion/n_H, label = 'H and He single ionized')
plt.plot(z, n_el/n_H)
plt.axvline(x=0.0,lw=0.5,ls='--',color='k')
plt.axvline(x=z[T==min(T)],lw=0.5,ls='--',color='k')
plt.axvline(x=z[T<20e3][-1], lw=0.5, ls='--',color='k')
plt.axvline(x=z[T>1e5][0],lw=0.5,ls='--',color='k')


plt.xlabel('Height [Mm]')
plt.ylabel(r'$n_{e^-}/n_H$')

plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.savefig('nenH.png')



# ==========================================================================
# ===================== SECOND PART ========================================
# ==========================================================================


# =======================================
# =========== TASK 2A ===================
# =======================================

import ChiantiPy.core as ch
import ChiantiPy.tools as chtools

Zatom = [6, 8, 7, 20, 14, 26] # C, O, N, Ca, Si, Fe

fig, axs = plt.subplots(2,3, sharex=True, sharey=True, figsize=(15,10))

i = [0, 0, 0, 1, 1, 1]
j = [0, 1, 2, 0, 1, 2]
for k in range(len(Zatom)):
    ionfrac = ch.ioneq(Zatom[k])
    ionfrac.load()
    tmask = (ionfrac.Temperature>1e4)&(ionfrac.Temperature<1e8)
    ion = chtools.constants.El[Zatom[k]-1].capitalize()
    for r in range(Zatom[k]):
        num = chtools.constants.Ionstage[r]
        axs[i[k], j[k]].loglog(ionfrac.Temperature[tmask],ionfrac.Ioneq[r][tmask],\
                label = num)
        axs[i[k], j[k]].axvline(x=1e5,color='k')
        axs[i[k], j[k]].axvline(x=10**(5.5),color='k')
        axs[i[k], j[k]].axvline(x=1e6,color='k')
        axs[i[k], j[k]].axvline(x=10**(6.5),color='k')
        axs[i[k], j[k]].set_title('Ionization Equilibrium for '+ ion)
        axs[i[k], j[k]].set_xlabel('Temperature [K]')
        axs[i[k], j[k]].set_ylabel('Ion Fraction')

plt.tight_layout()
lgd = plt.legend(bbox_to_anchor=(1.3, -0.05),loc='upper right', ncol = 13, borderaxespad=2.)


plt.savefig('iontemp', bbox_extra_artists=(lgd,), bbox_inches='tight')

# FALTA JUGAR CON EL PLOT PARA PONERLO EN CONDICIONES

# ionfrac.Temperature # array con la temperatura
# ionfrac.Ioneq # array con las fracciones de ionizacion

# chtools.constants.El # aqui vienen todos los elementos
# chtools.const.Ionstage # aqui los numeros romanos


# =========================================
# ========== TASK 2B ======================
# =========================================

temp = T[np.log10(T) > 4.5]
height = z[np.log10(T) > 4.5]

fig, axs = plt.subplots(2,3, sharex=True, sharey=True, figsize=(15,10))


i = [0, 0, 0, 1, 1, 1]
j = [0, 1, 2, 0, 1, 2]
for k in range(len(Zatom)):
    ionfrac = ch.ioneq(Zatom[k])
    ionfrac.load()
    ionfrac.calculate(temp)
    ion = chtools.constants.El[Zatom[k]-1].capitalize()
    for r in range(Zatom[k]):
        num = chtools.constants.Ionstage[r]
        axs[i[k], j[k]].plot(np.log10(height), ionfrac.Ioneq[r], label = num)
        axs[i[k], j[k]].set_title('Ionization Equilibrium for '+ ion)
        axs[i[k], j[k]].set_xlabel('log(Height) [Mm]')
        axs[i[k], j[k]].set_ylabel('Ion Fraction')

plt.tight_layout()
lgd = plt.legend(bbox_to_anchor=(1.3, -0.05),loc='upper right', ncol = 13, borderaxespad=2.)

    
plt.savefig('ionheight', bbox_extra_artists=(lgd,), bbox_inches='tight')  

# ========================================
# ========= TASK 2C ======================
# ========================================

# NADA DE PROGRAMACION



# =========================================
# ========== TASK 3A ======================
# =========================================


t3 = np.logspace(4.5, 7.5, 50)
ne3 = 1e16/t3 


ions = ['fe_9', 'fe_14', 'fe_8', 'o_3','n_3','n_4','o_5','mg_6','si_8','mg_8']


plt.figure()
for io in ions:
    sp = ch.ion(io, temperature=t3, eDensity=ne3, abundance='sun_coronal_1992_feldman_ext')
    sp.intensity(allLines=1)
    G = np.sum(sp.Intensity['intensity'],axis=1)
    plt.loglog(t3,G,label=io)
plt.xlabel('log T [K]')
plt.ylabel(r'summed G(T) [erg $cm^{3}s^{-1}st^{-1}$]')
plt.title('Feldman 1992')

lgd = plt.legend(bbox_to_anchor=(1.2, -0.05),loc='upper right', ncol = 13, borderaxespad=2.)


plt.savefig('feldman', bbox_extra_artists=(lgd,), bbox_inches='tight')


plt.figure()
for io in ions[:3]:
    ci = ch.ion(io, temperature=t3, eDensity=ne3)
    ci.intensity(allLines=1)
    Gch = np.sum(ci.Intensity['intensity'],axis=1)
    fl = ch.ion(io, temperature=t3, eDensity=ne3, abundance='sun_coronal_1992_feldman_ext')
    fl.intensity(allLines=1)
    Gfl = np.sum(fl.Intensity['intensity'],axis=1)
    plt.loglog(t3,Gch,label=io+' chianti')
    plt.loglog(t3,Gfl, label=io+' feldman')
plt.xlabel('log T [K]')
plt.ylabel(r'summed G(T) [erg $cm^{3}s^{-1}st^{-1}$]')

lgd = plt.legend(bbox_to_anchor=(1., -0.05),loc='upper right', ncol = 2, borderaxespad=2.1)


plt.savefig('chiantidefault', bbox_extra_artists=(lgd,), bbox_inches='tight')

# =========================================
# ========== TASK 3B ======================
# =========================================

dl = 4.7/2.
ti1 = np.where(np.log10(t3)<5.5)[0][-1]
ti2 = np.where(np.log10(t3)<6.0)[0][-1]
ti3 = np.where(np.log10(t3)<6.5)[0][-1]

tin = [ti1, ti2, ti3]

for io in ions[:3]:
    sp = ch.ion(io, temperature=t3, eDensity=ne3, abundance='sun_coronal_1992_feldman_ext')
    sp.intensity(allLines=1)
    G = np.sum(sp.Intensity['intensity'],axis=1)
    for t in tin:
        sp.intensityPlot(index=t, wvlRange=[171.0-dl,171.+dl],linLog='log')
        plt.savefig(io+str(t)+'.png')


