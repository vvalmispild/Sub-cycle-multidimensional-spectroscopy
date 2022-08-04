#!/usr/bin/env python 

################ Libraries ###############################
import numpy as np
from   fractions import Fraction
import matplotlib.pyplot as plt
import matplotlib.colors as collors
from   matplotlib.colors import LogNorm
import matplotlib.cm as cmx
import scipy.signal
import sys
import glob
import re
import math

import matplotlib as mpl
mpl.use('MacOSX')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pylab




from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange
from matplotlib.collections import PolyCollection
from matplotlib import ticker
plt.rcParams.update({'font.size': 13})
plt.rcParams['text.usetex'] = True
FourLn2=4.0*np.log(2.0)




h = 0.01 # timestep
h_phi = 0.16*6 #
delta_phi = 0.039269
w_pulse_hop = 1.92
t_hop = 0.43
w_pulse_eV = w_pulse_hop * t_hop

hop_to_fs = 0.6582119514




time_vp_1 = np.load("../data_1/data_0topi2/time_ar_2.npy")
#time_vp_2 = np.load("./data_pi2topi_2/time_pulse_2.npy")
#time_vp_ar = np.concatenate((time_vp_1, time_vp_2), axis=1)

vect_pot_1 = np.load("../data_1/data_0topi2/LHB_2.npy")
vect_pot_2 = np.load("../data_1/data_pi2topi/LHB_2.npy")


#vect_pot_1 = np.load("../data_all_E/data_E_01/data_0topi2/LHB_4.npy")
#vect_pot_2 = np.load("../data_all_E/data_E_01/data_pi2topi/LHB_4.npy")





#### data cleaning. get rid of trivial contributions and construct nonlinear spectr ##

#vect_pot_1 = np.load("../data_1/data_0topi2/vect_pot_2.npy")
#vect_pot_2 = np.load("../data_1/data_pi2topi/vect_pot_2.npy")

vect_pot_ar_0 = np.concatenate((vect_pot_1, vect_pot_2), axis=1)/(np.pi*0.43)

#vect_pot_ar_0 = vect_pot_ar_0[::2]
time_vp_ar_0 = time_vp_1#[::10]
print(vect_pot_ar_0.shape)
#### cut first 5 fs of the data ###
#vect_pot_ar_cut = np.delete(vect_pot_ar_0, np.s_[164:1071:1], axis=0)
#time_vp_ar_cut = np.delete(time_vp_ar_0, np.s_[164:1071:1],0)
#vect_pot_ar_cut = np.delete(vect_pot_ar_0, np.s_[0:120:1], axis=0)
#time_vp_ar_cut = np.delete(time_vp_ar_0, np.s_[0:120:1], axis=0)
#vect_pot_ar_cut = np.imag(vect_pot_ar_0)
time_vp_ar_cut = time_vp_ar_0
#vect_pot_ar_cut = vect_pot_ar_0[::2]
vect_pot_ar_cut = vect_pot_ar_0
l_A= len(vect_pot_ar_0[0,:])
####### phi=0 shift ########
phi1 = np.arange(0, 80, 1)
phi_0 = vect_pot_ar_cut[:,0]

phi_ar = np.array([])
for i in phi1:
    phi_i = np.roll(phi_0, -i)
    if i==0:
        phi_ar=phi_0
    if i>0:
        phi_ar = np.column_stack((phi_ar,phi_i))

########################


####### phi=0 shift ########
phi1 = np.arange(0, l_A, 1)
time_coef = 0.01*0.6582119514/0.43
phi_coef = 0.039269*0.6582119514/0.8256
phi_0 = vect_pot_ar_cut[:,0]
phi_pi2 = vect_pot_ar_cut[:,40]
     
########## add edges phi 0#####

phi_ar1 = np.roll(phi_0, 1)
phi_ar2 = np.roll(phi_0, 2)
#print(phi_ar1)
phi_ar = np.column_stack((phi_ar2, phi_ar1, phi_ar))


phi_ar = np.row_stack((phi_ar, phi_ar[0,:]))
phi_ar = np.row_stack((phi_ar[-2,:], phi_ar))


print("phi_ar =",  phi_ar.shape)
##########################


print(vect_pot_ar_cut.shape)
#phi_ar1 = np.column_stack((phi_ar1, phi_ar1[:,-1]))
vect_pot_ar_cut = np.row_stack((vect_pot_ar_cut, vect_pot_ar_cut[-2,:]))
vect_pot_ar_cut = np.row_stack((vect_pot_ar_cut[0,:], vect_pot_ar_cut))
vect_pot_ar_cut = np.column_stack((vect_pot_ar_cut, vect_pot_ar_cut[:,0]))
vect_pot_ar_cut = np.column_stack((vect_pot_ar_cut[:,-2], vect_pot_ar_cut))
#print(phi_ar.shape)
print(vect_pot_ar_cut.shape)

# steps 1 and 2
dx_phi = 0.039269/0.8256
dy_time = 0.02/0.43
dx_phi_0 = 0.0384/0.8256 #!!!!
g_1 = np.gradient(vect_pot_ar_cut, dx_phi, axis=1)
g_1_phi0 = np.gradient(phi_ar, dx_phi_0, axis=1, edge_order=1) #!!!

g_2 = np.gradient(vect_pot_ar_cut, dy_time, axis=0)
g_2_phi0 = np.gradient(phi_ar, dy_time, axis=0, edge_order=1)
#g_2_phi0 = np.diff(phi_ar[0::,0:-1], 1, axis=0)
#g_2_phi0 = np.diff(phi_ar, 2, axis=0)
#print(g_2)
print(g_1.shape, g_2.shape)

g_1 = g_1[1:-1,1:-1]
g_2 = g_2[1:-1,1:-1]
g_1_phi0 = g_1_phi0[1:-1,1:-1]
g_2_phi0 = g_2_phi0[1:-1,1:-1]



g_3 = g_2 - g_1
print("g_3 =",  g_3.shape)
g_3_phi0 = g_2_phi0 - g_1_phi0
print("g_3_phi0 =",  g_3_phi0.shape)
#g_3_phi0 = np.subtract(g_2_phi0,g_1_phi0)
g_3_phi0_max=np.amax(np.imag(g_3_phi0))
print(g_3_phi0_max)
#print(g_3_phi0.shape)

#print(g_1.shape, g_2.shape)


max_phi0 = np.max(np.real(phi_ar))
min_phi0 = np.min(np.real(phi_ar))
max_g_1_phi0 = np.max(np.real(g_1_phi0))
min_g_1_phi0 = np.min(np.real(g_1_phi0))
max_g_2_phi0 = np.max(np.real(g_2_phi0))
min_g_2_phi0 = np.min(np.real(g_2_phi0))

print('derivative phi=0:  max =', max_phi0, 'min =', min_phi0)
print('derivative g_1 phi=0:  max =', max_g_1_phi0, 'min =', min_g_1_phi0)
print('derivative g_2 phi=0:  max =', max_g_2_phi0, 'min =', min_g_2_phi0)

# step 3
g_4 = g_3 - g_3_phi0

# step 4
####### 2d2 FT rad_to_ft ###########
f = np.fft.fft2(g_4, norm="forward")
fshift = np.fft.fftshift(f)
n = g_4[:,0].size
freq = np.fft.fftfreq(n, d=dy_time)*2*np.pi
n_phi = g_4[0,:].size
#freq_phi = np.fft.fftfreq(n_phi, d=timestep_phi)*0.43/0.8256*2*np.pi
#freq_phi = np.fft.fftfreq(n_phi, d=timestep_phi)*0.8256*2*np.pi
freq_phi = np.fft.fftfreq(n_phi, d=dx_phi)*2*np.pi

max_fshift = np.max(np.abs(fshift))
min_fshift = np.min(np.abs(fshift))
#print('phi=0:  max =', max_fshift, 'min =', min_fshift)
##############
####################


####### case of phi=pi/2 shift ########
phi1_pi2 = np.arange(0, l_A, 1)

phi_ar_pi2 = np.array([])
for i in phi1_pi2:
    phi_i = np.roll(phi_pi2, -i)
    if i==0:
        phi_ar_pi2=phi_pi2
    if i>0:
        phi_ar_pi2 = np.column_stack((phi_ar_pi2,phi_i))


print("phi_ar_pi2_0 =",  phi_ar_pi2.shape)
######
phi_ar_pi2_1 = np.roll(phi_pi2, 1)
phi_ar_pi2_2 = np.roll(phi_pi2, 2)
#print(phi_ar1)
phi_ar_pi2 = np.column_stack((phi_ar_pi2_2, phi_ar_pi2_1, phi_ar_pi2))


phi_ar_pi2 = np.row_stack((phi_ar_pi2, phi_ar_pi2[0,:]))
phi_ar_pi2 = np.row_stack((phi_ar_pi2[-2,:], phi_ar_pi2))

print("phi_ar_pi2 =",  phi_ar_pi2.shape)
###########

#steps 1 and 2
g_1_phi_pi2 = np.gradient(phi_ar_pi2, dx_phi_0, axis=1)

g_2_phi_pi2 = np.gradient(phi_ar_pi2, dy_time, axis=0)
#print(g_2)

g_1_phi_pi2 = g_1_phi_pi2[1:-1,1:-1]
g_2_phi_pi2 = g_2_phi_pi2[1:-1,1:-1]
g_3_phi_pi2 = g_2_phi_pi2 - g_1_phi_pi2

g_3_phi_pi2_max=np.amax(np.imag(g_3_phi_pi2))
print(g_3_phi_pi2_max)

# step 3
g_4_pi2 = g_3 - g_3_phi_pi2


# step 4
####### 2d2 FT rad_to_ft ###########
f_pi2 = np.fft.fft2(g_4_pi2, norm="forward")
fshift_pi2 = np.fft.fftshift(f_pi2)

max_fshift_pi2 = np.max(np.abs(fshift_pi2))
min_fshift_pi2 = np.min(np.abs(fshift_pi2))
#print('phi=pi/2:  max =', max_fshift_pi2, 'min =', min_fshift_pi2)
##############

fshift_minus = fshift_pi2 - fshift

############ FT time ###########g_3_phi0
#f_ch_time = np.fft.fft(g_4, axis=0, norm="forward")
f_ch_time = np.fft.fft(g_4, axis=0, norm="forward")
fshift_ch_time = np.fft.fftshift(f_ch_time)

f_ch_phi = np.fft.fft(g_4, axis=1, norm="forward")
fshift_ch_phi = np.fft.fftshift(f_ch_phi)

f_ch_2d = np.fft.fft(f_ch_time, axis=1, norm="forward")
fshift_ch_2d = np.fft.fftshift(f_ch_2d)








######### GABOR TRANFORM PART ##########

#plt.rcParams.update({'font.size': 22})
##############  defining arrays  #############
timeFS = np.array([])
Current_fs = np.array([])
#E_field = np.array([])

#PARAMETERS for Fourier transform:
MaxFreq2D=3

#PARAMETERS for Gabor transform:
wPlotMax=16
tick_spacingW = 2#2.0
# tick step in W (eV)
tick_spacingT = 2
# tick step in t (fs) 
# wPlotMax, tick_spacing in eV for Gabor 3D plot

# Desired width of Gabor slice in fs (will be slightly adjusted, see GaborSliceWidth)
GaborSliceWidth=0.2 #defolt 0.5

# Width of Gabor Gaussian FWHM in fs
GaborFWHM=2.5#4#8.0


#LCO-XY-Linear-polarization-specific script !!!!!!!!!!!!!!!!!!!!!!!!
renorm=0.43
#############  script for LCO, a0 is in-plane lattice constant  !!!!!!!!!!
a_0=3.78e-10 
# one E component only available in control and pulse files,
# should be read and multiplied by Efactor=sqrt(2) 
Efactor=np.sqrt(2)

### distance between consequent time points from input files in fs
#spacing = float(inpDict["__h"])*0.66/renorm
spacing = 0.02*0.6582119514/renorm
spacing_time = 0.01*0.6582119514/renorm
### pulse central frequency in eV
#omega_eV= float(inpDict["__omega"])*renorm
omega_eV = 1.92*renorm
############### reading files and converting data to the proper units  ##########################
### Current and time:
timeFS = time_vp_ar_cut*spacing_time
Current_fs = g_4[:,40]#*renorm*Efactor

##### read data
# Number of sample points
Nt=len(Current_fs)
Ns=np.int(Nt*spacing/(GaborSliceWidth))

# new Gabor Width, in order to maintain integer number of slices
GaborSliceWidth=Nt*spacing/Ns
Current_eV = np.abs(fft(Current_fs))

FreqStep=4.13566767901/(Nt*spacing)
wmax=0.5*Nt*FreqStep
xf =  np.linspace(0.0, wmax, Nt//2)

# preparing an array of Gaussian enveloped current with stepwise shifted window

EnvelopedCurrent_fs=[[Current_fs[j]*np.exp(-((j*spacing)-i*GaborSliceWidth)**2/((GaborFWHM**2)/(FourLn2))) for j in range(0,Nt)] for i in range(0,Ns)]


GaborCurrent_eV=[np.abs(fft(np.asarray(EnvelopedCurrent_fs[i])))[0:Nt] for i in range(0,Ns)]
print("GaborCurrent_eV =", len(GaborCurrent_eV))



########################  Plotting figure  ########################
fig = plt.figure()
axs0 = fig.add_axes([0.15, 0.73, 0.8, 0.2])
axs1 = fig.add_axes([0.15, 0.1, 0.8, 0.63])

plt.tick_params(top='on', bottom='off', left='on', right='on', labelleft='on', labelbottom='off',direction="in")
#axs1 = fig.add_subplot(gridspace_fs[1,:])
plt.tick_params(top='on', bottom='off', left='on', right='on', labelleft='on', labelbottom='off',direction="in")

maxJ=max(np.abs(Current_fs))
axs0.set_xlim(0,32.5)
axs1.set_xlim(0,32.5)
axs1.set_ylim(-6,6)


###### plot current #####
plotJre, = axs0.plot(timeFS, np.real(Current_fs), label="$Re$", color='r')
plotJim, = axs0.plot(timeFS, np.imag(Current_fs), label="$Im$", color='b')
axs0.set_ylabel(r"$G^{<}(time,\omega=LHB)$")

plt.figlegend((plotJre, plotJim), ('$Re$', '$Im$'), bbox_to_anchor=[0.95, 0.94], loc='upper right')
plt.setp(axs0.get_xticklabels(), visible=False)

# making a generator, giving us four (x,y) tuples of rectangle vertices coordinates:

rectVertexes1 = (((0+i,0+k),(0+i,FreqStep+k+0.05),(GaborSliceWidth+i+0.05,FreqStep+k+0.05),(GaborSliceWidth+i+0.05,0+k)) for i in np.arange(0, GaborSliceWidth*Ns, GaborSliceWidth) for k in (np.arange(-FreqStep*(Nt), 0, FreqStep)))

rectVertexes2 = (((0+i,0+k),(0+i,FreqStep+k+0.05),(GaborSliceWidth+i+0.05,FreqStep+k+0.05),(GaborSliceWidth+i+0.05,0+k)) for i in np.arange(0, GaborSliceWidth*Ns, GaborSliceWidth) for k in (np.arange(0, FreqStep*(Nt), FreqStep)))



#### plot Gabor ###
#colll = PolyCollection(rectVertexes, array=np.array(GaborCurrent_eV).reshape(Ns*(Nt//2)), linewidths=0, alpha=1.0, norm = LogNorm(vmin=0.003, vmax=50), cmap="jet") # !!!

colll1 = PolyCollection(rectVertexes1, array=np.array(GaborCurrent_eV).reshape(Ns*(Nt)), linewidths=0, alpha=1.0, norm = LogNorm(vmin=0.003, vmax=50), cmap="jet")

colll2 = PolyCollection(rectVertexes2, array=np.array(GaborCurrent_eV).reshape(Ns*(Nt)), linewidths=0, alpha=1.0, norm = LogNorm(vmin=0.003, vmax=40), cmap="jet")

# adding the collection to the axis
axs1.add_collection(colll1)
axs1.add_collection(colll2)
#cbaxes1 = fig.add_axes([0.8, 0.11, 0.03, 0.21])
cbaxes1 = fig.add_axes([0.84, 0.12, 0.03, 0.24])
#cbaxes1 = fig.add_axes([0.82, 0.11, 0.03, 0.21]) 
cb1 = plt.colorbar(colll2, cax = cbaxes1)
cbytick_obj1 = plt.getp(cb1.ax.axes, 'yticklabels')
plt.setp(cbytick_obj1, color='w')
tick_locator_1 = ticker.LogLocator(base=10.0, numticks=25)
cb1.locator = tick_locator_1
cb1.update_ticks()
#axs1.set_xlabel('Time (fs)')
axs1.set_xlabel('$Time$, $fs$')

cm = plt.get_cmap("terrain")
#axs1.axhline(y=1.25, color='k', linestyle='--')
#axs1.axhline(y=1.25+0.8256, color='k', linestyle='--')
#axs1.axhline(y=0.8256, color='k', linestyle='--')
#axs1.axhline(y=2*0.8256, color='k', linestyle='--')
#axs1.axhline(y=3*0.8256, color='k', linestyle='--')
axs1.yaxis.set_major_locator(plt.MaxNLocator(6))
tick_locator = ticker.MaxNLocator(nbins=4)

axs1.set_ylabel(r'$E$, $eV$')

plt.setp(axs1.get_yticklabels()[-1], visible=False)
#plt.setp(axs2.get_xticklabels(), visible=False)
axs0.yaxis.set_major_locator(plt.MaxNLocator(6))

plt.show()

#plt.savefig("CEP_pi2_clean_FWHM_E_2_0HB.pdf", dpi=150)
