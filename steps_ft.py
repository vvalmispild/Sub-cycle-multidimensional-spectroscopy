import h5py as h5
import matplotlib as mpl
mpl.use('MacOSX')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pylab
from scipy import ndimage, misc
from numpy import arange, cos, sin
import matplotlib.ticker as ticker

mpl.rc('xtick', labelsize=13)
mpl.rc('ytick', labelsize=13)

plt.rcParams['text.usetex'] = True

#plt.rcParams['text.latex.preamble'] = [r"\usepackage{siunitx}"]

E_l_ar = np.array([])

time_vp_1 = np.load("./data_all_E/data_E_05/data_0topi2/time_ar_4.npy")
#time_vp_2 = np.load("./data_pi2topi_2/time_pulse_2.npy")
#time_vp_ar = np.concatenate((time_vp_1, time_vp_2), axis=1)

vect_pot_1 = np.load("./data_all_E/data_E_05/data_0topi2/LHB_4.npy")
vect_pot_2 = np.load("./data_all_E/data_E_05/data_pi2topi/LHB_4.npy")
# 1-period
#vect_pot_ar_0 = np.concatenate((vect_pot_1, vect_pot_2), axis=1)/(np.pi*0.43)

# 2-periods
vect_pot_ar_0 = np.concatenate((vect_pot_1, vect_pot_2, vect_pot_1, vect_pot_2), axis=1)/(np.pi*0.43)

vect_pot_ar_0 = vect_pot_ar_0[::1]
time_vp_ar_0 = time_vp_1[::1]
print(vect_pot_ar_0.shape)
#### cut first 5 fs of the data ###

vect_pot_ar_cut = np.delete(vect_pot_ar_0, np.s_[0:162:1], axis=0)
time_vp_ar_cut = np.delete(time_vp_ar_0, np.s_[0:162:1], axis=0)
#vect_pot_ar_cut = np.imag(vect_pot_ar_cut)
l_A= len(vect_pot_ar_0[0,:])



####### phi=0 shift ########
phi1 = np.arange(0, l_A, 1)
time_coef = 0.01*0.6582119514/0.43
phi_coef = 0.039269*0.6582119514/0.8256
phi_0 = vect_pot_ar_cut[:,0]
phi_pi2 = vect_pot_ar_cut[:,40]


phi_ar = np.array([])
for i in phi1:
    phi_i = np.roll(phi_0, -i)
    if i==0:
        phi_ar = phi_0
    if i>0:
        phi_ar = np.column_stack((phi_ar,phi_i))
     
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

#steps 1 and 2
dx_phi = 0.039269/0.8256
dy_time = 0.02/0.43
dx_phi_0 = 0.0384/0.8256 #!!!!
g_1 = np.gradient(vect_pot_ar_cut, dx_phi, axis=1)
g_1_phi0 = np.gradient(phi_ar, dx_phi_0, axis=1, edge_order=1) #!!!

g_2 = np.gradient(vect_pot_ar_cut, dy_time, axis=0)
g_2_phi0 = np.gradient(phi_ar, dy_time, axis=0, edge_order=1)
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
f = np.fft.ifft2(g_4, norm="forward")
n = g_4[:,0].size
freq = np.fft.fftfreq(n, d=dy_time)*2*np.pi
freq_time_shift = np.fft.fftshift(freq)

n_phi = g_4[0,:].size
#freq_phi = np.fft.fftfreq(n_phi, d=timestep_phi)*0.43/0.8256*2*np.pi
#freq_phi = np.fft.fftfreq(n_phi, d=timestep_phi)*0.8256*2*np.pi
freq_phi = np.fft.fftfreq(n_phi, d=dx_phi)*2*np.pi
freq_phi_shift = np.fft.fftshift(freq_phi)


###############################
####### phi=pi/2 shift ########
phi1_pi2 = np.arange(0, l_A, 1)

phi_ar_pi2 = np.array([])
for i in phi1_pi2:
    phi_i = np.roll(phi_pi2, -i)
    if i==0:
        phi_ar_pi2=phi_pi2
    if i>0:
        phi_ar_pi2 = np.column_stack((phi_ar_pi2,phi_i))

print("phi_ar_pi2_0 =",  phi_ar_pi2.shape)
###########
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
##############

fshift_minus = fshift_pi2 - f

############ FT time ###########g_3_phi0
#f_ch_time = np.fft.fft(g_4, axis=0, norm="forward")
f_ch_time = np.fft.fft(g_4, axis=0)
fshift_ch_time = np.fft.fftshift(f_ch_time)

f_ch_phi = np.fft.fft(g_4, axis=1, norm="forward")
fshift_ch_phi = np.fft.fftshift(f_ch_phi)

f_ch_2d = np.fft.fft(f_ch_time, axis=1, norm="forward")
fshift_ch_2d = np.fft.fftshift(f_ch_2d)



###### Plots ########
###### QP ###########
fig = plt.figure()
plt.xlabel(r"$\tau $", fontsize=13)
plt.ylabel("$\omega, $ $eV$", fontsize=13)
plot=pylab.imshow(np.abs(fshift_ch_time), cmap='gist_ncar', aspect=0.48, interpolation='bilinear', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(freq_time_shift), np.amax(freq_time_shift)])
#plt.plot(x,y,linestyle='-',color='w')
#plt.xlim(-4, 4)
plt.ylim(-5, 5)

#, origin='lower'
#plt.clim(min_fshift_pi2,max_fshift_pi2)
fig.colorbar(plot)
#plt.savefig("./picl/q_mag_g4_FT_time.pdf", dpi=150)
plt.show()

'''
fig = plt.figure()
plt.xlabel(r"$\omega_{CEP}, $ $eV$", fontsize=13)
plt.ylabel(r"$\omega, $ $eV$", fontsize=13)
#plot = pylab.imshow(np.abs(fshift), extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
plot = pylab.imshow(np.abs(f), cmap='gist_ncar', origin='lower', interpolation='bilinear')#, extent=[np.amin(freq_phi_shift), np.amax(freq_phi_shift),np.amin(freq_time_shift), np.amax(freq_time_shift)])
#plot = pylab.imshow(np.abs(f), cmap='gist_ncar', origin='lower', interpolation='bilinear')#, extent=[np.amin(freq_phi_shift), np.amax(freq_phi_shift),np.amin(freq_time_shift), np.amax(freq_time_shift)])
#plt.plot(x,y,linestyle='-',color='w')
#plt.xlim(-8, 8)
#plt.ylim(-8, 8)

fig.colorbar(plot)
#plt.clim(min_fshift_pi2,max_fshift_pi2)
#plt.savefig("./picl/q_mag_0.pdf", dpi=150)
plt.show()

'''

'''
plt.axhline(y=0, color='w', linestyle='--')
plt.axhline(y=0+0.8256, color='w', linestyle='--')
plt.axhline(y=0+2*0.8256, color='w', linestyle='--')
plt.axhline(y=0+3*0.8256, color='w', linestyle='--')
plt.axhline(y=0+4*0.8256, color='w', linestyle='--')
plt.axhline(y=0+5*0.8256, color='w', linestyle='--')

plt.axvline(x=0, color='w', linestyle='--')
plt.axvline(x=0+0.8256, color='w', linestyle='--')
plt.axvline(x=0+2*0.8256, color='w', linestyle='--')
plt.axvline(x=0+3*0.8256, color='w', linestyle='--')
plt.axvline(x=0+4*0.8256, color='w', linestyle='--')
plt.axvline(x=0+5*0.8256, color='w', linestyle='--')

plt.axhline(y=0, color='w', linestyle='--')
plt.axhline(y=0-0.8256, color='w', linestyle='--')
plt.axhline(y=0-2*0.8256, color='w', linestyle='--')
plt.axhline(y=0-3*0.8256, color='w', linestyle='--')
plt.axhline(y=0-4*0.8256, color='w', linestyle='--')
plt.axhline(y=0-5*0.8256, color='w', linestyle='--')

plt.axvline(x=0, color='w', linestyle='--')
plt.axvline(x=0-0.8256, color='w', linestyle='--')
plt.axvline(x=0-2*0.8256, color='w', linestyle='--')
plt.axvline(x=0-3*0.8256, color='w', linestyle='--')
plt.axvline(x=0-4*0.8256, color='w', linestyle='--')
plt.axvline(x=0-5*0.8256, color='w', linestyle='--')
'''
'''
###### LHB ###########

fig = plt.figure()
plt.xlabel(r"$\tau $", fontsize=13)
plt.ylabel("$\omega, $ $eV$", fontsize=13)
plot=pylab.imshow(np.abs(fshift_ch_time), cmap='gist_ncar', origin='lower', aspect=0.48, interpolation='bilinear', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(freq), np.amax(freq)])
#plt.plot(x,y,linestyle='-',color='w')
#plt.xlim(-4, 4)
plt.ylim(-5, 5)
plt.axhline(y=1.25, color='r', linestyle='--')
plt.axhline(y=1.25+0.8256, color='w', linestyle='--')
plt.axhline(y=1.25+2*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25+3*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25+4*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25+5*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25+6*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25+7*0.8256, color='w', linestyle='--')


#plt.axhline(y=1.25, color='w', linestyle='--')
plt.axhline(y=1.25-0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-2*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-3*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-4*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-5*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-6*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-7*0.8256, color='w', linestyle='--')
#plt.axhline(y=0-0.8256, color='w', linestyle='--')
#plt.clim(min_fshift_pi2,max_fshift_pi2)
fig.colorbar(plot)
plt.savefig("./picl/mag_g4_FT_time.pdf", dpi=150)
#plt.show()


fig = plt.figure()
plt.xlabel(r"$\omega_{CEP}, $ $eV$", fontsize=13)
plt.ylabel(r"$\omega, $ $eV$", fontsize=13)
#plot = pylab.imshow(np.abs(fshift), extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
plot = pylab.imshow(np.abs(fshift), cmap='gist_ncar', origin='lower', interpolation='bilinear', extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
#plt.plot(x,y,linestyle='-',color='w')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.axhline(y=1.25, color='r', linestyle='--')
plt.axhline(y=1.25+0.8256, color='w', linestyle='--')
plt.axhline(y=1.25+2*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25+3*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25+4*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25+5*0.8256, color='w', linestyle='--')
#plt.axhline(y=1.25+6*0.8256, color='w', linestyle='--')
#plt.axhline(y=1.25+7*0.8256, color='w', linestyle='--')
#plt.axhline(y=1.25+8*0.8256, color='w', linestyle='--')

plt.axhline(y=1.25-0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-2*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-3*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-4*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-5*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-6*0.8256, color='w', linestyle='--')
plt.axhline(y=1.25-7*0.8256, color='w', linestyle='--')

plt.axvline(x=1.25, color='r', linestyle='--')
plt.axvline(x=1.25+0.8256, color='w', linestyle='--')
plt.axvline(x=1.25+2*0.8256, color='w', linestyle='--')
plt.axvline(x=1.25+3*0.8256, color='w', linestyle='--')
plt.axvline(x=1.25+4*0.8256, color='w', linestyle='--')
plt.axvline(x=1.25+5*0.8256, color='w', linestyle='--')
#plt.axvline(x=1.25+6*0.8256, color='w', linestyle='--')
#plt.axvline(x=1.25+7*0.8256, color='w', linestyle='--')

plt.axvline(x=1.25-0.8256, color='w', linestyle='--')
plt.axvline(x=1.25-2*0.8256, color='w', linestyle='--')
plt.axvline(x=1.25-3*0.8256, color='w', linestyle='--')
plt.axvline(x=1.25-4*0.8256, color='w', linestyle='--')
plt.axvline(x=1.25-5*0.8256, color='w', linestyle='--')
plt.axvline(x=1.25-6*0.8256, color='w', linestyle='--')
plt.axvline(x=1.25-7*0.8256, color='w', linestyle='--')
fig.colorbar(plot)
plt.clim(min_fshift_pi2,max_fshift_pi2)
plt.savefig("./picl/mag_0.pdf", dpi=150)
#plt.show()








###### LHB ###########

fig = plt.figure()
plt.xlabel(r"$\tau $", fontsize=13)
plt.ylabel("$\omega, $ $eV$", fontsize=13)
plot=pylab.imshow(np.abs(fshift_ch_time), cmap='gist_ncar', origin='lower', aspect=0.48, interpolation='bilinear', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(freq), np.amax(freq)])
#plt.plot(x,y,linestyle='-',color='w')
#plt.xlim(-4, 4)
plt.ylim(-5, 5)
plt.axhline(y=-1.25, color='r', linestyle='--')
plt.axhline(y=-1.25+0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+2*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+3*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+4*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+5*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+6*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+7*0.8256, color='w', linestyle='--')
#plt.axhline(y=1.25, color='w', linestyle='--')
plt.axhline(y=-1.25-0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25-2*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25-3*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25-4*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25-5*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25-6*0.8256, color='w', linestyle='--')
#plt.axhline(y=0-0.8256, color='w', linestyle='--')
#plt.clim(min_fshift_pi2,max_fshift_pi2)
fig.colorbar(plot)
plt.savefig("./picl/u_mag_g4_FT_time.pdf", dpi=150)
#plt.show()


fig = plt.figure()
plt.xlabel(r"$\omega_{CEP}, $ $eV$", fontsize=13)
plt.ylabel(r"$\omega, $ $eV$", fontsize=13)
#plot = pylab.imshow(np.abs(fshift), extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
plot = pylab.imshow(np.abs(fshift), cmap='gist_ncar', origin='lower', interpolation='bilinear', extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
#plt.plot(x,y,linestyle='-',color='w')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.axhline(y=-1.25, color='r', linestyle='--')
plt.axhline(y=-1.25+0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+2*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+3*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+4*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+5*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+6*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+7*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25+8*0.8256, color='w', linestyle='--')

plt.axhline(y=-1.25-0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25-2*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25-3*0.8256, color='w', linestyle='--')
plt.axhline(y=-1.25-4*0.8256, color='w', linestyle='--')
#plt.axhline(y=-1.25-5*0.8256, color='w', linestyle='--')
#plt.axhline(y=-1.25-6*0.8256, color='w', linestyle='--')

plt.axvline(x=-1.25, color='r', linestyle='--')
plt.axvline(x=-1.25+0.8256, color='w', linestyle='--')
plt.axvline(x=-1.25+2*0.8256, color='w', linestyle='--')
plt.axvline(x=-1.25+3*0.8256, color='w', linestyle='--')
plt.axvline(x=-1.25+4*0.8256, color='w', linestyle='--')
plt.axvline(x=-1.25+5*0.8256, color='w', linestyle='--')
plt.axvline(x=-1.25+6*0.8256, color='w', linestyle='--')
plt.axvline(x=-1.25+7*0.8256, color='w', linestyle='--')

plt.axvline(x=-1.25-0.8256, color='w', linestyle='--')
plt.axvline(x=-1.25-2*0.8256, color='w', linestyle='--')
plt.axvline(x=-1.25-3*0.8256, color='w', linestyle='--')
plt.axvline(x=-1.25-4*0.8256, color='w', linestyle='--')
#plt.axvline(x=-1.25-5*0.8256, color='w', linestyle='--')
#plt.axvline(x=-1.25-6*0.8256, color='w', linestyle='--')
fig.colorbar(plot)
plt.clim(min_fshift_pi2,max_fshift_pi2)
plt.savefig("./picl/u_mag_0.pdf", dpi=150)
#plt.show()



'''






'''

####### Plot ###########
#x = np.arange(-9, 9, 1)
#y = x
# Introduction

###############
fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.real(vect_pot_ar_cut), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/re_LHB.pdf", dpi=150)
#plt.show()


fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.imag(vect_pot_ar_cut), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/im_LHB.pdf", dpi=150)


fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.real(phi_ar), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/re_phi0.pdf", dpi=150)

fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.imag(phi_ar), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/im_phi0.pdf", dpi=150)
#plt.show()



# Step 1 #
#############
fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.real(g_1), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/re_g_1.pdf", dpi=150)

fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.imag(g_1), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/im_g_1.pdf", dpi=150)

###########
fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.real(g_2), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/re_g_2.pdf", dpi=150)

fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.imag(g_2), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/im_g_2.pdf", dpi=150)

fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.real(g_3), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/re_g_3.pdf", dpi=150)

fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.imag(g_3), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/im_g_3.pdf", dpi=150)
#plt.show()



# Step 2 #
############
fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.real(g_1_phi0), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
#plt.show()
plt.savefig("./picl/re_g_1_phi0.pdf", dpi=150)

fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.imag(g_1_phi0), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/im_g_1_phi0.pdf", dpi=150)

############
fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.real(g_2_phi0), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/re_g_2_phi0.pdf", dpi=150)

fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.imag(g_2_phi0), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/im_g_2_phi0.pdf", dpi=150)

#########
fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.real(g_3_phi0), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
#plt.clim(-0.001,0.001)
plt.savefig("./picl/re_g_3_phi0.pdf", dpi=150)
#plt.show()
fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.imag(g_3_phi0), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
#plt.clim(-0.001,0.001)
plt.savefig("./picl/im_g_3_phi0.pdf", dpi=150)
#plt.show()



# Step 3 #
#########

fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.real(g_4), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/re_g_4.pdf", dpi=150)


fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel(r"$time,$ $fs$", fontsize=13)
plot=pylab.imshow(np.imag(g_4), aspect=0.17 ,origin='lower', extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(time_vp_ar_cut*time_coef), np.amax(time_vp_ar_cut*time_coef)])
fig.colorbar(plot)
plt.savefig("./picl/im_g_4.pdf", dpi=150)
#plt.show()
##########

fig = plt.figure()
plt.xlabel(r"$\omega_{CEP}, $ $eV$", fontsize=13)
plt.ylabel(r"$\omega, $ $eV$", fontsize=13)
#plot = pylab.imshow(np.abs(fshift), extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
plot = pylab.imshow(np.abs(fshift), cmap='gist_ncar', origin='lower', extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
#plt.plot(x,y,linestyle='-',color='w')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
fig.colorbar(plot)
plt.clim(min_fshift_pi2,max_fshift_pi2)
plt.savefig("./picl/mag_0.pdf", dpi=150)
#plt.show()

fig = plt.figure()
plt.xlabel(r"$\omega_{CEP}, $ $eV$", fontsize=13)
plt.ylabel(r"$\omega, $ $eV$", fontsize=13)
#plot = pylab.imshow(np.abs(fshift), extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
plot = pylab.imshow(np.abs(fshift_pi2), cmap='gist_ncar', origin='lower', extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
#plt.plot(x,y,linestyle='-',color='w')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
fig.colorbar(plot)
plt.clim(min_fshift_pi2,max_fshift_pi2)
plt.savefig("./picl/mag_pi2.pdf", dpi=150)
#plt.show()

fig = plt.figure()
plt.xlabel(r"$\omega_{CEP}, $ $eV$", fontsize=13)
plt.ylabel(r"$\omega, $ $eV$", fontsize=13)
#plot = pylab.imshow(np.abs(fshift), extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
plot = pylab.imshow(np.abs(fshift_minus), cmap='gist_ncar', origin='lower', extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
plt.clim(min_fshift_pi2,max_fshift_pi2)
#plt.plot(x,y,linestyle='-',color='w')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
fig.colorbar(plot)
plt.clim(min_fshift_pi2,max_fshift_pi2)
plt.savefig("./picl/mag_m.pdf", dpi=150)
#plt.show()

###########
fig = plt.figure()
plt.xlabel(r"$\omega_{CEP}, $ $eV$", fontsize=13)
plt.ylabel(r"$\omega, $ $eV$", fontsize=13)
plot = pylab.imshow(np.angle(fshift), origin='lower', extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
#plt.plot(x,y,linestyle='-',color='w')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
fig.colorbar(plot)
plt.savefig("./picl/phase_0.pdf", dpi=150)

fig = plt.figure()
plt.xlabel(r"$\omega_{CEP}, $ $eV$", fontsize=13)
plt.ylabel(r"$\omega, $ $eV$", fontsize=13)
plot = pylab.imshow(np.angle(fshift_pi2), origin='lower', extent=[np.amin(freq_phi), np.amax(freq_phi),np.amin(freq), np.amax(freq)])
#plt.plot(x,y,linestyle='-',color='w')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
fig.colorbar(plot)
plt.savefig("./picl/phase_pi2.pdf", dpi=150)

###########
#aspect=0.25 for 0-pi
### FT time ##
fig = plt.figure()
plt.xlabel(r"$\tau,$ $fs$", fontsize=13)
plt.ylabel("$\omega, $ $eV$", fontsize=13)
plot=pylab.imshow(np.abs(fshift_ch_time), cmap='gist_ncar', origin='lower', aspect=0.4, extent=[np.amin(phi1*phi_coef), np.amax(phi1*phi_coef), np.amin(freq), np.amax(freq)])
#plt.plot(x,y,linestyle='-',color='w')
#plt.xlim(-4, 4)
plt.ylim(-6, 6)
#plt.clim(min_fshift_pi2,max_fshift_pi2)
fig.colorbar(plot)
plt.savefig("./picl/mag_g4_FT_time.pdf", dpi=150)
#plt.show()
'''
