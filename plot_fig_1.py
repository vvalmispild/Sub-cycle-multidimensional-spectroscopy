import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
fsize = 21.2
plt.rc('xtick', labelsize=fsize)
plt.rc('ytick', labelsize=fsize)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams['figure.figsize'] = [12, 7]

ax = plt.gca()
array_txt = np.loadtxt("Spectr_les.out",usecols=(0,1,2))
x = array_txt[:,0]*0.01*0.66/0.43
y = array_txt[:,1]*0.43
z = array_txt[:,2]/(np.pi*0.43)
 
x=np.unique(x)
y=np.unique(y)
X,Y = np.meshgrid(x,y)
Z=z.reshape(len(x),len(y))
Z=np.transpose(Z)
data = Z

im = ax.imshow(data, vmin = 0, vmax = 0.35, cmap = 'terrain', interpolation = 'gaussian', origin='lower',\
           aspect=2.6,  extent = [min(x), max(x), 4, -4])

ax.invert_yaxis()
ax.set_xlabel(r'$\mathdefault{Time}$, $\mathdefault{fs}$', fontsize=fsize, labelpad=12)
ax.set_ylabel(r'$\mathdefault{Energy}$, $\mathdefault{eV}$', fontsize=fsize)
ax.tick_params(axis='x',pad=8)

plt.xticks(np.arange(min(x), max(x)+1, 5.0))
plt.yticks(np.arange(-4, 5, 1.0))

# pulse
pulse_E = np.loadtxt("Pulse_xy.dat")
x_pulse_E = pulse_E[:,0]*0.66/0.43
y_pulse_E = -pulse_E[:,1]*5.5
plt.plot(x_pulse_E, y_pulse_E,'tab:orange')

# shadow
plt.axvspan(0, 2.5, color='black', alpha=0.5, lw=0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=-4.73)

# cbar
cbar = plt.colorbar(im, cax=cax)
nbins = 7
tick_locator = ticker.MaxNLocator(nbins=nbins)
cbar.locator = tick_locator
cbar.update_ticks()

plt.show()
