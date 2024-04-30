import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 25})
mpl.rcParams['font.family']='serif'
mpl.rc('text', usetex=True)
plt.rcParams.update({'lines.linewidth': 3.75})

kB = 1.381e-23 * 6.022e23 / (4.184*1000) # Bolztmann constant in kcalmol-1K-1
T = 300 #K
beta = 1/(kB*T)
kBT_Ha = 0.00095 #1kBT in Ha

def colorFader(c1, c2, mix=0):
	c1=np.array(mpl.colors.to_rgb(c1))
	c2=np.array(mpl.colors.to_rgb(c2))
	return mpl.colors.to_hex((1.-mix)*c1 + mix*c2)

def DoS(E,theta):
	"Linear model for the density of states in TBG"
	L = (0.246/0.529)/(2*np.sin((1.1*np.pi/180)/2)) # Moire lattice constant at magic angle
	ell = 1
	v_max = 1*(1/(2.2)) # Dirac velocity of monolayer graphene in au
	theta_m = 1.1*np.pi/180 # Magic angle
	N=8
	v_F = v_max*np.abs(np.sin(3*(theta-theta_m)/2)) 
	s=9e4
	E_c = 0.5*(np.sqrt((np.pi/s)+4*(v_F)**2)-np.sqrt(np.pi/s))
	if np.abs(E) < E_c:
		D = np.abs(E)/(v_F**2)
	elif E < -E_c:
		D = (np.abs(E_c)/(v_F**2))*np.exp(-s*(E+E_c)**2)
	else:
		D = (np.abs(E_c)/(v_F**2))*np.exp(-s*(E-E_c)**2)
	return D/L**2

def l_TF(epsilon,theta):
	"Thomas-Fermi screening length"
	return 0.529/np.sqrt(DoS(epsilon,theta))

def l_TFgraphene(theta):
	"Screening length appropriate for a graphene monolayer"
	diel = 35
	rho = 1e11*(5.29e-11/1e-2)**2
	v_max = 1*(1/(2.2)) # Dirac velocity of monolayer graphene in au
	theta_m = 1.1*np.pi/180
	v_0 = 0
	v_F = v_max*np.abs(np.sin(3*(theta-theta_m)/2)) + v_0
	l = (diel*0.529*v_F)/(4*np.sqrt(np.pi*rho))
	return l

def xi_fit(x, a, b, c, f):
	"Functional form to fit the image charge scaling function"
	return (a/(b+c*(x-f)**2)) 

def pdE(E,l):
	"Screening-dependent energy gap dsitributions"
	driving=0
	bulk = 32.8*kBT_Ha
	reorg =(xi_fit(l,*popt)/(4*4.5593/0.529)) + bulk 
	return (1/np.sqrt(4*np.pi*kBT_Ha*reorg))*np.exp(-(E-(driving+reorg))**2/(4*kBT_Ha*reorg))

def pdE_ref(E):
	"Screening-independent energy gap dsitributions"
	driving=0
	bulk = 32.8*kBT_Ha
	reorg = bulk 
	return (1/np.sqrt(4*np.pi*kBT_Ha*reorg))*np.exp(-(E-(driving+reorg))**2/(4*kBT_Ha*reorg))

def fermi(E):
	"Fermi-Dirac distribution"
	return 1/(1+np.exp(E/kBT_Ha))

epsilon = 4e-5 # finite Fermi energy from quantum capacitance

# Read in image potential data from empty capacitor
lTF = np.arange(0,20.2,0.2)
lTF = list(lTF)
for i in range(21,51):
	lTF.append(i)
for i in range(6,51):
	lTF.append(10*i)
lTF=np.array(lTF)

U=[]

with open('empty_lTFsweep.txt','r') as f:
	lines=f.readlines()
	for line in lines:	
		U.append(float(line.split()[5]))

dE = np.array(U)
xi = ((dE-dE[-1])*0.0016)*4*4.5593/0.529
# Fit image charge scaling function:
popt, pcov = curve_fit(xi_fit, lTF, xi)

eltief=np.linspace(0,100,1000)
reorg_plot =(xi_fit(eltief,*popt)/(4*4.5593/0.529)) + 32.8*kBT_Ha 

#Plot of screening-dependent reorganization energy
#plt.figure()
#plt.plot(eltief,reorg_plot)
#plt.show()

E=np.linspace(-200,200,100000)*kBT_Ha

D=[]
p=[]
lamb=[]
theta_p = np.array([0.8,1.1,1.5,2,2.5,3])
theta = np.linspace(0*np.pi/180,10*np.pi/180,1000)
theta_m=1.1*np.pi/180
for i in range(len(theta_p)):
	D.append([DoS(E[j],theta_p[i]*np.pi/180) for j in range(len(E))])
	lamb.append((xi_fit(l_TF(epsilon,theta_p[i]),*popt)/(4*4.5593/0.529)) + 32.8*kBT_Ha)

#Plot of angle-dependent reorganization energy
#plt.figure()
#plt.plot(180*theta_p/np.pi,np.array(lamb)*27,'-o')
#plt.xlabel(r'$\theta °$')
#plt.ylabel(r'$\lambda/ eV$')
#plt.show()

c1='firebrick'
c2='cornflowerblue'
# Define the colors for your custom colormap
colors = ['firebrick', 'cornflowerblue']      # End at position 1
# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
n=len(theta_p)


#Rate calculations
rate=[]
rate_ref=[]
rate_ref2=[]
rate_ref3=[]
for i in range(len(theta)):
#Full angle-dependent rate
	rate.append(integrate.quad(lambda x: DoS(x,theta[i])*(fermi(x))*pdE(x,l_TF(epsilon,theta[i])),-np.inf,np.inf)[0])
# Constant reorganization energy references
	rate_ref.append(integrate.quad(lambda x: DoS(x,theta[i])*(fermi(x))*pdE(x,l_TF(epsilon,theta[-1])),-np.inf,np.inf)[0])
	rate_ref2.append(integrate.quad(lambda x: DoS(x,theta[i])*(fermi(x))*pdE_ref(x),-np.inf,np.inf)[0])
	rate_ref3.append(integrate.quad(lambda x: DoS(x,theta[i])*(fermi(x))*pdE(x,l_TF(epsilon,theta[0])),-np.inf,np.inf)[0])


fig, ax = plt.subplots(figsize=(8,6))
#ax.plot(theta*180/np.pi,np.array(rate_ref)/rate[0],'darkblue',label=r'$k(\theta;\ell_{TF}(\theta=\pi/3))$')
ax.plot(theta*180/np.pi,np.array(rate)/rate[50],'firebrick',label=r'$\lambda(\theta)$')#r'$k(\theta;\ell_{TF}(\theta))$')
ax.plot(theta*180/np.pi,np.array(rate_ref3)/rate[50],'royalblue',label=r'$\lambda(\theta = 0°)$')#$k(\theta;\ell_{TF}(\theta=0))$')
ax.plot(theta*180/np.pi,(np.array(rate_ref2)/rate[50]),'darkblue',label=r'$\lambda = \lambda_{bulk}$')
ax.scatter(theta[::30]*180/np.pi,np.array(rate)[::30]/rate[50],s=120,facecolors='w',edgecolors='firebrick',linewidths=3,zorder=10)
ax.scatter(theta[::30]*180/np.pi,np.array(rate_ref3)[::30]/rate[50],s=120,facecolors='w',edgecolors='royalblue',linewidths=3,zorder=10)
ax.scatter(theta[::30]*180/np.pi,np.array(rate_ref2[::30])/rate[50],s=120,facecolors='w',edgecolors='darkblue',linewidths=3,zorder=10)
ax.plot(1.1*np.ones(2),np.linspace(-1,1e5,2),'k--')#,label=r'$\theta_m = 1.1 °$')
ax.legend(loc='upper right')
ax.set_xlabel(r'$\theta °$')
plt.ylabel(r'$k(\theta)/k(0)$')
plt.yscale('log')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.tight_layout()
plt.savefig('rate_angle_plot.png')

fig, ax = plt.subplots(figsize=(5,4.5))
ax.scatter(theta[::70]*180/np.pi,np.array([l_TF(epsilon,theta[j]) for j in range(len(theta))])[::70],s=150,facecolors='w',edgecolors='firebrick',linewidths=3,zorder=10)
ax.plot(theta*180/np.pi,np.array([l_TF(epsilon,theta[j]) for j in range(len(theta))]),'firebrick')
ax.set_xlabel(r'$\theta °$')
plt.plot(1.1*np.ones(2),np.linspace(-1,100,2),'k--')#,label=r'$\theta_m = 1.1 °$')
ax.set_ylabel(r'$\ell_{TF}(\theta) / \AA$')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.tight_layout()
plt.savefig('screening_length_plot.png')

fig = plt.figure(figsize=(9,9.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.6],wspace=0.65,hspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])#, sharex=ax1)
for i in range(len(theta_p)):
	ax1.plot(E*27e3,np.array(D[i])/(27e3*0.529**2),color=colorFader(c1,c2,i/n),label=r'$\theta = $'+str(round(theta[i],1))+r' $ °$')
#	plt.plot(E/kBT_Ha,p[i],color=colorFader(c1,c2,i/n),label=r'$\theta = $'+str(round(theta[i],1))+r' $ °$')
ax1.set_xlabel(r'$E/\mathrm{m}e\mathrm{V}$')
ax1.set_ylabel(r'$D(E;\theta)/\mathrm{m}e\mathrm{V}^{-1}\mathrm{\AA}^{-2}$')
ax1.set_xlim(-800,800)
#ax.legend(loc='center right',bbox_to_anchor=(1.05,0.6),fontsize=14)
# Create ScalarMappable object
sm = plt.cm.ScalarMappable(cmap=custom_cmap)
sm.set_array(theta_p)  # Set the data values for the colormap
# Create color-bar
colorbar=fig.colorbar(sm,cax=ax1.inset_axes([0.95, 0.1, 0.05, 0.8])) 
colorbar.set_label(r'$\theta^o$',rotation=60)
ax2.scatter(theta[::80]*180/np.pi,np.array([l_TF(epsilon,theta[j]) for j in range(len(theta))])[::80],s=120,facecolors='w',edgecolors='firebrick',linewidths=3,zorder=10)
ax2.plot(theta*180/np.pi,np.array([l_TF(epsilon,theta[j]) for j in range(len(theta))]),'firebrick')
ax2.set_xlabel(r'$\theta^o$')
ax2.plot(1.1*np.ones(2),np.linspace(-1,100,2),'k--')#,label=r'$\theta_m = 1.1 °$')
ax2.set_ylabel(r'$\ell_{TF}(\theta) / \mathrm{\AA}$')
ax2.set_ylim(-2,90)
ax2.set_xlim(-0.5,6)
ax3.plot(theta*180/np.pi,np.array(rate)/rate[0],'firebrick',linewidth=4.5)#r'$k(\theta;\ell_{TF}(\theta))$')
#ax.plot(theta*180/np.pi,np.array(rate_ref3)/rate[0],'royalblue',label=r'$\lambda(\theta = 0°)$')#$k(\theta;\ell_{TF}(\theta=0))$')
ax3.plot(theta*180/np.pi,(np.array(rate_ref2)/rate[0]),'darkblue',linewidth=4.5)
ax3.scatter(theta[::25]*180/np.pi,np.array(rate)[::25]/rate[0],s=150,facecolors='w',edgecolors='firebrick',linewidths=3,zorder=10,label=r'$\lambda(\theta)$')
#ax.scatter(theta[::50]*180/np.pi,np.array(rate_ref3)[::50]/rate[0],s=150,facecolors='w',edgecolors='royalblue',linewidths=3,zorder=10)
ax3.scatter(theta[::25]*180/np.pi,np.array(rate_ref2[::25])/rate[0],s=150,facecolors='w',edgecolors='darkblue',linewidths=3,zorder=10,label=r'$\lambda = \lambda_{B}$')
ax3.plot(1.1*np.ones(2),np.linspace(-1,1e5,2),'k--')#,label=r'$\theta_m = 1.1 °$')
ax3.legend(loc='upper right', frameon=False)
ax3.set_xlabel(r'$\theta^o$')
ax3.set_ylabel(r'$k(\theta)/k(0)$')
ax3.set_yscale('log')
ax3.set_ylim(0.01,50000)
ax3.set_xlim(-0.2,6.2)
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.tight_layout()
plt.savefig('DoS_lTF_rate_combined.png')
