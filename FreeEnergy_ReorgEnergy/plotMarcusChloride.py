import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from scipy.stats import linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 22})
mpl.rcParams['font.family']='serif'
mpl.rc('text', usetex=True)
plt.rcParams.update({'lines.linewidth': 3.75})

def read_data(file,index1,index2,r,shift):
	x = []
	y = []
	with open(file,'r') as f:
		lines=f.readlines()
		for line in lines:
			try:
				x.append(float(line.split()[index1])*r)
				y.append(-float(line.split()[index2])*r+shift)
			except: continue
	return np.array(x), np.array(y)
def read_datmean(file,index1,index2,ratio):
	x = []
	y = []
	with open(file,'r') as f:
		lines=f.readlines()
		for line in lines:
			try:
				x.append(float(line.split()[index1]))
				y.append(ratio*float(line.split()[index2]))
			except: continue
	return [np.array(x), np.array(y)], np.mean(y)
def I_1(z,l,eps_sol,eps_el):
	"First integral in continuum dielectric theory"
	integrand = lambda x: (4*(eps_sol/eps_el)*x*np.exp(-2*x))/(np.sqrt(x**2+(z/l)**2)+(eps_sol/eps_el)*x)
	I = integrate.quad(integrand,0,np.inf)
	return I[0]
def I_2(z,l,eps_sol,eps_el):
	"Second integral in continuum dielectric theory"
	integrand = lambda x: (2*(z/l)**2*x*np.exp(-2*x))/((np.sqrt(x**2+(z/l)**2)+(eps_sol/eps_el)*x)**2*np.sqrt(x**2+(z/l)**2))
	I = integrate.quad(integrand,0,np.inf)
	return I[0]

kB = 1.381e-23 * 6.022e23 / (4.184*1000) # Bolztmann constant in kcalmol-1K-1
T = 300 #K
beta = 1/(kB*T)

# Read in Free energy surfaces
phi0, betaF_phi0 = read_data('chloride/lTF_0/FES/forward.txt',0,1,1,0)
phi0o, betaF_phi0o = read_data('chloride/lTF_0/FES/backward.txt',0,1,1,0)
phi2, betaF_phi2 = read_data('chloride/lTF_2/FES/forward.txt',0,1,1,0)
phi2o, betaF_phi2o = read_data('chloride/lTF_2/FES/backward.txt',0,1,1,0)
phi5, betaF_phi5 = read_data('chloride/lTF_5/FES/forward.txt',0,1,1,0)
phi5o, betaF_phi5o = read_data('chloride/lTF_5/FES/backward.txt',0,1,1,0)
phi_inf, betaF_phi_inf = read_data('chloride/lTF_inf/FES/forward.txt',0,1,1,0)
phio_inf, betaF_phio_inf = read_data('chloride/lTF_inf/FES/backward.txt',0,1,1,0)

# Horizontal Shift

dF0=0#-21.61483905067803
dF2=0#-23.363827656086297
dF5=0#-24.753643852947956
dFinf=0#-27.505288170483553

phi2-=154.2391+dF2
phi2o-=154.2391+dF2
phi5-=153.5796+dF5
phi5o-=153.5796+dF5
phi0-=152.56204+dF0
phi0o-=152.56204+dF0
phi_inf-=152.69836+dFinf
phio_inf-=152.69836+dFinf

# Linear dependence relation
bF_im2 = betaF_phi2 + phi2
bF_im0 = betaF_phi0 + phi0
bF_im2s = betaF_phi2o - phi2o 
bF_im0s = betaF_phi0o - phi0o 
bF_iminf = betaF_phi_inf + phi_inf
bF_im5 = betaF_phi5 + phi5
bF_imsinf = betaF_phio_inf - phio_inf 
bF_im5s = betaF_phi5o - phi5o 

# Biasing Parameter
#lamb_vals = np.array([-0.2,0,0.2,0.4,0.5,0.6,0.8,1.0,1.2])
lamb_vals = np.array([0.2,0,-0.2,-0.4,-0.5,-0.6,-0.8,-1.0,-1.2])

# Statistics from raw energy gap data
phim12,meanm12 = read_datmean('chloride/lTF_2/Egap_data/lambdam1/sample.txt',0,1,1)
phi02,mean02 = read_datmean('chloride/lTF_2/Egap_data/lambda0/sample.txt',0,1,1)
phi12,mean12 = read_datmean('chloride/lTF_2/Egap_data/lambda1/sample.txt',0,1,1)
phi22,mean22 = read_datmean('chloride/lTF_2/Egap_data/lambda2/sample.txt',0,1,1)
phi32,mean32 = read_datmean('chloride/lTF_2/Egap_data/lambda3/sample.txt',0,1,1)
phi42,mean42 = read_datmean('chloride/lTF_2/Egap_data/lambda4/sample.txt',0,1,1)
phi52,mean52 = read_datmean('chloride/lTF_2/Egap_data/lambda5/sample.txt',0,1,1)
phiTS2,meanTS2 = read_datmean('chloride/lTF_2/Egap_data/lambdaTS/sample.txt',0,1,1)
phip12,meanp12 = read_datmean('chloride/lTF_2/Egap_data/lambdap1/sample.txt',0,1,1)

phim1inf,meanm1inf = read_datmean('chloride/lTF_inf/Egap_data/lambdam1/sample.txt',0,1,1)
phi0inf,mean0inf = read_datmean('chloride/lTF_inf/Egap_data/lambda0/sample.txt',0,1,1)
phi1inf,mean1inf = read_datmean('chloride/lTF_inf/Egap_data/lambda1/sample.txt',0,1,1)
phi2inf,mean2inf = read_datmean('chloride/lTF_inf/Egap_data/lambda2/sample.txt',0,1,1)
phi3inf,mean3inf = read_datmean('chloride/lTF_inf/Egap_data/lambda3/sample.txt',0,1,1)
phi4inf,mean4inf = read_datmean('chloride/lTF_inf/Egap_data/lambda4/sample.txt',0,1,1)
phi5inf,mean5inf = read_datmean('chloride/lTF_inf/Egap_data/lambda5/sample.txt',0,1,1)
phiTSinf,meanTSinf = read_datmean('chloride/lTF_inf/Egap_data/lambdaTS/sample.txt',0,1,1)
phip1inf,meanp1inf = read_datmean('chloride/lTF_inf/Egap_data/lambdap1/sample.txt',0,1,1)

phim10,meanm10 = read_datmean('chloride/lTF_0/Egap_data/lambdam1/sample.txt',0,1,1)
phi00,mean00 = read_datmean('chloride/lTF_0/Egap_data/lambda0/sample.txt',0,1,1)
phi10,mean10 = read_datmean('chloride/lTF_0/Egap_data/lambda1/sample.txt',0,1,1)
phi20,mean20 = read_datmean('chloride/lTF_0/Egap_data/lambda2/sample.txt',0,1,1)
phi30,mean30 = read_datmean('chloride/lTF_0/Egap_data/lambda3/sample.txt',0,1,1)
phi40,mean40 = read_datmean('chloride/lTF_0/Egap_data/lambda4/sample.txt',0,1,1)
phi50,mean50 = read_datmean('chloride/lTF_0/Egap_data/lambda5/sample.txt',0,1,1)
phiTS0,meanTS0 = read_datmean('chloride/lTF_0/Egap_data/lambdaTS/sample.txt',0,1,1)
phip10,meanp10 = read_datmean('chloride/lTF_0/Egap_data/lambdap1/sample.txt',0,1,1)

phim15,meanm15 = read_datmean('chloride/lTF_5/Egap_data/lambdam1/sample.txt',0,1,1)
phi05,mean05 = read_datmean('chloride/lTF_5/Egap_data/lambda0/sample.txt',0,1,1)
phi15,mean15 = read_datmean('chloride/lTF_5/Egap_data/lambda1/sample.txt',0,1,1)
phi25,mean25 = read_datmean('chloride/lTF_5/Egap_data/lambda2/sample.txt',0,1,1)
phi35,mean35 = read_datmean('chloride/lTF_5/Egap_data/lambda3/sample.txt',0,1,1)
phi45,mean45 = read_datmean('chloride/lTF_5/Egap_data/lambda4/sample.txt',0,1,1)
phi55,mean55 = read_datmean('chloride/lTF_5/Egap_data/lambda5/sample.txt',0,1,1)
phiTS5,meanTS5 = read_datmean('chloride/lTF_5/Egap_data/lambdaTS/sample.txt',0,1,1)
phip15,meanp15 = read_datmean('chloride/lTF_5/Egap_data/lambdap1/sample.txt',0,1,1)

# Average energy gap in biased ensembles
means2 = np.array([meanp12,mean02,mean12,mean22,meanTS2,mean32,mean42,mean52,meanm12])- 153.2391 
means0 = np.array([meanp10,mean00,mean10,mean20,meanTS0,mean30,mean40,mean50,meanm10])-152.562
means5 = np.array([meanp15,mean05,mean15,mean25,meanTS5,mean35,mean45,mean55,meanm15])-153.58
meansinf = np.array([meanp1inf,mean0inf,mean1inf,mean2inf,meanTSinf,mean3inf,mean4inf,mean5inf,meanm1inf])-152.6983

# Linear regression of gap as a function of biasing parameter                                                                 
regress0 = linregress(lamb_vals[1:7],means0[1:7])
regress2 = linregress(lamb_vals[1:7],means2[1:7])
regress5 = linregress(lamb_vals[1:7],means5[1:7])
regressinf = linregress(lamb_vals[1:7],meansinf[1:7])
dr_0 = 0.5*((1+regress0.intercept/regress0.slope)*(regress0.slope+regress0.intercept)-(regress0.intercept**2)/regress0.slope)
dr2 = 0.5*((1+regress2.intercept/regress2.slope)*(regress2.slope+regress2.intercept)-(regress2.intercept**2)/regress2.slope)
dr_5 = 0.5*((1+regress5.intercept/regress5.slope)*(regress5.slope+regress5.intercept)-(regress5.intercept**2)/regress5.slope)
dr_inf = 0.5*((1+regressinf.intercept/regressinf.slope)*(regressinf.slope+regressinf.intercept)-(regressinf.intercept**2)/regressinf.slope)

lamb_vals = -lamb_vals

# Plot of Free energy Surfaces
fig, ax = plt.subplots(figsize=(8,6))
axins2 = inset_axes(ax, width=1.7, height=1.4,bbox_to_anchor=(1,0.9),bbox_transform=ax.transAxes, loc='upper right')
ax.scatter(phi0[::5],bF_im0[::5],s=175,facecolors='none',edgecolors='royalblue',linewidths=3)#,zorder=8)
ax.scatter(phi0[::5],bF_im0s[::5],s=175,facecolors='none',edgecolors='royalblue',linewidths=3)#,zorder=8)
ax.scatter(phi2[::5],bF_im2[::5],s=175,facecolors='none',edgecolors='green',linewidths=3)#,zorder=8)
ax.scatter(phi2[::5],bF_im2s[::5],s=175,facecolors='none',edgecolors='green',linewidths=3)#,zorder=8)
ax.scatter(phi5[::5],bF_im5[::5],s=175,facecolors='none',edgecolors='firebrick',linewidths=3)#,zorder=8)
ax.scatter(phi5[::5],bF_im5s[::5],s=175,facecolors='none',edgecolors='firebrick',linewidths=3)#,zorder=8)
ax.scatter(phi_inf[::5],bF_iminf[::5],s=175,facecolors='none',edgecolors='k',linewidths=3)#,zorder=8)
ax.scatter(phi_inf[::5],bF_imsinf[::5],s=175,facecolors='none',edgecolors='k',linewidths=3)#,zorder=8)
ax.plot(phi0,betaF_phi0,c='royalblue',label=r'$\ell_{TF} = 0 \AA$')
ax.plot(phi0o,betaF_phi0o,c='royalblue')#,label=r'l_{TF} = 0 \AA')
ax.plot(phi2,betaF_phi2,c='green',label=r'$\ell_{TF} = 2 \AA$')
ax.plot(phi2o,betaF_phi2o,c='green')#,label=r'l_{TF} = 5 \AA')
ax.plot(phi5,betaF_phi5,c='firebrick',label=r'$\ell_{TF} = 5 \AA$')
ax.plot(phi5o,betaF_phi5o,c='firebrick')#,label=r'l_{TF} = 5 \AA')
ax.plot(phi_inf,betaF_phi_inf,c='k',label=r'$\ell_{TF} \to \infty$')
ax.plot(phio_inf,betaF_phio_inf,c='k')#,label=r'l_{TF} = 5 \AA')
ax.legend(loc='upper left')
ax.set_xlabel(r'$\beta \Delta E$')
ax.set_ylabel(r'$-\ln p(\beta \Delta E)$')
ax.set_xlim(-270,290)
ax.set_ylim(-5,60)
axins2.plot(lamb_vals,means0,c='royalblue',label=r'$l_{TF} = 0 \AA$')
axins2.scatter(lamb_vals,means0,s=150,facecolors='none',edgecolors='royalblue',linewidths=2.5,zorder=20)
axins2.plot(lamb_vals,means2,c='green',label=r'$l_{TF} = 2 \AA$')
axins2.scatter(lamb_vals,means2,s=150,facecolors='none',edgecolors='green',linewidths=2.5,zorder=20)
axins2.plot(lamb_vals,means5,c='firebrick',label=r'$l_{TF} = 5 \AA$')
axins2.scatter(lamb_vals,means5,s=150,facecolors='none',edgecolors='firebrick',linewidths=2.5,zorder=20)
axins2.plot(lamb_vals,meansinf,c='k',label=r'$l_{TF} \to \infty$')
axins2.scatter(lamb_vals,meansinf,s=150,facecolors='none',edgecolors='k',linewidths=2.5,zorder=20)
axins2.set_xlabel(r'$\eta$',fontsize=18)
axins2.set_ylabel(r'$\beta \langle \Delta E \rangle_{\eta}$',fontsize=18)
axins2.set_xlim(-0.1,1.1)
axins2.set_ylim(-220,220)
axins2.tick_params(axis='both', which='major', labelsize=16)
#axins2.xaxis.set_label_position('top')
#axins2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
fig.tight_layout()
plt.savefig('MarcusChlorideSI.png')
plt.show()
