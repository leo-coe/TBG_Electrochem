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
phi0, betaF_phi0 = read_data('ferrous_ferric/lTF_0/FES/forward.txt',0,1,5/3,0)
phi0o, betaF_phi0o = read_data('ferrous_ferric/lTF_0/FES/backward.txt',0,1,5/3,0)
phi2, betaF_phi2 = read_data('ferrous_ferric/lTF_2/FES/forward.txt',0,1,5/3,0)
phi2o, betaF_phi2o = read_data('ferrous_ferric/lTF_2/FES/backward.txt',0,1,5/3,0)
phi5, betaF_phi5 = read_data('ferrous_ferric/lTF_5/FES/forward.txt',0,1,5/3,0)
phi5o, betaF_phi5o = read_data('ferrous_ferric/lTF_5/FES/backward.txt',0,1,5/3,0)
phi_inf, betaF_phi_inf = read_data('ferrous_ferric/lTF_inf/FES/forward.txt',0,1,5/3,0)
phio_inf, betaF_phio_inf = read_data('ferrous_ferric/lTF_inf/FES/backward.txt',0,1,5/3,0)

# Horizontal Shift
phi2-=752.4837
phi2o-=752.4837
phi5-=750.63886
phi5o-=750.63886
phi0-=743.5907
phi0o-=743.5907
phi_inf-=749.8606
phio_inf-=749.8606

# Linear dependence relation
bF_im2 = betaF_phi2 - phi2
bF_im0 = betaF_phi0 - phi0
bF_im2s = betaF_phi2o + phi2o 
bF_im0s = betaF_phi0o + phi0o 
bF_iminf = betaF_phi_inf - phi_inf
bF_im5 = betaF_phi5 - phi5
bF_imsinf = betaF_phio_inf + phio_inf 
bF_im5s = betaF_phi5o + phi5o 

# Biasing Parameter
lamb_vals = np.array([-0.2,0,0.2,0.4,0.5,0.6,0.8,1.0,1.2])

# Statistics from raw energy gap data
phim12,meanm12 = read_datmean('ferrous_ferric/lTF_2/Egap_data/lambdam1/sample.txt',0,1,5/3)
phi02,mean02 = read_datmean('ferrous_ferric/lTF_2/Egap_data/lambda0/sample.txt',0,1,5/3)
phi12,mean12 = read_datmean('ferrous_ferric/lTF_2/Egap_data/lambda1/sample.txt',0,1,5/3)
phi22,mean22 = read_datmean('ferrous_ferric/lTF_2/Egap_data/lambda2/sample.txt',0,1,5/3)
phi32,mean32 = read_datmean('ferrous_ferric/lTF_2/Egap_data/lambda3/sample.txt',0,1,5/3)
phi42,mean42 = read_datmean('ferrous_ferric/lTF_2/Egap_data/lambda4/sample.txt',0,1,5/3)
phi52,mean52 = read_datmean('ferrous_ferric/lTF_2/Egap_data/lambda5/sample.txt',0,1,5/3)
phiTS2,meanTS2 = read_datmean('ferrous_ferric/lTF_2/Egap_data/lambdaTS/sample.txt',0,1,5/3)
phip12,meanp12 = read_datmean('ferrous_ferric/lTF_2/Egap_data/lambdap1/sample.txt',0,1,5/3)

phim1inf,meanm1inf = read_datmean('ferrous_ferric/lTF_inf/Egap_data/lambdam1/sample.txt',0,1,5/3)
phi0inf,mean0inf = read_datmean('ferrous_ferric/lTF_inf/Egap_data/lambda0/sample.txt',0,1,5/3)
phi1inf,mean1inf = read_datmean('ferrous_ferric/lTF_inf/Egap_data/lambda1/sample.txt',0,1,5/3)
phi2inf,mean2inf = read_datmean('ferrous_ferric/lTF_inf/Egap_data/lambda2/sample.txt',0,1,5/3)
phi3inf,mean3inf = read_datmean('ferrous_ferric/lTF_inf/Egap_data/lambda3/sample.txt',0,1,5/3)
phi4inf,mean4inf = read_datmean('ferrous_ferric/lTF_inf/Egap_data/lambda4/sample.txt',0,1,5/3)
phi5inf,mean5inf = read_datmean('ferrous_ferric/lTF_inf/Egap_data/lambda5/sample.txt',0,1,5/3)
phiTSinf,meanTSinf = read_datmean('ferrous_ferric/lTF_inf/Egap_data/lambdaTS/sample.txt',0,1,5/3)
phip1inf,meanp1inf = read_datmean('ferrous_ferric/lTF_inf/Egap_data/lambdap1/sample.txt',0,1,5/3)

phim10,meanm10 = read_datmean('ferrous_ferric/lTF_0/Egap_data/lambdam1/sample.txt',0,1,5/3)
phi00,mean00 = read_datmean('ferrous_ferric/lTF_0/Egap_data/lambda0/sample.txt',0,1,5/3)
phi10,mean10 = read_datmean('ferrous_ferric/lTF_0/Egap_data/lambda1/sample.txt',0,1,5/3)
phi20,mean20 = read_datmean('ferrous_ferric/lTF_0/Egap_data/lambda2/sample.txt',0,1,5/3)
phi30,mean30 = read_datmean('ferrous_ferric/lTF_0/Egap_data/lambda3/sample.txt',0,1,5/3)
phi40,mean40 = read_datmean('ferrous_ferric/lTF_0/Egap_data/lambda4/sample.txt',0,1,5/3)
phi50,mean50 = read_datmean('ferrous_ferric/lTF_0/Egap_data/lambda5/sample.txt',0,1,5/3)
phiTS0,meanTS0 = read_datmean('ferrous_ferric/lTF_0/Egap_data/lambdaTS/sample.txt',0,1,5/3)
phip10,meanp10 = read_datmean('ferrous_ferric/lTF_0/Egap_data/lambdap1/sample.txt',0,1,5/3)

phim15,meanm15 = read_datmean('ferrous_ferric/lTF_5/Egap_data/lambdam1/sample.txt',0,1,5/3)
phi05,mean05 = read_datmean('ferrous_ferric/lTF_5/Egap_data/lambda0/sample.txt',0,1,5/3)
phi15,mean15 = read_datmean('ferrous_ferric/lTF_5/Egap_data/lambda1/sample.txt',0,1,5/3)
phi25,mean25 = read_datmean('ferrous_ferric/lTF_5/Egap_data/lambda2/sample.txt',0,1,5/3)
phi35,mean35 = read_datmean('ferrous_ferric/lTF_5/Egap_data/lambda3/sample.txt',0,1,5/3)
phi45,mean45 = read_datmean('ferrous_ferric/lTF_5/Egap_data/lambda4/sample.txt',0,1,5/3)
phi55,mean55 = read_datmean('ferrous_ferric/lTF_5/Egap_data/lambda5/sample.txt',0,1,5/3)
phiTS5,meanTS5 = read_datmean('ferrous_ferric/lTF_5/Egap_data/lambdaTS/sample.txt',0,1,5/3)
phip15,meanp15 = read_datmean('ferrous_ferric/lTF_5/Egap_data/lambdap1/sample.txt',0,1,5/3)

# Average energy gap in biased ensembles
means2 = np.array([meanm12,mean02,mean12,mean22,meanTS2,mean32,mean42,mean52,meanp12])- 752.4837
means0 = np.array([meanm10,mean00,mean10,mean20,meanTS0,mean30,mean40,mean50,meanp10]) - 743.5907
means5 = np.array([meanm15,mean05,mean15,mean25,meanTS5,mean35,mean45,mean55,meanp15])-750.63886
meansinf = np.array([meanm1inf,mean0inf,mean1inf,mean2inf,meanTSinf,mean3inf,mean4inf,mean5inf,meanp1inf])-749.8606 

# Linear regression of gap as a function of biasing parameter
regress0 = linregress(lamb_vals[1:7],means0[1:7])
regress2 = linregress(lamb_vals[1:7],means2[1:7])
regress5 = linregress(lamb_vals[1:7],means5[1:7])
regressinf = linregress(lamb_vals[1:7],meansinf[1:7])
dr_0 = 0.5*((1+regress0.intercept/regress0.slope)*(regress0.slope+regress0.intercept)-(regress0.intercept**2)/regress0.slope)
dr2 = 0.5*((1+regress2.intercept/regress2.slope)*(regress2.slope+regress2.intercept)-(regress2.intercept**2)/regress2.slope)
dr_5 = 0.5*((1+regress5.intercept/regress5.slope)*(regress5.slope+regress5.intercept)-(regress5.intercept**2)/regress5.slope)
dr_inf = 0.5*((1+regressinf.intercept/regressinf.slope)*(regressinf.slope+regressinf.intercept)-(regressinf.intercept**2)/regressinf.slope)

# Plot of Free energy Surfaces
fig, ax = plt.subplots(figsize=(8.5,6.14))
axins2 = inset_axes(ax, width=2.25, height=1.8,bbox_to_anchor=(1.07,0.93),bbox_transform=ax.transAxes, loc='upper right', borderpad=0.8)
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
ax.set_ylabel(r'$-ln p(\beta \Delta E)$')
ax.set_xlim(-215,250)
ax.set_ylim(-2,60)
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
axins2.set_xlim(-0.08,1.08)
axins2.set_ylim(-140,140)
axins2.tick_params(axis='both', which='major', labelsize=16)
axins2.xaxis.set_label_position('top')
axins2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
fig.tight_layout()
plt.show()

# Read in the data for the image potential in an empty capacitor
U=[]
with open('empty_lTFsweep.txt','r') as f:
	lines=f.readlines()
	for line in lines:	
		U.append(float(line.split()[5]))
dE = np.array(U)
# Compute image charge scaling function
xi = ((dE-dE[-1])*0.0016)*4*4.5593/0.529

# Array of screening lengths evaluated in the empty capacitor
lTF = np.arange(0,20.2,0.2)
lTF = list(lTF)
for i in range(21,51):
	lTF.append(i)
for i in range(6,51):
	lTF.append(10*i)
lTF=np.array(lTF)

# Fit the data of the image potential to function (not used for this plot)
#def fit(x, a, b, c, f):
#	return (a/(b+c*(x-f)**2))
#popt, pcov = curve_fit(fit, lTF, xi)

eps_sol = 1
eps_el = 1
l_TF = np.linspace(0,5,100)/0.529
z = np.linspace(0,25,100)/0.529

l_sim = [0,2,5]

# Values of Reorganization Energy, from simulation data
reorg_Fe = [86.21838253,92.23673654,99.91190562]
Fe_err = 0.5*np.array([7.999500106,3.171494499,2.757120518])
reorgFe_inf=113.271101
Feinf_err=0.5*3.004425952
reorg_K=[102.5078914,108.52377,116.8172928]
K_err= 0.5*np.array([4.900441725,4.676012689,4.847706602])
reorgK_inf=130.9060845
Kinf_err =0.5*4.622906403 
reorg_Cl=[115.6148713,121.5992905,129.7321891]
Cl_err=0.5*np.array([9.975208105,9.61577807,8.959142437])
reorgCl_inf=143.4173526
Clinf_err=0.5*9.002323229


ellTF = np.linspace(0,500,10000)
eps_infty = 1
eps_stat = 70
eps_elec = 1

# Code to evaluate the continuum integrals (not needed here, because they are pre-computed)
#xi_infty=np.zeros(len(ellTF))
#xi_stat=np.zeros(len(ellTF))
#xi_corr=np.zeros(len(lTF))
#for i in range(len(ellTF)):
#	xi_infty[i] = I_1(z0,ellTF[i],eps_infty,eps_elec)-eps_infty*I_2(z0,ellTF[i],eps_infty,eps_elec)-1
#	xi_stat[i] = I_1(z0,ellTF[i],eps_stat,eps_elec)-eps_stat*I_2(z0,ellTF[i],eps_stat,eps_elec)-1
#for i in range(len(lTF)):
#	xi_corr[i] = I_1(z0,lTF[i],eps_stat,eps_elec)-eps_stat*I_2(z0,lTF[i],eps_stat,eps_elec)-1
#continuum = ((1/(4*(z0+jellium)/0.529))*(((xi_infty+1)/eps_infty)-((xi_stat+1)/eps_stat)))/0.00095
#continuum_corr = ((1/(4*(z0+jellium)/0.529))*(((xi_corr+1)/eps_stat)))/0.00095
#continuuminf = ((1/(4*(z0+jellium)/0.529))*((xi_infty+1)/eps_infty))/0.00095

# Prediction of reoganization energy from continuum dielectric theory
continuum = np.zeros(len(ellTF))
continuum_corr = np.zeros(len(lTF))
continuuminf = np.zeros(len(ellTF))

with open('continuum.txt', 'r') as f:
	lines=f.readlines()
	for i in range(len(lines)):
		continuum[i]=float(lines[i].split()[0])
with open('continuum_corr.txt', 'r') as f:
	lines=f.readlines()
	for i in range(len(lines)):
		continuum_corr[i]=float(lines[i].split()[0])
with open('continuuminf.txt', 'r') as f:
	lines=f.readlines()
	for i in range(len(lines)):
		continuuminf[i]=float(lines[i].split()[0])

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(ellTF,continuum,'k',zorder=1)
ax.plot(lTF,(((xi-xi[0])/(4*4.5593/0.529))/0.00095-continuum_corr),'k',zorder=1)
ax.scatter(ellTF[::25],continuum[::25],s=200,facecolors='w',edgecolors='k',marker='^',linewidths=3,label='Dielectric Theory',zorder=2)
ax.scatter(lTF[::5],((xi[::5]-xi[0])/(4*4.5593/0.529))/0.00095-continuum_corr[::5],s=180,facecolors='w',edgecolors='k',marker='s',linewidths=3,label='Simulation',zorder=2)
ax.errorbar(l_sim,(np.array(reorg_K)-reorg_K[0]),yerr=K_err,capsize=10,fmt='o',c='firebrick',label=r'$K^+/K^0$',zorder=100)
ax.scatter(l_sim,(np.array(reorg_K)-reorg_K[0]),s=200,facecolors='none',edgecolors='firebrick',linewidths=3,zorder=100)#,zorder=8)
ax.errorbar(l_sim,(np.array(reorg_Cl)-reorg_Cl[0]),yerr=Cl_err,capsize=10,fmt='o',c='green',label=r'$Cl^0/Cl^-$',zorder=100)
ax.scatter(l_sim,(np.array(reorg_Cl)-reorg_Cl[0]),s=200,facecolors='none',edgecolors='green',linewidths=3,zorder=100)#,zorder=8)
ax.errorbar(l_sim,(np.array(reorg_Fe)-reorg_Fe[0]),yerr=Fe_err,capsize=10,fmt='o',c='royalblue',label=r'$Fe^{2+}/Fe^{3+}$',zorder=100)
ax.scatter(l_sim,(np.array(reorg_Fe)-reorg_Fe[0]),s=200,facecolors='none',edgecolors='royalblue',linewidths=3,zorder=100)#,zorder=8)
ax.plot(np.linspace(-1,500,2),np.ones(2)*(reorgK_inf-reorg_K[0]),'--',c='firebrick',linewidth=3)#,label=r'$l_{TF} \to \infty$, Simulation')
ax.plot(np.linspace(-1,500,10),np.ones(10)*(reorgFe_inf-reorg_Fe[0]),'--',c='royalblue',linewidth=3)#,label=r'$l_{TF} \to \infty$, Simulation')
ax.plot(np.linspace(-1,500,2),np.ones(2)*(reorgCl_inf-reorg_Cl[0]),'--',c='green',linewidth=3)#,label=r'$l_{TF} \to \infty$, Simulation')
ax.fill_between(np.linspace(-1,500,10),np.ones(10)*(reorgCl_inf-reorg_Cl[0])-Clinf_err,np.ones(10)*(reorgCl_inf-reorg_Cl[0])+Clinf_err,color='grey',linewidth=3,alpha=0.2)#,label=r'$l_{TF} \to \infty$, Simulation')
ax.legend(loc='lower right')
ax.set_xlabel(r'$\ell_{\mathrm{TF}} / \mathrm{\AA}$')
ax.set_ylabel(r'$\beta \lambda(\ell_{\mathrm{TF}}) - \beta \lambda(0)$')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.tight_layout()
plt.show()



fig = plt.figure(figsize=(7.7,10))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[0, 0])
axins2 = inset_axes(ax2, width=1.7, height=1.4,bbox_to_anchor=(0.38,0.73),bbox_transform=ax.transAxes, loc='upper right')
ax2.scatter(phi0[::5],bF_im0[::5],s=125,facecolors='none',edgecolors='royalblue',linewidths=3)#,zorder=8)
ax2.scatter(phi0[::5],bF_im0s[::5],s=125,facecolors='none',edgecolors='royalblue',linewidths=3)#,zorder=8)
ax2.scatter(phi2[::5],bF_im2[::5],s=125,facecolors='none',edgecolors='green',linewidths=3)#,zorder=8)
ax2.scatter(phi2[::5],bF_im2s[::5],s=125,facecolors='none',edgecolors='green',linewidths=3)#,zorder=8)
ax2.scatter(phi5[::5],bF_im5[::5],s=125,facecolors='none',edgecolors='firebrick',linewidths=3)#,zorder=8)
ax2.scatter(phi5[::5],bF_im5s[::5],s=125,facecolors='none',edgecolors='firebrick',linewidths=3)#,zorder=8)
ax2.scatter(phi_inf[::5],bF_iminf[::5],s=125,facecolors='none',edgecolors='k',linewidths=3)#,zorder=8)
ax2.scatter(phi_inf[::5],bF_imsinf[::5],s=125,facecolors='none',edgecolors='k',linewidths=3)#,zorder=8)
ax2.plot(phi0,betaF_phi0,c='royalblue',label=r'$\ell_{\mathrm{TF}} = 0 \mathrm{\AA}$')
ax2.plot(phi0o,betaF_phi0o,c='royalblue')#,label=r'l_{TF} = 0 \AA')
ax2.plot(phi2,betaF_phi2,c='green',label=r'$\ell_{\mathrm{TF}} = 2 \mathrm{\AA}$')
ax2.plot(phi2o,betaF_phi2o,c='green')#,label=r'l_{TF} = 5 \AA')
ax2.plot(phi5,betaF_phi5,c='firebrick',label=r'$\ell_{\mathrm{TF}} = 5 \mathrm{\AA}$')
ax2.plot(phi5o,betaF_phi5o,c='firebrick')#,label=r'l_{TF} = 5 \AA')
ax2.plot(phi_inf,betaF_phi_inf,c='k',label=r'$\ell_{\mathrm{TF}} \to \infty$')
ax2.plot(phio_inf,betaF_phio_inf,c='k')#,label=r'l_{TF} = 5 \AA')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, columnspacing=0.8, frameon=False, fontsize=16)
ax2.set_xlabel(r'$\beta \Delta E$')
ax2.set_ylabel(r'$-\ln p(\beta \Delta E)$')
ax2.set_xlim(-215,250)
ax2.set_ylim(-2,59)
axins2.plot(lamb_vals,means0,c='royalblue',label=r'$l_{TF} = 0 \mathrm{\AA}$')
axins2.scatter(lamb_vals,means0,s=150,facecolors='none',edgecolors='royalblue',linewidths=2.5,zorder=20)
axins2.plot(lamb_vals,means2,c='green',label=r'$l_{TF} = 2 \mathrm{\AA}$')
axins2.scatter(lamb_vals,means2,s=150,facecolors='none',edgecolors='green',linewidths=2.5,zorder=20)
axins2.plot(lamb_vals,means5,c='firebrick',label=r'$l_{TF} = 5 \mathrm{\AA}$')
axins2.scatter(lamb_vals,means5,s=150,facecolors='none',edgecolors='firebrick',linewidths=2.5,zorder=20)
axins2.plot(lamb_vals,meansinf,c='k',label=r'$l_{TF} \to \infty$')
axins2.scatter(lamb_vals,meansinf,s=150,facecolors='none',edgecolors='k',linewidths=2.5,zorder=20)
axins2.set_xlabel(r'$\eta$',fontsize=18)
axins2.set_ylabel(r'$\beta \langle \Delta E \rangle_{\eta}$',fontsize=18)
axins2.set_xlim(-0.08,1.08)
axins2.set_ylim(-140,140)
axins2.tick_params(axis='both', which='major', labelsize=16)
axins2.xaxis.set_label_position('top')
axins2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax1.plot(ellTF,continuum,'k',zorder=1)
ax1.plot(lTF,(((xi-xi[0])/(4*4.5593/0.529))/0.00095-continuum_corr),'k',zorder=1)
ax1.scatter(ellTF[::25],continuum[::25],s=200,facecolors='w',edgecolors='k',marker='^',linewidths=3,label='Continuum Theory',zorder=2)
ax1.scatter(lTF[::5],((xi[::5]-xi[0])/(4*4.5593/0.529))/0.00095-continuum_corr[::5],s=180,facecolors='w',edgecolors='k',marker='s',linewidths=3,label='Atomistic Theory',zorder=2)
ax1.errorbar(l_sim,(np.array(reorg_K)-reorg_K[0]),yerr=K_err,capsize=10,fmt='o',c='firebrick',label=r'$\mathrm{K}^+/\mathrm{K}^0$',zorder=100)
ax1.scatter(l_sim,(np.array(reorg_K)-reorg_K[0]),s=200,facecolors='none',edgecolors='firebrick',linewidths=3,zorder=100)#,zorder=8)
ax1.errorbar(l_sim,(np.array(reorg_Cl)-reorg_Cl[0]),yerr=Cl_err,capsize=10,fmt='o',c='green',label=r'$\mathrm{Cl}^0/\mathrm{Cl}^-$',zorder=100)
ax1.scatter(l_sim,(np.array(reorg_Cl)-reorg_Cl[0]),s=200,facecolors='none',edgecolors='green',linewidths=3,zorder=100)#,zorder=8)
ax1.errorbar(l_sim,(np.array(reorg_Fe)-reorg_Fe[0]),yerr=Fe_err,capsize=10,fmt='o',c='royalblue',label=r'$\mathrm{Fe}^{2+}/\mathrm{Fe}^{3+}$',zorder=100)
ax1.scatter(l_sim,(np.array(reorg_Fe)-reorg_Fe[0]),s=200,facecolors='none',edgecolors='royalblue',linewidths=3,zorder=100)#,zorder=8)
ax1.plot(np.linspace(-1,500,2),np.ones(2)*(reorgK_inf-reorg_K[0]),'--',c='firebrick',linewidth=3)#,label=r'$l_{TF} \to \infty$, Simulation')
ax1.plot(np.linspace(-1,500,10),np.ones(10)*(reorgFe_inf-reorg_Fe[0]),'--',c='royalblue',linewidth=3)#,label=r'$l_{TF} \to \infty$, Simulation')
ax1.plot(np.linspace(-1,500,2),np.ones(2)*(reorgCl_inf-reorg_Cl[0]),'--',c='green',linewidth=3)#,label=r'$l_{TF} \to \infty$, Simulation')
ax1.fill_between(np.linspace(-1,500,10),np.ones(10)*(reorgCl_inf-reorg_Cl[0])-Clinf_err,np.ones(10)*(reorgCl_inf-reorg_Cl[0])+Clinf_err,color='grey',linewidth=3,alpha=0.2)#,label=r'$l_{TF} \to \infty$, Simulation')
ax1.legend(loc='lower right', frameon=False, fontsize=15)
ax1.set_xlabel(r'$\ell_{\mathrm{TF}} / \mathrm{\AA}$')
ax1.set_ylabel(r'$\beta \lambda(\ell_{\mathrm{TF}}) - \beta \lambda(0)$')
ax1.set_xlim(-0.7,20.7)
ax1.set_ylim(-5.93,32.7)
ax1.set_ylabel(r'$\beta \lambda(\ell_{\mathrm{TF}}) - \beta \lambda(0)$')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
fig.tight_layout()
fig.savefig('Fig_MarcusReorg.png',bbox_inches='tight')
plt.show()
