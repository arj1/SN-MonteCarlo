import random
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

# Run parameters: ===================================
N = 3E5 # Number of packets in MC simulation
type = 'linescattering' # 'electronscattering' or 'linescattering'
T = 5000 # in K
v_out = 3000 # in km/s
vlim = 1500 # For emission line: maximum emission velocity. For P-Cygni simulation: photospheric velocity.
lambda0 = 1000. 
Nbin = 400 # Number of bins for creating emergent spectrum.
# == Constants ========================================
me = 9.11E-28
mp = 1.67E-24
k = 1.38E-16
c = 3E5
# == Outputs ====================================
save = 0 #save=0 --> plot on screen, 1 --> save to file.
# ===================================================

fig = plt.figure(figsize=(10,6))

if (type == 'electronscattering'):
    Nmodels = 2 # Number of models to compute
    tau0vec = [0.,1.0]  # Radial optical depth to electron scattering.
    Tvec = [0., 0.] # Temperatures
    #peakval = np.zeros(shape=(Nmodels,1)) # Initialization
elif (type == 'linescattering'):
    Nmodels = 2 # Number of models to compute
    Nlinesmax=2 # Maximum number of lines in a model
    tau0vec = np.zeros(shape=(Nmodels,Nlinesmax)) # Initialize
    wllines = np.zeros(shape=(Nmodels,Nlinesmax)) # Initialize
    destrprob = np.zeros(shape=(Nmodels,Nlinesmax)) # Initialize
    tau0vec[0,0:1] = [1E4] # Line optical depths, model 1..
    wllines[0,0:1] = [1000.] # Line wavelengths, model 1..
    destrprob[0,0:1] = [0.] # Destruction probabilities model 1..
    tau0vec[1,0:1] = [1E4] # Line optical depths, model 2..
    wllines[1,0:1] = [1000.] # Line wavelengths, model 2..
    destrprob[1,0:1] = [0.] # Destruction probabilities, model 2..
    #peakval = np.zeros(sh ape=(Nmodels,1)) # Initialization
    #peakind = np.zeros(shape=(Nmodels,1)) # Initialization
    #xpeak = np.zeros(shape=(Nmodels,1)) # Initialization
    #wllines[0,:] = np.arange(1003., 1050.,3.)

if (type == 'linescattering'):
	wlmin = lambda0*(1-1.55*v_out/c)
	wlmax = lambda0*(1+2.5*v_out/c)
	rin = vlim/v_out
elif (type == 'electronscattering'):
	wlmin = lambda0*(1-1.55*v_out/c)
	wlmax = lambda0*(1+2.5*v_out/c)
	rin = vlim/v_out

binarr = np.linspace(wlmin,wlmax,Nbin)
hist = np.zeros(Nbin)
histscatt = np.zeros(Nbin)
usesize = 12
plt.rcParams.update({'font.size': usesize})
colorvec=['b','r','k']

for j in range(0,Nmodels): 

    if (type == 'electronscattering'):
        tau0 = tau0vec[j]    

    hist = np.zeros(Nbin)
    histscatt = np.zeros(Nbin)

    for i in range(0,int(N)): # A new MC packet..

        # Initialize state:
        scattered = False
        escaped = False
        destroyed = False

        # Starting wavelength of packet:
        if (type == 'linescattering'):
            wl = wlmin + (wlmax-wlmin)*random.random() # A flat continuum
            mu = np.sqrt(random.random())
            r = rin + 1E-5 # All emission from just outside the photosphere
        elif (type == 'electronscattering'):
            wl = lambda0 # A single line
            mu = 1. - 2*random.random()
            r = rin*random.random()**(1./3.) # Uniform sphere sampling

        sin_theta = np.sqrt(1. - mu**2)

        while (escaped == False and destroyed == False): # While packet is alive..

            # Compute distance to outer shell edge:
            distouter = -r*mu + np.sqrt((r*mu)**2 - (r**2 - 1**2))

            # Compute distance to inner shell edge:
            if ((type == 'linescattering') and (mu < 0 and r*sin_theta < rin)):
                distinner = -r*mu - np.sqrt((r*mu)**2 - (r**2 - rin**2))
            else:
                distinner = 2.

            # Draw random "lifetime" (=optical depth to absorption even) for this packet
            xi = random.random()

            tau_thisphoton = -np.log(1-xi) # Draw lifetime for this packet

            distscatt = 2. # Initialize to a value > 1.

            if (type == 'electronscattering'):

                distscatt = tau_thisphoton/tau0

            elif (type == 'linescattering'):

                # Find indeces of lines redward of current (co-moving) wavelength:
                ind = (wllines[j,:] > wl)
           
                if (any(ind)): # There is at least one more line remaining..

                    distscatt = (wllines[j,ind][0]-wl)/wl*c/v_out # Distance to the closest line..

                else: # There is no redward line remaining

                    distscatt = 10.

            sin_theta = np.sqrt(1.-mu**2)

            if (distscatt < distouter and distscatt < distinner): # If distance to interaction is smaller than to both shell edges..

                dr = distscatt
                
                r_nextpos = np.sqrt(r**2 + dr**2 - 2*r*dr*(-mu)) # Law of cosines for next radius..

                gamma = 1./np.sqrt(1.-(dr*v_out/c)**2) # Gamma factor..

                wl = wl*(1 + v_out/c*dr)*gamma    # New CMF wavelength (Rybicki Lightman eq 4.11 with cos_theta = -1 (homologous flow)).
                
                # Thermal reshuffling
                xi = random.random()

                if (type == 'electronscattering'):
                    vthermal_use = np.sqrt(k*Tvec[j]/me)/1e5 # km/s
                elif (type == 'linescattering'):
                    vthermal_use = np.sqrt(k*T/mp)/1e5 # km/s

                dwl_thermal = wl*vthermal_use/c*(1.-2*xi) # TEMP

                wl = wl + dwl_thermal

                #print("After thermal shuffling:", wl)

                if (type == 'electronscattering'):

                    mu = 1 - 2*random.random()  # Isotropic scattering
	
               	    sin_theta = np.sqrt(1.-mu**2) # Update sin(theta).. #TEMP needs to happen also for ES

                    scattered = True

                else: # TEMP rewrite this so that tau_accum >= tau_this has been checked already..

                    xi = random.random()

                    if (xi < 1. - np.exp(-tau0vec[j,ind][0])): # Interaction happens..

                        mu = 1. - 2*random.random()  # Isotropic scattering --> random new mu

                        sin_theta = np.sqrt(1.-mu**2) # Update sin(theta).. #TEMP needs to happen also for ES

                        xi = random.random()

                        if (xi < destrprob[j,ind][0]): # Random draw if photon is thermalized..

                            destroyed = True # TEMP                    

                        else:

                            scattered = True

                    else: # Interaction does not happen..

                        sin_theta = r/r_nextpos*sin_theta

                        if (mu < 0 and dr < abs(r*mu)):
                    
                            mu = -np.sqrt(1.-sin_theta**2)

                        else:
                    
                            mu = np.sqrt(1.-sin_theta**2)
                            
                r = r_nextpos                    

            elif (distouter < distinner): # Next event is hitting outer shell edge..

                dr = distouter

                # Comoving frame properties at edge

                r_nextpos = 1.

                sin_theta = r/r_nextpos*sin_theta

                mu = np.sqrt(1.-sin_theta**2) # mu positive reaching surface..

                gamma = 1./np.sqrt(1.-(dr*v_out/c)**2)

                wl =  wl*(1 + v_out/c*dr)*gamma # Transformation to new CMF..

                # In observer frame:

                gamma = 1./np.sqrt(1.-(v_out/c)**2)  

                wl_escaped = wl*(1. - v_out/c*mu)*gamma  # Rybicki Lightman eq 4.11

                # Bin it..

                new = (binarr - wl_escaped)**2
        
                ind = new.argmin()

                hist[ind] = hist[ind] + 1 # Add this packet to number of escaped in this (observer frame) wavelength bin..TEMP energy of packet?

                if (scattered):

                    histscatt[ind] = histscatt[ind] + 1

                escaped = True

            else:  # TEMP, what is this?

                destroyed = True

# ========= PLOTTING ===========================

    ynorm = 0.077

    # Smooth MC noise:
    gauss_kernel = Gaussian1DKernel(2)
    hist = convolve(hist, gauss_kernel) 
    histscatt = convolve(histscatt, gauss_kernel) 

    if (j <= 2):

        plt.plot((binarr-lambda0)/lambda0/(v_out/c),hist/N/ynorm*Nbin/50/0.26,color=colorvec[j],label=str(j),linewidth=2) # TEMP

        plt.plot((binarr-lambda0)/lambda0/(v_out/c),histscatt/N/ynorm*Nbin/50/0.26,'--',color=colorvec[j],label='scattered '+str(j),linewidth=1)

        plt.plot((binarr-lambda0)/lambda0/(v_out/c),(hist-histscatt)/N/ynorm*Nbin/50/0.26,':',color=colorvec[j],label='unscattered '+str(j),linewidth=1)

    elif (j == 3):

        plt.plot((binarr-lambda0)/lambda0/(v_out/c),hist/N/ynorm*Nbin/50/2,'--',color=colorvec[j-3])

    elif (j == 4):

        plt.plot((binarr-lambda0)/lambda0/(v_out/c),hist/N/ynorm*Nbin/50/2,':',color=colorvec[j-3])

    elif (j == 5):

        plt.plot((binarr-lambda0)/lambda0/(v_out/c),hist/N/ynorm*Nbin/50/2,'-.',color=colorvec[j-3])
        

    #peakval[j] = max(hist[0:int(Nbin/2)]/N/ynorm*Nbin/50)
    #peakind[j] = np.argmax(hist[0:int(Nbin/2)]/N/ynorm*Nbin/50)
    #xpeak[j] =  (binarr[peakind[j]]-1000.)/1000./(v_out/c)
    #print(j, peakval[j], peakind[j], (binarr[peakind[j]]-1000.)/1000./(v_out/c))
    #plt.plot((binarr-1000)/1000/(v_out/c), 0.06*(binarr[2]-binarr[1])/(40/50.)*(1-((binarr-1000)/10)**2))

plt.ylim([0,2])
plt.xlim([-1.5,1.5])

plt.legend()

plt.plot([-1.5,1.5],[1,1],'k--')
plt.plot([0,0],[0,1],'k--')
plt.plot([-1,-1],[0,1],'k--')

plt.xlabel('Wavelength ($\Delta \lambda/\lambda_0/(V_{max}/c)$)')
plt.ylabel('Flux')

if (save == 0):
    plt.show()
else:
    if (type == 'electronscattering'):
        plt.savefig("electronscattering.eps",bbox_inches='tight')
    else:
        plt.savefig("linescattering.eps",bbox_inches='tight')
        
 
