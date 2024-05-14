import random
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel



# Run parameters: ===================================
N = 3E5 # Number of packets in simulation 1E5 default
type = 'linescattering' # 'electronscattering' or 'linescattering'
Nlinesmax=2
#rin = 0.5 # rout is at 1 : where is photosphere? (value 0 - 1)
T = 5000 # in K
v_out = 3000 # in km/s
vlim = 1500 # For emission line: maximum emission velocity. For PCygni simulation: photospheric velocity.
# ==Constants========================================
me = 9.11E-28
mp = 1.67E-24
k = 1.38E-16
c = 3E5
# ==Output format====================================
save = 0 #save=0 --> plot on screen, 1 --> save to file.
# ===================================================

fig = plt.figure(figsize=(10,6))

if (type == 'electronscattering'):
    Nmodels = 2
    tau0vec = [0.,1.0]  # For continuous simulations (elecronscattering)
    Tvec = [0., 0.]
    peakval = np.zeros(shape=(Nmodels,1)) # Initialization
elif (type == 'linescattering'):
    Nmodels = 2 # Number of runs:
    tau0vec = np.zeros(shape=(Nmodels,Nlinesmax))
    tau0vec[0,0:1] = [1E4]
    tau0vec[1,0:1] = [1E4]
    #peakval = np.zeros(shape=(Nmodels,1)) # Initialization
    #peakind = np.zeros(shape=(Nmodels,1)) # Initialization
    #xpeak = np.zeros(shape=(Nmodels,1)) # Initialization
    wllines = np.zeros(shape=(Nmodels,Nlinesmax))
    wllines[0,0:1] = [1000.]
    wllines[1,0:1] = [1000.]
    #wllines[0,:] = np.arange(1003., 1050.,3.)
    destrprob = np.zeros(shape=(Nmodels,Nlinesmax)) # epsilon
    destrprob[0,0:1] = [0.] 
    destrprob[1,0:1] = [0.] 

lambda0 = 1000.
Nbin = 400

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
            wl = wlmin + (wlmax-wlmin)*random.random()
            mu = np.sqrt(random.random())
            r = rin + 1E-5
        elif (type == 'electronscattering'):
            wl = lambda0
            mu = 1. - 2*random.random()
            r = rin*random.random()**(1./3.)

        sin_theta = np.sqrt(1. - mu**2)

        while (escaped == False and destroyed == False): # While packet is alive..

            # Distance to outer shell edge:
            distouter = -r*mu + np.sqrt((r*mu)**2 - (r**2 - 1**2))

            # Distance to inner shell edge:
            if ((type == 'linescattering') and (mu < 0 and r*sin_theta < rin)):
                distinner = -r*mu - np.sqrt((r*mu)**2 - (r**2 - rin**2))
            else:
                distinner = 2.

            # Draw random "lifetime" (=optical depth to absorption even) for this packet
            z = random.random()

            tau_thisphoton = -np.log(1-z) # TEMP righ log?

            distscatt = 2. # Initialize to a value > 1.

            if (type == 'electronscattering'):

                #print "tau", tau_thisphoton

                distscatt = tau_thisphoton/tau0

                #print("distscatt",distscatt)

            elif (type == 'linescattering'):

                # Find indeces of lines redward of current (co-moving) wavelength
                ind = (wllines[j,:] > wl)

                #print("ind",ind, wllines[j,:])

           
                if (any(ind)): # There is at least one more line remaining

                    #print("apa", wllines[j,ind][0])

                    distscatt = (wllines[j,ind][0]-wl)/wl*c/v_out # Distance to the closest line..
                    #print("distscatt", distscatt)

                else: # There is no redward line remaining

                    distscatt = 10.

            sin_theta = np.sqrt(1.-mu**2)

            #print "In", r, mu, distouter, wl

            if (distscatt < distouter and distscatt < distinner): # If distance to interaction is smaller than to both shell edges..

                dr = distscatt
                
                r_nextpos = np.sqrt(r**2 + dr**2 - 2*r*dr*(-mu)) # Law of cosines for next radius..

                #print("r", r, r_nextpos, mu)

                gamma = 1./np.sqrt(1.-(dr*v_out/c)**2) # Gamma factor..

                wl = wl*(1 + v_out/c*dr)*gamma    # New CMF wavelength (Rybicki Lightman eq 4.11 with cos_theta = -1 (homologous flow)).

                #print("New CMF wl",wl)
                
                # Thermal reshuffling
                z = random.random()

                if (type == 'electronscattering'):
                    vthermal_use = np.sqrt(k*Tvec[j]/me)/1e5 # km/s
                elif (type == 'linescattering'):
                    vthermal_use = np.sqrt(k*T/mp)/1e5 # km/s

                dwl_thermal = wl*vthermal_use/c*(1.-2*z) # TEMP

                wl = wl + dwl_thermal

                #print("After thermal shuffling:", wl)

                if (type == 'electronscattering'):

                    mu = 1 - 2*random.random()  # Isotropic scattering
	
               	    sin_theta = np.sqrt(1.-mu**2) # Update sin(theta).. #TEMP needs to happen also for ES

                    scattered = True

                else: # TEMP rewrite this so that tau_accum >= tau_this has been checked already..

                    x = random.random()

                    if (x < 1. - np.exp(-tau0vec[j,ind][0])): # Interaction happens..

                        mu = 1. - 2*random.random()  # Isotropic scattering --> random new mu

                        sin_theta = np.sqrt(1.-mu**2) # Update sin(theta).. #TEMP needs to happen also for ES

                        x = random.random()

                        if (x < destrprob[j,ind][0]): # Random draw if photon is thermalized..

                            destroyed = True # TEMP
                    
                            #print "scattered", wl, r_nextpos

                        else:

                            scattered = True

                    else: # Interaction does not happen..

                        sin_theta = r/r_nextpos*sin_theta

                        if (mu < 0 and dr < abs(r*mu)):
                    
                            mu = -np.sqrt(1.-sin_theta**2)

                        else:
                    
                            mu = np.sqrt(1.-sin_theta**2)

                            # print "noscatter", wl, r_nextpos
                            
                r = r_nextpos                    

            elif (distouter < distinner): # Next event is hitting outer shell edge..

                dr = distouter

                # print "distedge", distouter, distscatt, wl

                # Comoving frame properties at edge

                r_nextpos = 1.

                sin_theta = r/r_nextpos*sin_theta

                mu = np.sqrt(1.-sin_theta**2) # mu positive reaching surface..

                gamma = 1./np.sqrt(1.-(dr*v_out/c)**2)

                wl =  wl*(1 + v_out/c*dr)*gamma # Transformation to new CMF..

                # In observer frame

                gamma = 1./np.sqrt(1.-(v_out/c)**2)  

                wl_escaped = wl*(1. - v_out/c*mu)*gamma  # Rybicki Lightman eq 4.11

                #print "escaped", wl, mu, wl_escaped
                
                # Bin it!

                new = (binarr - wl_escaped)**2

                #print new
        
                ind = new.argmin()

                #print ind

                #print "Out", r_nextpos, mu, wl, wl_escaped, ind

                #print ind

                hist[ind] = hist[ind] + 1 # Add this packet to number of escaped in this (observer frame) wavelength bin..

                if (scattered):

                    histscatt[ind] = histscatt[ind] + 1

                escaped = True

            else:  # TEMP, what is this?

                destroyed = True

# ========= PLOTTING ===========================

    ynorm = 0.077

    gauss_kernel = Gaussian1DKernel(2)
    hist = convolve(hist, gauss_kernel) #
    histscatt = convolve(histscatt, gauss_kernel) #

    if (j <= 2):

        plt.plot((binarr-1000)/1000/(v_out/c),hist/N/ynorm*Nbin/50/0.26,color=colorvec[j],label=str(j),linewidth=2) # TEMP

        plt.plot((binarr-1000)/1000/(v_out/c),histscatt/N/ynorm*Nbin/50/0.26,'--',color=colorvec[j],label='scattered '+str(j),linewidth=1)

        plt.plot((binarr-1000)/1000/(v_out/c),(hist-histscatt)/N/ynorm*Nbin/50/0.26,':',color=colorvec[j],label='unscattered '+str(j),linewidth=1)

        #fid=open('banan'+str(j),'w')
        #x=(binarr-1000)/1000/(v_out/c)
        #y1=hist/N/ynorm*Nbin/50/0.26
        #y2=histscatt/N/ynorm*Nbin/50/0.26
        #np.savetxt(fid, np.c_[x,y1,y2])
        #fid.close()

    elif (j == 3):

        plt.plot((binarr-1000)/1000/(v_out/c),hist/N/ynorm*Nbin/50/2,'--',color=colorvec[j-3])

    elif (j == 4):

        plt.plot((binarr-1000)/1000/(v_out/c),hist/N/ynorm*Nbin/50/2,':',color=colorvec[j-3])

    elif (j == 5):

        plt.plot((binarr-1000)/1000/(v_out/c),hist/N/ynorm*Nbin/50/2,'-.',color=colorvec[j-3])
        

    #peakval[j] = max(hist[0:int(Nbin/2)]/N/ynorm*Nbin/50)
    #peakind[j] = np.argmax(hist[0:int(Nbin/2)]/N/ynorm*Nbin/50)
    #xpeak[j] =  (binarr[peakind[j]]-1000.)/1000./(v_out/c)
    #print(j, peakval[j], peakind[j], (binarr[peakind[j]]-1000.)/1000./(v_out/c))

    #plt.plot((binarr-1000)/1000/(v_out/c), 0.06*(binarr[2]-binarr[1])/(40/50.)*(1-((binarr-1000)/10)**2))

plt.ylim([0,2])
plt.plot([-1.5,1.5],[1,1],'k--')
plt.xlim([-1.5,1.5])
plt.legend()


#if (type == 'electronscattering'):
#    plt.plot([0.48,0.55],[0.06/ynorm,0.065/ynorm],linewidth=5,'k')
#    plt.text(0.56,0.066/ynorm,r'$\tau=0$')
#    plt.plot([0.64,0.75],[0.035/ynorm,0.045/ynorm],'k')
#    plt.text(0.8,0.046/ynorm,r'$\tau=1$')
#    plt.plot([0.71,0.78],[0.028/ynorm,0.035/ynorm],'k')
#    plt.text(0.82,0.036/ynorm,r'$\tau=2$')
#    plt.plot([0.73,0.8],[0.024/ynorm,0.030/ynorm],'k')
#    plt.text(0.8,0.031/ynorm,r'$\tau=3$')
#else:
    #plt.text(0.01,0.95,'Optically thin')
    #plt.text(0.9,0.8,'Optically thick,')
    #plt.text(0.9,0.75,'scatters')
    #plt.text(0.01,0.6,'Optically thick,')
    #plt.text(0.01,0.55,'destroyed')
#plt.plot([xpeak[1],xpeak[1]],[peakval[1],peakval[1]+0.02],'k')
#plt.plot([xpeak[2],xpeak[2]],[peakval[2],peakval[2]+0.02],'k')
#plt.plot([xpeak[3],xpeak[3]],[peakval[3],peakval[3]+0.02],'k')
#if (type == 'electronscattering'):
#    plt.plot([-0.13,-0.13],[peakval[1],peakval[1]+0.02],'k')
##    plt.plot([-0.20,-0.20],[peakval[2],peakval[2]+0.02],'k')
#    plt.plot([-0.27,-0.27],[peakval[3],peakval[3]+0.02],'k')

#    plt.text(xpeak[1]-0.1,peakval[1]+0.03,'-0.13')
#    plt.text(xpeak[2]-0.1,peakval[2]+0.03,'-0.20')
#    plt.text(xpeak[3]-0.1,peakval[3]+0.03,'-0.27')



plt.plot([0,0],[0,1],'k--')
plt.plot([-1,-1],[0,1],'k--')

plt.xlabel('Wavelength ($\Delta \lambda/\lambda_0/(V_{max}/c)$)')
plt.ylabel('Flux')

#plt.yscale('log')

if (save == 0):
    plt.show()
else:
    if (type == 'electronscattering'):
        plt.savefig("electronscattering.eps",bbox_inches='tight')
    else:
        plt.xlim([-1,1.5])
        plt.savefig("linescattering.eps",bbox_inches='tight')
        
 
