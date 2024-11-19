import numpy as np
import scipy.io
import matplotlib.pyplot as plt 
import time


def initK(kBounds, ligand_conc, bin_edges):
#Initialize K to random variables
#k - [k_1, k_2, k_d, k_d*, k_syn]
#Input: kBounds (5 x 2)- bounds of k in log space
#Output: k (5 x 1 vector) 
#5 results from 5 reaction rates in system
    numK = np.shape(kBounds)[0]
    K = np.random.rand(5)
    while True:
        for i in range(numK):
            K[i] = np.random.uniform(kBounds[i, 0], kBounds[i, 1])
            #K[i] = kBounds[i, 1]
        abu = calcSteadyState(K, ligand_conc)
        if inBounds(K, kBounds, abu, bin_edges):
            return K, abu

def calcPsi(data, nbins, buffer = 0.2):
#Given a data vector and the number of bins, create a histogram in which
#each bin consists of 1/nbins of the total histogram (histogram will be flat)
#Buffer: denotes the abu buffer value for the lowest and highest bins
    #ie bin_edges[0] = np.min(data) - buffer

#Input: nbins (scalar, number of bins), data (nSamples x 1, abundances for each sample)
#Output: histogram (nbins x 1), bin_edges (nBins + 1 x 1, 
    nSamples = np.shape(data)[0]
    
    
    data = np.sort(data)
 
   
    #Initialize bin_edges and histogram
    bin_edge= np.zeros(nbins + 1)
    hist = np.random.rand(nbins)

    bin_edge[0] = np.min(data) - buffer *np.min(data) #First bin edge is the min of vector
    count = 0
    for i in range(nbins - 1):
        
        ind = int(nSamples/nbins * (i + 1))
        #bin_edge is the midpoint between the nth and (n + 1)th index 
        #n determined by nSamples and nbins
        hist[i] = (ind - count)/nSamples #fraction of samples between the two bin edges
        bin_edge[i + 1] = (data[ind] + data[ind- 1]) / 2

        count = ind
    hist[nbins - 1] = (nSamples - count)/ nSamples
    bin_edge[nbins] = np.max(data) * (1 + buffer) #Last bin edge is the max of the vector

    return bin_edge, hist

def getHistBins(data, nbins):
#Input: data (nExperiments x nCells) contains all data for all experiments
#nbins (scalar) number of bins
#Output: edges (nExperiments x numBins + 1), hist (nExperiments x nBins)
    hist = []
    edges = []

    for i in range(len(data)):
        histo = calcPsi(data[:][i], nbins)
        hist.append(histo[1])
        edges.append(histo[0])
    print('Data Successfully Histogrammed')
    hist = np.asarray(hist)
    edges = np.asarray(edges)
   
   
    
    return np.asarray(hist), np.asarray(edges)
def calcSteadyState(k, L):
#Calculates steady state abundances for given k vector and ligand concentration
#Input: k (5 x 1 vector) (log space!), L (ligand concentration, numExperiments x 1)
#Ouput: R+B_ss (numExperiments x 1) steady state solutions of receptors (R) and bounded 
#receptors (B)
    
    #k - [k_1, k_2, k_d, k_d*, k_syn]
    #Differential Eqtns:
    #dR/dt = (-k_1 * L * - k_d) * R + (k_2) * B + k_syn
    #dB/dt = (-k_1 * L) * R  + (-k_2 - k_d*) * B
  
    #Assign variable names to each k entry: 

  
    k_1 = k[0] 
    k_2 = k[1]
    k_d = k[2] 
    k_dstar = k[3]
    k_syn = k[4]
    
    #Convert from log scale to base 10
    k_1 = 10**k_1
    k_2 = 10**k_2 
    k_d = 10**k_d
    k_dstar = 10**k_dstar
    k_syn = 10**k_syn   
    #Create matrices of system and solve w linalg
    RB_ss = np.random.rand(np.shape(L)[0])
    
    count = 0
    # k_1 = 10**-2.3
    # k_2 = 10**-0.5
    # k_d = 10**-3.25
    # k_dstar = 10**-3.25
    # k_syn = 10**-1.2

    for conc in L:
        
        #iterate through each ligand concentration (conc) in L
        a = np.asarray([[-k_1 * conc - k_d, +k_2], [k_1 * conc, -k_2 -k_dstar]])
        b = np.asarray([-k_syn, 0])
        RB = np.linalg.solve(a, b)
   
        RB_ss[count] = np.sum(RB)

        count += 1

    return RB_ss


def inBounds(k, k_bounds, abu, bin_edges):
    #Return True if:
    #k is within k_bounds
    #all entries of abu are within bin_edges
    n_rates = np.shape(k)[0]
    n_conc = np.shape(abu)[0]
    k_in_bounds = np.all([k[i] > k_bounds[i, 0] and k[i] < k_bounds[i, -1] for i in range(n_rates)])
    abu_in_bounds = np.all([abu[i] > bin_edges[i, 0] and abu[i] < bin_edges[i, -1] for i in range(n_conc)])
    #Require that k_dstar (k[3]) > k_d (k[2])
    if k[3] < k[2]:
        return False
    return k_in_bounds and abu_in_bounds
def perturbK(k_old, k_bounds, ligand_conc, bin_edges, eps = 10, n_Perturb = 5):
    #eps: deltaK is random uniform(-length/eps, length/eps)
    #numPerturb: number of variables perturbed at a time
    #   max allowed value is n_rates defined below
    #Better version of perturbk
    #Returns: k (num_k_rates x 1), abu (num_conc x 1), indicator (n_bins x 1) 
    n_conc = np.shape(ligand_conc)[0]
    n_rates = np.shape(k_old)[0]
    length = np.asarray([k_bounds[i, 1] - k_bounds[i, 0] for i in range(n_rates)])
    #Get new K
    k_new = np.copy(k_old)
    while(True):
        delta = np.asarray([np.random.uniform(-length[i]/eps, length[i]/eps) for i in range(n_rates)])
        index = np.random.choice(n_rates, n_Perturb, replace = False)

        k_new[index] = k_old[index] + delta[index]
        #np.clip
        abu = calcSteadyState(k_new, ligand_conc)
     
        if inBounds(k_new, k_bounds, abu, bin_edges):
            # print(np.round(abu, 2))
            # print(np.round([bin_edges[:, 0], bin_edges[:, -1]], 2))
            # print(np.round(k_new, 2))
            
            return k_new, abu, delta, index
    exit()
    return 0
def acceptReject(abu_new, abu_old, lamb, bin_edges):
    # print("New abu: ", abu_new)
    # print("Old abu: ", abu_old)
    E_old, ind_old = calcEnergy(abu_old, lamb, bin_edges)
    E_new, ind_new = calcEnergy(abu_new, lamb, bin_edges)
    dE = E_new - E_old

    prob = np.min([1, np.exp(-dE)])
    
    rand = np.random.rand()
    TrueFalse = rand < prob
    # print(prob, np.exp(-dE))
    # print("dE: ", dE, ", ", TrueFalse)
    # exit()
    return TrueFalse, dE, ind_old, ind_new
    exit()
    return 0
def calcEnergy(abu, lamb, bin_edges):
#Input: abu (steady state abundance RB_ss) (nExperiments x 1), 
# lamb (ligand_concentrations x nbins), bin_edges (nExperiments x nBins + 1)
#Output: Energy for given abundance

    nExperiments = np.shape(abu)[0]
    
    nBins = np.shape(lamb)[1]

    indicator = []
    
    for n in range(nExperiments):
        index = np.digitize(abu[n], bin_edges[n]) - 1
        # if index < 0:
        #     index = 0
        # if index >= nBins:
        #     index -= 1
        if index == nBins:
            print("error. bin out of range")
            exit()
            index -= 1
        indicator.append(index)
    
    
    #E = np.sum(lamb[:, indicator])
    E = np.sum([lamb[i, indicator[i]] for i in range(len(indicator))])
    return E, indicator
def mcmc(lamb, bin_edges, ligand, k_bounds, numDraws = 5000, throwOut = 1000, saveiter = 50, EPS = 5, iter = 0):
    #For a given lambda and bin fractions, perform MCMC to return 
    #n samples drawn from the distribution
    #Input: lambda (nBins x nExperiments), bin_edges (nBins + 1 x 1)
    #k_bounds (5 x 2) upper and lower bounds of k
    #numDraws (scalar) number of draws from the distribution
    #throwOut (scalar) number of draws to throw out before assuming equilibrium
    #saveiter (scalar) every saveiter-th iteration, save the k draw value
    #Output: kDraws (5 x numDraws) numDraws number of k draws
    # 
    
    #list to hold k draws
    k = []
    k_old, abu_old = initK(k_bounds, ligand, bin_edges)

  
    count = 0
    A = 0
    R = 0   
    print('Entered MCMC')
    deltaE = []
    bigE = 0
  
    while np.shape(k)[0] < numDraws:
        numPerturb = np.random.randint(1, 6)
        k_new, abu_new, delta, ind = perturbK(k_old, k_bounds, ligand, bin_edges, eps = EPS, n_Perturb = numPerturb)
        
        accept, dE, iold, inew = acceptReject(abu_new, abu_old, lamb, bin_edges)
        
        deltaE.append(dE)
        if accept:
            count += 1

            A+= 1
            if count % saveiter == 0:
                #print("added")
                k.append(k_new)
               
            k_old = np.copy(k_new)
            abu_old = np.copy(abu_new)
        else:
            if count >= throwOut:
                R += 1
       
            #print("Reject")
    print("Accept/Reject: ", A/(A + R), R/(A + R))

    return np.asarray(k)

def initLambda(bins, experiments):
    #INput: bins (scalar, number of bins)
    #experiments (scalar, number of experiments)
    return np.random.rand(bins, experiments)
def dL(phi, psi):
#Input: phi (PREDICTED) (nBins x nExperiments) predicted bin fractions from mcmc
#psi (DATA) (nBins x nExperiments) bin fractions from data
#Output: dL defined as phi (predicted) - psi (data)
    delta = phi - psi
    #print("dL norm: ", np.linalg.norm(delta, ord = 2))
    return delta
def ADAM(dx, t, v, s, eta = 0.001, beta1= 0.9, beta2 = 0.999, eps = 10*-8):
    v = beta1 * v + (1-beta1) * dx
    s = beta2 * s + (1-beta2) * dx**2
    v_c = v/(1-beta1**t)
    s_c = s/(1-beta2**t)
    dx = -eta * v_c/((s_c**0.5) + eps)
    return v, s, dx
def plotPhi(steadystate, count, ligand, binEdge, folder):
    nBins = np.shape(binEdge)[1]
    for i in range(10):
        hist, edge = np.histogram(steadystate[:, i], bins = binEdge[i])
        hist = hist/np.sum(hist)
        plt.plot(hist)
    target = 1/nBins * np.ones(nBins)
    plt.plot(target, 'ro')
    plt.legend(ligand)
    plt.show()
    plt.savefig(folder + str(count) + '.png')
    

    plt.clf()
    #np.savetxt('Figures/PhiPlots/' + str(count) + '.csv', steadystate, delimiter = ',')

    return 0
def calcPhi(k, ligand_conc, bin_edges, count, folder):
    #Input: k (numDraws x 5), ligand_conc (numExperiments x 1)
    #bin_edges (numExperiments x nBins + 1)
    #Output: phi (numExperiments x nBins) histogrammed data from drawn k
    #For each k, calculate the steady state abundance for each ligand conc
    #(shape = (numDraws x numExperiments)). 
    #Then histogram these results into (nBins x nExperiments)


    #Calculate steady state abu for all k and ligand concentrations
    
    
    numDraws = np.shape(k)[0]
    numExperiments = np.shape(ligand_conc)[0]
    numBins = np.shape(bin_edges)[1] - 1

    SS = np.random.rand(numDraws, numExperiments) #Declare steady state 
    for n in range(numDraws):
        SS[n, :] = calcSteadyState(k[n, :], ligand_conc)
    plotPhi(SS, count, ligand_conc, bin_edges, folder)
    
    phi = np.random.rand(numBins, numExperiments)

    for conc in range(numExperiments):
        phi[:, conc], edge = np.histogram(SS[:, conc], bins = bin_edges[conc])
    
    phi = np.transpose(phi)
    phi_sum = np.sum(phi, axis = 1, keepdims=True)
    phi = phi/phi_sum
    #Normalize

    return phi


def main(data, ligand_conc, k_bounds, alpha = 10**0, nbins = 15, maxiter = 150, numDraws = 20000, folder = 'test/', initializeLambda = False):
#Inputs: data (nsamples x nExperiments. the abundance of (R+B)_ss at steady state)
#k_bounds - (5 x 2. the lower and upper bounds for each k (5 k's in this reaction)
    #Note- bounds should be in log space
#alpha - learning rate for lambda update
#ligand_conc - ligand concentrations. should be unique to each data set
#nbins - number of bins for the histograms
#maxiter - max iterations for gradient descent
#numDraws - number of k draws 
#folder - folder in which data is saved

    psi, bin_edges = getHistBins(data, nbins)
    nExperiments = np.shape(ligand_conc)[0]
    
    np.savetxt(folder + "bin_edges.csv", bin_edges, delimiter = ',')
    print('Starting')
    print("Num Draws: ", numDraws)
    #print(np.shape(bin_edges), np.shape(psi))
    #Need to somehow get distribution of k
    



    #L = initLambda(nExperiments, nbins)
   # L = np.genfromtxt('test/lambda.csv', delimiter = ',')
    # if initializeLambda:
    #     L = np.genfromtxt('run3/lambda.csv', delimiter = ',')
    # else: 
    #     L = initLambda(nExperiments, nbins)

    dLambda = []
    dLambdaNN = [] #Not NOrmalized
    count = 1
    np.savetxt(folder + "psi.csv", psi, delimiter = ',')
    
    d1 = 0
    d2 = 0

    #initialize variables for ADAM optimization
  #  v_lamb = np.zeros(np.shape(L))
   # s_lamb = np.zeros(np.shape(L))
    
    while count < maxiter:
        print("\n")
        print("Iteration: ", count)
        start = time.time()
       #kDraws = mcmc(L, bin_edges, ligand_conc, k_bounds,numDraws, iter = count)
        kDraws = np.genfromtxt('kDraws.csv', delimiter = ',')
        phi = calcPhi(kDraws, ligand_conc, bin_edges, count, folder)
        # kDraws2 = mcmc(L, bin_edges, ligand_conc, k_bounds,numDraws, iter = count)
        # phi = calcPhi(kDraws2, ligand_conc, bin_edges, count, "test2/")
        end = time.time()
        np.savetxt(folder + "kdraws.csv", kDraws, delimiter = ',')
        exit()
        print("MCMC time: ", end - start)


        # kDraws2 = mcmc(L, bin_edges, ligand_conc, k_bounds, numDraws, iter = count)
    
        # phi2 = calcPhi(kDraws2, ligand_conc, bin_edges, count)
        # dLamb2 = dL(phi2, psi)
        dLamb = dL(phi, psi)
        # print(np.linalg.norm(dLamb2, ord = 1))
        # print(np.linalg.norm(dLamb, ord = 1))
        # print(np.mean(np.abs(dLamb - dLamb2)))
        # print(np.linalg.norm(dLamb - dLamb2, ord = 1))
        # np.savetxt("dlamb1 2k.csv", dLamb, delimiter = ',')
        # np.savetxt("dlamb2 2k.csv", dLamb2, delimiter = ',')
        # exit()
        L = L + alpha * dLamb
        #dLamb2 = dL(phi2, psi)
        # plt.plot(np.ndarray.flatten(phi2 - phi), 'ro')
        # plt.yscale('log')
        # plt.show()
        #Save some stuff
        #
        


        dLambda.append(np.linalg.norm(dLamb, ord = 2)/np.linalg.norm(L, ord = 2))
        dLambdaNN.append(np.linalg.norm(dLamb, ord = 2))
       
        if count == 1:
            np.savetxt(folder + "phiInitial.csv", phi, delimiter = ',')

        #Save progress every 1 iterations
        if count % 1 == 0: 

            np.savetxt(folder + "dlambda.csv", np.asarray(dLambda), delimiter = ',')
            np.savetxt(folder + "lambda.csv", L, delimiter = ',')
            np.savetxt(folder + "phiFinal.csv", phi, delimiter = ',')
            np.savetxt(folder + "kdraws.csv", kDraws, delimiter = ',')

        print("error: ", np.mean(np.abs(phi - psi)))
        print("DLAMBDA: ", np.linalg.norm(dLamb, ord = 2)/np.linalg.norm(L, ord = 2))
        print("NOT NORM: ", np.linalg.norm(dLamb, ord = 2))
       
     
        
        count += 1
    np.savetxt(folder + "dlambdaNN.csv", np.asarray(dLambdaNN), delimiter = ',')
    np.savetxt(folder + "dlambda.csv", np.asarray(dLambda), delimiter = ',')
    np.savetxt(folder + "lambda.csv", L, delimiter = ',')
    np.savetxt(folder + "phiFinal.csv", phi, delimiter = ',')
    return 0
def getData(fileName = 'data.mat'):
    #Specific to the file from Hoda
    datKey = 'data_segfr'
    doseKey = 'egf_doses_egfr'
    numDoses = 10
    dat = scipy.io.loadmat(fileName)
    ligand_dose = dat[doseKey]
    abu = np.transpose(dat[datKey])
 
    abundance = []
    #abu[i][0][0][j] gives the abundances for the ith conc
    #j_max = 2 for three repeated experiments at each conc
    for i in range(numDoses):
    #     print('here')
        singleDoseAbu = np.ndarray.flatten(np.concatenate(abu[i][0][0][:]))
      
        abundance.append(singleDoseAbu)

   ######################################
   ########################################
   
    
    #Convert ligand concentrations into a nicer array
    #very inefficient but whatever. we're only doing it once
    numConc = np.shape(ligand_dose)[0]
    ligand_dose = np.ndarray.flatten(ligand_dose)
    

    for i in range(10):
         hist, edges = np.histogram(abundance[i])
         hist = hist/np.sum(hist)
         plt.plot(edges[1:], hist)
    plt.legend(ligand_dose)
    plt.savefig('dataHistogram.png')
    exit()
    plt.clf()

    return abundance, ligand_dose

if __name__ == "__main__":
    dat, lig = getData('data.mat')
    #dat = 0
    #lig = np.genfromtxt('ligand.csv', delimiter = ',') 
#k - [k_1, k_2, k_d, k_d*, k_syn]
    #k_bounds = np.asarray([[-2.3, -1.3], [-1, 0], [-4.25, -3.25], [-4.25, -3.25], [-2.2, -1.2]])


    k_bounds = np.asarray([[-2.3, 0], [-2, 0], [-5.25, -3.25], [-4.25, -2.25], [-2.2, -1]])
    #k0 upper bound increase
    #k1 lower bound decrease    
    #k2 lower bound decrease
    #k3 upper bound increase

    # print(np.shape(lig)[0])
    #dat = fakeData()
    # print(np.shape(dat))
    # exit()
    
    # exit()
    main(dat, lig, k_bounds, initializeLambda = True)
