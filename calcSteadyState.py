import numpy as np
from scipy.optimize import fsolve

L = np.genfromtxt('data/ligand.csv', delimiter = ',')
def calcDerivative(x, k, u):
    #u: ligand concentration. here we assume u is a scalar
    #x: all concentrations of receptors. they are given in the order given by the max entropy paper (eq S23 - eq S38):
    #    [0,  1,  2, 3  ,  4, 5   , 6 , 7   , 8 , 9   , 10, 11  , 12    , 13    ,   14, 15] 
    #x = [R, R_i, B, B_i, D1, D1_i, D2, D2_i, P1, P1_i, P2, P2_i, P1_Akt, P2_Akt, pAkt, Akt]
    #k: all k rates. they are given by the supplemental table 1 and are listed as follows:
    #    [0 , 1  , 2 , 3  , 4  , 5  , 6    , 7   , 8   , 9 , 10  , 11 , 12   , 13   , 14 , 15, 16]
    #k = [k1, k-1, k2, k-2, kap, kdp, k*deg, kdeg, ksyn, ki, krec, k*i, k*rec, kbind, kdb, kp, ka]
    
    #dR = ksyn - k1 *u * R + k-1 * B - ki * R + krec * R_i - k2 * R * B + k-2 * D1
    dR = k[8] - k[0] * u * x[0] + k[1] * x[2] - k[9] * x[0] + k[10] * x[1] 
    #dRi = ki * R - krec * R_i - kdeg * R_i
    dRi = k[9] * x[0] - k[10] * x[1] - k[7] * x[1]
    #dB = k1 * u * R - k-1 * B - k2 * R * B + k-2 * D1 - 2 * k2 * B^2 + 2 * k-2 * D2 - ki * B + krec * Bi
    dB = k[0] * u * x[0] - k[1] * x[2]  - k[2] * x[0] * x[2] + k[3] * x[4] - 2 * k[2] * x[2]**2
    + 2 * k[3] * x[6] - k[9] * x[2] + k[10] * x[3]
    #dBi = ki * B - krec * Bi - kdeg * Bi
    dBi = k[9] * x[2] - k[10] * x[3] - k[7] * x[3]
    #dD1 = k2 * R * B - k-2 * D1 - kap * D1 + kdp * P1 - k1 * u * D1 + k-1 * D2 - ki * D1 + krec * D1i
    dD1 = k[2] * x[0] * x[2] - k[3] * x[4] - k[4] * x[4] + k[5] * x[8] - k[0] * u * x[4] + k[1] * x[6] 
    - k[9] * x[4] + k[10] * x[5]
    #dD1i = ki * D1 - krec * D1i - kdeg * D1i + kdp * P1i - kap * D1i
    dD1i = k[9] * x[4] - k[10] * x[5] - k[7] * x[5] + k[5] * x[9] - k[4] * x[5]
    #dD2 = k2 * B^2 - k-2* D2 - ki * D2 + krec * D2i - kap * D2 + kdp * P2 + k1 * U * D1 - k-1 * D2
    dD2 = k[2] * x[2] **2 - k[3] * x[6] - k[9] * x[6] + k[10] * x[7] - k[4] * x[6] + k[5] * x[10] 
    + k[0] * u * x[4] - k[1] * x[6]
    #dD2i = ki * D2 - krec * D2i - kdeg * D2i + kdp * P2i - kap * D2i
    dD2i = k[9] * x[6] - k[10] * x[7] - k[7] * x[7] + k[5] * x[11] - k[4] * x[7]
    #dP1 = kap * D1 - kdp * P1 - k1 * u * P1 + k-1 * P2 - k*i * P1 + k*rec * P1i - kbind * P1 * Akt + kdp * P1atk + ka * P1akt
    dP1 = k[4] * x[4] - k[14] * x[8] - k[0] * u * x[8] + k[1] * x[10] - k[11] * x[8] + k[12] * x[9]
    -k[13] * x[8] * x[15] + k[5] * x[12] + k[16] * x[12]
    #dP1i = k*i * P1 - k*rec * P1i - k*deg * P1i - kdp * P1i + kap * D1i
    dP1i = k[9] * x[8] - k[12] * x[9] - k[6] * x[9] - k[5] * x[9] + k[4] * x[5]
    #dP2 = kap * D2 - kdp * P2 + k1 * u * P1 - k-1 * P2 - k*i * P2 + k*rec * P2i - kbind * P2 * Akt + kdp * P2akt + ka * P2akt
    dP2 = k[4] * x[6] - k[5] * x[10] + k[0] * u * x[8] - k[1] * x[10] - k[11] * x[10] + k[12] * x[11] - k[13] * x[10] * x[15]
    + k[5] * x[13] + k[16] * x[13]
    #dP2i = k*i * P2 - k*rec * P2i - k*deg * P2i - kdp * P2i + kap * D2i
    dP2i = k[11] * x[10] - k[12] *x[11] - k[6] * x[11] - x[5] * x[11] + k[4] * x[7]
    #dP1akt = kbind * P1 * Akt - kdb * P1akt - ka * P1akt
    dP1akt = k[14] * x[8] * x[15] - x[14] * x[12] - k[16] * x[12]
    #dP2akt = kbind * P2 * Akt - kdb * P2akt - ka * P2akt
    dP2akt = k[13] * x[10] * x[15] - k[14] * x[13] - k[16] * x[13]
    #dpAkt = ka * P1akt + ka * P2akt - kp * pAkt
    dpakt = k[16] * x[12] + k[16] * x[13] - k[15] * x[14]
    #dAkt = -kbind * Akt *(P1 + P2 ) + kdp * (P1akt + P2akt) + kp * pAkt
    dakt = -1 * k[13] * x[15] * (x[8] + x[10]) + k[5] * (x[12] + x[13]) + k[15] * x[14]
    dX = np.asarray([dR, dRi, dB, dBi, dD1, dD1i, dD2, dD2i, dP1, dP1i, dP2, dP2i, dP1akt, dP2akt, dpakt, dakt])
    return dX
def model(y, K ,L):
  """Model to be solved by the differential equation"""
  # y is of length 16 
  # K is a vector of length 20  - the first 17 are the parameters and the index 18 is protein abundance and the last two have to do with noise. 
#     t  = t*60 #converting time to seconds
  ns = 16
  K = np.asarray(K) #changing it to numpy array 

  K = 10**K
#     K[:17] = K1
 # K  = np.concatenate((K1 ,K2),axis=0, out=None)
#         % define rates
#     % EGF binding to EGFR monomer
  k1      = K[0]
#     % EGF unbinding from EGFR
  kn1     = K[1]
#     % EGFR EGF-EGFR dimerization
  k2      = K[2]
#     % EGFR-EGF-EGFR undimerization
  kn2     = K[3]
#     % receptor phosphorylation
  kap     = K[4]
#     % receptor dephosphorylation
  kdp     = K[5]
#     % degradation of inactive (not phosphorylated)
  kdeg    = K[6]
#     % degradation of active (phosphorylated)
  kdegs   = K[7]
#     % internalization of inactive
  ki      = K[8]
#     % internalization of active
  kis     = K[9]
#     % recycling of inactive
  krec    = K[10]
#     % recycling of active
  krecs   = K[11]
#     % rate of pEGFR binding to Akt
  kbind   = K[12]
#     % rate of pEGFR-Akt unbinding
  kunbind = K[13]
#     % Rate of Akt phosphorylation
  kpakt   = K[14]
#     % rate of pAkt dephosphorylation
  kdpakt  = K[15]
#     % EGFR delivery rate
  ksyn    = K[16]

#     % define concentrations

#     % ligand free receptors, plasma 
  r    = y[0]
#     % ligand free receptors, endosomes
  ri   = y[1]
#     % ligand bound receptors, plasma membrane
  b    = y[2]
#     % ligand bound receptors, endosomes
  bi   = y[3]
#     % 1 ligand bound dimers, plasma membrane
  d1   = y[4]
#     % 1 ligand bound dimers, endosomes
  d1i  = y[5]
#     % 2 ligand bound dimers, plasma membrane
  d2   = y[6]
#     % 2 ligand bound dimers, endosomes
  d2i  = y[7]
#     % 1 ligand bound phosphorylated dimers, plasma membrane
  p1   = y[8]
#     % 1 ligand bound phosphorylated dimers, endosomes
  p1i  = y[9]
#     % 2 ligand bound phosphorylated dimers, plasma membrane
  p2   = y[10]
#     % 2 ligand bound phosphorylated dimers, endosomes
  p2i  = y[11]
#     % 1L dimer bound to Akt
  p1a  = y[12]
#     % 2L dimer bound to Akt
  p2a  = y[13]
#     % pakt
  pakt = y[14]
#     % free akt
  akt  = y[15]
#     % Need to set one of the rate constants for thermodynamic consistency
  #initialize the differential equation variable
  dys = np.zeros(ns)
# %
#     % free receptors, plasma membrane
  dys[0]  = ksyn - k1*L*r + kn1*b - ki*r + krec*ri - k2*r*b + kn2*d1
#     % free receptors, endosomes
  dys[1]  = ki*r - krec*ri - kdeg*ri
#     % bound receptors, plasma membrane
  dys[2]  = k1*L*r - kn1*b - k2*r*b + kn2*d1 - 2*k2*b*b + 2*kn2*d2 - ki*b + krec*bi
#     % bound receptors, endosomes
  dys[3]  = ki*b - krec*bi - kdeg*bi
#     % 1 ligand bound dimer, plasma membrane
  dys[4]  = k2*r*b - kn2*d1 - kap*d1 + kdp*p1 - k1*L*d1 + kn1*d2 - ki*d1 + krec*d1i
#     % 1 ligand bound dimer, endosomes
  dys[5]  = ki*d1 - krec*d1i - kdeg*d1i + kdp*p1i - kap*d1i
#     % 2 ligand bound dimer, plasma membrane
  dys[6]  = k2*b*b - kn2*d2 - ki*d2 + krec*d2i - kap*d2 + kdp*p2 + k1*L*d1 - kn1*d2
#     % 2 ligand bound dimer, endosomes
  dys[7]  = ki*d2 - krec*d2i - kdeg*d2i + kdp*p2i - kap*d2i
#     % 1 ligand bound phosphorylated dimer, plasma membrane
  dys[8]  = kap*d1 - kdp*p1  - kis*p1 + krecs*p1i - k1*L*p1 + kn1*p2 - kbind*akt*p1 + kunbind*p1a + kpakt*p1a
#     % 1 ligand bound phosphorylated dimer, endosomes
  dys[9] = kis*p1 - krecs*p1i - kdegs*p1i + kap*d1i - kdp*p1i
#     % 2 ligand bound phosphorylated dimer, plasma membrane
  dys[10] = kap*d2 - kdp*p2 - kis*p2 + krecs*p2i + k1*L*p1 - kn1*p2 - kbind*akt*p2 + kunbind*p2a + kpakt*p2a
#     % 2 ligand bound phosphorylated dimer, endosomes
  dys[11]= kis*p2 - krecs*p2i - kdegs*p2i - kdp*p2i + kap*d2i
#     % p1 bound to Akt
  dys[12] = kbind*p1*akt - kpakt*p1a - kunbind*p1a
#     % p2 bound to Akt
  dys[13] = kbind*p2*akt - kpakt*p2a - kunbind*p2a
#     % pAkt
  dys[14] = kpakt*(p1a+p2a) - kdpakt*pakt
#     % free Akt
  dys[15] = -kbind*akt*(p1+p2) + kdpakt*pakt + kunbind*(p1a+p2a)
  return dys
def calcSteadyState2(k_all, L = np.genfromtxt('data/ligand.csv', delimiter = ',')):
    #Input: k (numSample x 17 vector) in log space, L, (ligand concentration, numLigands x 1)
        #    [0,  1,  2, 3  ,  4, 5   , 6 , 7   , 8 , 9   , 10, 11  , 12    , 13    ,   14, 15] 
    #x = [R, R_i, B, B_i, D1, D1_i, D2, D2_i, P1, P1_i, P2, P2_i, P1_Akt, P2_Akt, pAkt, Akt]
    #output: tot_rec: total number of receptors on the cell surface. this is defined as R + B + 2*D1 + 2*D2 + 2 * P1 + 2 * P2
        #  + 2 * P2akt + 2 * P1akt
    #k_all = 10**k_all
    numVar = 16
    numLigand = np.shape(L)[0] 
    x_init = np.random.rand(numVar)
    sol = []
    for i in range(numLigand):
        y = fsolve(model, x_init, args = (k_all, L[i]))
       # s = fsolve(calcDerivative, x_init, args = (k_all, L[i]))
        s =  y[0] + y[2] + 2*(y[4] + y[6] + y[8]+ y[10] + y[12] + y[13] )
        sol.append(s)
    return sol
def calcSteadyState(k_all, L = np.genfromtxt('data/ligand.csv', delimiter = ',')):
#Calculates steady state abundances for given k vector and ligand concentration
#Input: k (numSample x 11 vector) (log space!), L (ligand concentration, numExperiments x 1)
#Ouput: R+B_ss (numExperiments x 1) steady state solutions of receptors (R) and bounded 
#receptors (B)
    numL = np.shape(L)[0]
    #Assign variable names to each k entry: 
    #[ksyn, k1, k2, ki, ki*, krec, krec*, kp, kp-, kdeg, kdeg*]
    numSample = np.shape(k_all)[0]
    output = []
    
    for i in range(numSample):
        k = k_all[i]

        ksyn = 10**k[0]
        k1 = 10**k[1]
        k2 = 10**k[2]
        ki = 10**k[3]
        ki0 = 10**k[4]
        krec = 10**k[5]
        krec0 = 10**k[6]
        kp = 10**k[7]
        kp0 = 10**k[8] 
        kdeg = 10**k[9]
        kdeg0 = 10**k[10]
        

        # k_1 = k[0] 
        # k_2 = k[1]
        # k_d = k[2] 
        # k_dstar = k[3]
        # k_syn = k[4]
        # k_1 = 

        #Create matrices of system and solve w linalg
        RB_ss = np.random.rand(np.shape(L)[0])
        RBP = np.random.rand(np.shape(L)[0])
        count = 0
        # k_1 = 10**-2.3
        # k_2 = 10**-0.5
        # k_d = 10**-3.25
        # k_dstar = 10**-3.25
        # k_syn = 10**-1.2

        for conc in L:
            a = np.asarray([[-ki - k1 * conc, k2, 0, krec, 0, 0], 
                            [k1 * conc, -kp - k2 - ki, kp0, 0, krec, 0], 
                            [0, kp, -kp0 - ki0, 0, 0, krec0],
                            [ki, 0, 0, -krec - kdeg, 0, 0],
                            [0, ki, 0, 0, -krec - kdeg, 0],
                            [0, 0, ki0, 0, 0, -krec0 - kdeg0]])
            b = np.asarray([-ksyn, 0, 0, 0, 0, 0]) 
            a = a.astype(np.float64)
            b = b.astype(np.float64)
    
            #iterate through each ligand concentration (conc) in L
            #######ALTERNATE CALCULATION FOR RB
            try:
                RB = np.linalg.solve(a, b)
            except:
                return np.zeros(numL) - 100
            RB_ss[count] = RB[0] + RB[1] + RB[2]
            count += 1
        output.append(RB_ss)
    
            #######ALTERNATE CALCULATION FOR RB

    return np.asarray(output)

def calcSteadyState_comp(k_all):
#Calculates steady state abundances for given k vector and ligand concentration
#Input: k (numSample x 11 vector) (log space!), L (ligand concentration, numExperiments x 1)
#Ouput: R+B_ss (numExperiments x 1) steady state solutions of receptors (R) and bounded 
#receptors (B)
    numL = np.shape(L)[0]
    #Assign variable names to each k entry: 
    #[ksyn, k1, k2, ki, ki*, krec, krec*, kp, kp-, kdeg, kdeg*]
    numSample = np.shape(k_all)[0]
    output = []
    
    for i in range(numSample):
        k = k_all[i]

        ksyn = 10**k[0]
        k1 = 10**k[1]
        k2 = 10**k[2]
        ki = 10**k[3]
        ki0 = 10**k[4]
        krec = 10**k[5]
        krec0 = 10**k[6]
        kp = 10**k[7]
        kp0 = 10**k[8] 
        kdeg = 10**k[9]
        kdeg0 = 10**k[10]
        

        # k_1 = k[0] 
        # k_2 = k[1]
        # k_d = k[2] 
        # k_dstar = k[3]
        # k_syn = k[4]
        # k_1 = 

        #Create matrices of system and solve w linalg
        RB_ss = np.random.rand(np.shape(L)[0])
        RBP = np.random.rand(np.shape(L)[0])
        count = 0
        # k_1 = 10**-2.3
        # k_2 = 10**-0.5
        # k_d = 10**-3.25
        # k_dstar = 10**-3.25
        # k_syn = 10**-1.2

        for conc in L:
            a = np.asarray([[-ki - k1 * conc, k2, 0, krec, 0, 0], 
                            [k1 * conc, -kp - k2 - ki, kp0, 0, krec, 0], 
                            [0, kp, -kp0 - ki0, 0, 0, krec0],
                            [ki, 0, 0, -krec - kdeg, 0, 0],
                            [0, ki, 0, 0, -krec - kdeg, 0],
                            [0, 0, ki0, 0, 0, -krec0 - kdeg0]])
            b = np.asarray([-ksyn, 0, 0, 0, 0, 0]) 
            a = a.astype(np.float64)
            b = b.astype(np.float64)
    
            #iterate through each ligand concentration (conc) in L
            #######ALTERNATE CALCULATION FOR RB
            try:
                RB = np.linalg.solve(a, b)
            except:
                return np.zeros(numL) - 100
            RB_ss[count] = RB[0] + RB[1] + RB[2]
            count += 1
        output.append(RB_ss)
    
            #######ALTERNATE CALCULATION FOR RB

    return np.asarray(output)
if __name__ == '__main__':
    kdraws = np.genfromtxt('data/kdraws.csv', delimiter = ',')
    kdraws = np.random.rand(17)
    output = calcSteadyState(kdraws)

    #np.savetxt('data/output.csv', output, delimiter = ',')
