

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from calcSteadyState import calcSteadyState
import math
import matplotlib.pyplot as plt
import time
def objective_KDE_multPC(beta, alpha = np.genfromtxt('data/alpha.csv', delimiter = ','), INPUT =np.genfromtxt('data/kdraws.csv', delimiter = ','), OUTPUT = np.genfromtxt('data/output.csv', delimiter = ',')):
    kdraws = INPUT

    O = OUTPUT

    eps = 0.01 #* min([np.linalg.norm(alpha, ord = 2), np.linalg.norm(beta, ord =2)])
    all_var = np.var(O, axis = 0)
    alpha = np.reshape(alpha, (1, 11))
    beta = np.reshape(beta, (1, 11))
    
    
    a_0 = np.matmul(alpha, kdraws.transpose())[0]
    b_0 = np.matmul(beta, kdraws.transpose())[0]

    a_0, alpha_min, alpha_len = scale_down(a_0)
    b_0, beta_min, beta_len = scale_down(b_0)
    o_resize = []
    o_min = []
    o_len = []
    for i in range(np.shape(OUTPUT)[1]):
        resize, minimum, length = scale_down(O[:, i])
        o_resize.append(resize)
        o_min.append(minimum)
        o_len.append(length)

    O = np.transpose(np.asarray(o_resize))
    
    all_data = []
    #epsilon = [0.01]

    for _ in range(1):
       # SI = []
        for _ in range(1):
            numSample = np.shape(kdraws)[0]

            numAlpha = 1000
            #eps = i
            N = []
            expectations = []
            # plt.plot(a_0, O[:,5], 'ro')
            # plt.show()
            # exit()
            for _ in range(numAlpha):
                
                #choose random element of a_0
                #sample alpha with mean a_0 and var = eps
    
                meanB = np.random.choice(b_0)
                ndrawB = np.random.normal(meanB, eps)
                meanA = np.random.choice(a_0)
                ndrawA = np.random.normal(meanA, eps)
            #  N.append(ndraw)
                N_i = []
                for j in range(numSample):
                    
                    # Define the mean and standard deviation
        #         print(j)
                    muB = b_0[j]
                  #  sigmaB = eps
                    muA = a_0[j]
                   # sigmaA = eps

                    # Calculate the value of the normal distribution PDF at x
                    xA = ndrawA
                    pdfA = (1 / (eps * math.sqrt(2 * math.pi))) * math.exp(-((xA - muA)**2) / (2 * eps**2))
                    xB = ndrawB
                    pdfB = (1 / (eps * math.sqrt(2 * math.pi))) * math.exp(-((xB - muB)**2) / (2 * eps**2))
                    #calculate pdf of norm centered around a_0[j] with var eps
                    N_i.append(pdfA * pdfB)
                
                expect = np.dot(N_i, O)/np.sum(N_i)
  
            #  print(np.shape(expect))
                expectations.append(expect)
           # SI.append(np.mean(np.var(expectations, axis = 0)/all_var))
           # print("eps: " + str(eps) +  "  " + str(np.mean(np.var(expectations, axis = 0)/all_var)))

        expectations = np.asarray(expectations)
        expectations = np.transpose(np.asarray([scale_up(expectations[:, i], o_min[i], o_len[i]) for i in range(9)]))
        
  
    # for i in range(9):
    #     plt.plot(expectations[:, i])
    #     plt.show()
    # print((all_var))
   # np.savetxt('temp2.csv', expectations, delimiter = ',')
    return -1 * np.mean((np.var(expectations, axis = 0)/all_var))
    return -1 * np.mean(np.var(expectations, axis = 0)/all_var)
def objective_KDE(alpha, INPUT = np.genfromtxt('data/kdraws.csv', delimiter = ','), OUTPUT = np.genfromtxt('data/output.csv', delimiter = ',')):

    kdraws = INPUT
    
    O = OUTPUT
    eps = 0.01 * np.linalg.norm(alpha, ord = 2)
    all_var = np.var(O, axis = 0)
    alpha = np.reshape(alpha, (1, 11))
    a_0 = np.matmul(alpha, kdraws.transpose())[0]
    all_data = []
    #epsilon = [0.01]

    for i in range(1):
       # SI = []
        for k in range(1):
            numSample = np.shape(kdraws)[0]

            numAlpha = 3000
            #eps = i
            N = []
            expectations = []
            # plt.plot(a_0, O[:,5], 'ro')
            # plt.show()
            # exit()
            for _ in range(numAlpha):
                
                #choose random element of a_0
                #sample alpha with mean a_0 and var = eps
                
                mean = np.random.choice(a_0)
                ndraw = np.random.normal(mean, eps)
            #  N.append(ndraw)
                N_i = []
                for j in range(numSample):
                    
                    # Define the mean and standard deviation
        #         print(j)
                    mu = a_0[j]
                    sigma = eps

                    # Calculate the value of the normal distribution PDF at x
                    x = ndraw

                    pdf = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu)**2) / (2 * sigma**2))
                 
                    #calculate pdf of norm centered around a_0[j] with var eps
                    N_i.append(pdf)
                
                expect = np.dot(N_i, O)/np.sum(N_i)
  
            #  print(np.shape(expect))
                expectations.append(expect)
           # SI.append(np.mean(np.var(expectations, axis = 0)/all_var))
           # print("eps: " + str(eps) +  "  " + str(np.mean(np.var(expectations, axis = 0)/all_var)))
 
        expectations = np.asarray(expectations)
    # for i in range(9):
    #     plt.plot(expectations[:, i])
    #     plt.show()
    # print((all_var))
   # np.savetxt('temp2.csv', expectations, delimiter = ',')
    return -1 * np.mean(np.var(expectations, axis = 0)/all_var)

def objective(ALPHA, i = 0, INPUT = np.genfromtxt('data/kDraws.csv', delimiter = ','), OUTPUT = np.genfromtxt('data/output.csv', delimiter = ','), nbins = 20, LIGAND = np.genfromtxt('data/ligand.csv', delimiter = ',')):
#Objective function for the simulated annealing
#In this case, the objective function is to calculate the first order sensitivity index 
    
    numLigand = np.shape(OUTPUT)[1]
   # var = np.var(OUTPUT, axis = 0)
    tot_var = np.var(OUTPUT, axis = 0)
    
    #Calculate the input which is linear combination of the K's
    input = ALPHA * INPUT
    
    input = np.sum(input, axis = 1)
    #print(np.shape(input))
    indices = np.argsort(input)
    sorted_input = input[indices]
    sorted_output = OUTPUT[indices, :]
    #print(np.shape(sorted_output))
    var_explained = []

    #Calculate Var(E(f|x_i))
    for l in range(numLigand):
        part = partition_array(sorted_output[:, l], nbins)
        expect = np.mean(part, axis = 1) 
        var_explained.append(np.var(expect))
    SI = var_explained/tot_var
  
    return -1 * np.average(SI)
def scale_down(arr):
    #scales the array to between 0 and 1
    length = np.max(arr) - np.min(arr)
    min = np.min(arr)
    return (arr - min)/length, min, length
def scale_up(arr, min, len):
    arr2 = arr * len + min
    return arr2
def objective_BIN_multPC(BETA, ALPHA = np.genfromtxt('data/alpha.csv', delimiter = ','), INPUT = np.genfromtxt('data/kDraws.csv', delimiter = ','), OUTPUT = np.genfromtxt('data/output.csv', delimiter = ','), nbins = 10, LIGAND = np.genfromtxt('data/ligand.csv', delimiter = ',')):
#Objective function for the simulated annealing
#In this case, the objective function is to calculate the first order sensitivity index
    
 
    numSample = np.shape(INPUT)[0]
    numLigand = np.shape(OUTPUT)[1]
   # var = np.var(OUTPUT, axis = 0)
    tot_var = np.var(OUTPUT, axis = 0)
    eps = 0.01

    PC1 = np.sum(ALPHA * INPUT, axis = 1)
    PC2 = np.sum(BETA * INPUT, axis = 1)
    pc1_bounds = [np.min(PC1) - eps, np.max(PC1)+ eps]
    pc2_bounds = [np.min(PC2)- eps, np.max(PC2) + eps]
    pc1_divider = np.linspace(pc1_bounds[0], pc1_bounds[1], nbins + 1)
    pc2_divider = np.linspace(pc2_bounds[0], pc2_bounds[1], nbins + 1)
    grid = np.empty((nbins, nbins), dtype = object) #(pc1, pc2)
    weight = np.zeros((nbins, nbins))
   # print(np.shape(pc1_divider))
    for i in range(nbins):
        for j in range(nbins):
            grid[i][j] = []     


    pc1_digitize = np.digitize(PC1, pc1_divider)
    pc2_digitize = np.digitize(PC2, pc2_divider)

    for i in range(numSample):
       # print(pc1_digitize[i], pc2_digitize[i])
        grid[pc1_digitize[i] - 1][pc2_digitize[i] - 1].append(OUTPUT[i])
        weight[pc1_digitize[i] - 1][pc2_digitize[i] - 1] += 1
    for i in range(nbins):
        for j in range(nbins):
            #check to see if there are any data points at a given grid point
          #  print(grid[i][j])
            
            if np.shape(grid[i][j])[0] > 0:
                mean = np.mean(grid[i][j], axis = 0)            
                #print(mean)
                grid[i][j] = mean
            else:
                #print('else')
                grid[i][j] = np.zeros(numLigand)
           # print(np.shape(grid[i][j]))

    
    grid = np.ndarray.flatten(grid)

   #print(np.shape(np.asarray(grid)))
    grid = np.asarray([grid[i] for i in range(np.shape(grid)[0])])
    
 
    weight = np.ndarray.flatten(weight)
    var = [weighted_var(grid[:, i], weights = weight)for i in range(numLigand)]

    #print(np.shape(var))
    #exit()
    return -1 * np.mean(var/tot_var)
def obj_all(BETA, ALPHA = np.genfromtxt('data/alpha.csv', delimiter = ','), INPUT = np.genfromtxt('data/kDraws.csv', delimiter = ','), OUTPUT = np.genfromtxt('data/output.csv', delimiter = ','), nbins = 10, LIGAND = np.genfromtxt('data/ligand.csv', delimiter = ',')):
#Objective function for the simulated annealing
#In this case, the objective function is to calculate the first order sensitivity index
    
    
    numSample = np.shape(INPUT)[0]
    numLigand = np.shape(OUTPUT)[1]
   # var = np.var(OUTPUT, axis = 0)
    tot_var = np.var(OUTPUT, axis = 0)
    eps = 0.01

    PC1 = np.sum(ALPHA * INPUT, axis = 1)
    PC2 = np.sum(BETA * INPUT, axis = 1)
    pc1_bounds = [np.min(PC1) - eps, np.max(PC1)+ eps]
    pc2_bounds = [np.min(PC2)- eps, np.max(PC2) + eps]
    pc1_divider = np.linspace(pc1_bounds[0], pc1_bounds[1], nbins + 1)
    pc2_divider = np.linspace(pc2_bounds[0], pc2_bounds[1], nbins + 1)
    grid = np.empty((nbins, nbins), dtype = object) #(pc1, pc2)
    weight = np.zeros((nbins, nbins))
   # print(np.shape(pc1_divider))
    for i in range(nbins):
        for j in range(nbins):
            grid[i][j] = []     


    pc1_digitize = np.digitize(PC1, pc1_divider)
    pc2_digitize = np.digitize(PC2, pc2_divider)

    for i in range(numSample):
       # print(pc1_digitize[i], pc2_digitize[i])
        grid[pc1_digitize[i] - 1][pc2_digitize[i] - 1].append(OUTPUT[i])
        weight[pc1_digitize[i] - 1][pc2_digitize[i] - 1] += 1
    for i in range(nbins):
        for j in range(nbins):
            #check to see if there are any data points at a given grid point
          #  print(grid[i][j])
            
            if np.shape(grid[i][j])[0] > 0:
                mean = np.mean(grid[i][j], axis = 0)            
                #print(mean)
                grid[i][j] = mean
            else:
                #print('else')
                grid[i][j] = np.zeros(numLigand)
           # print(np.shape(grid[i][j]))

    
    grid = np.ndarray.flatten(grid)

   #print(np.shape(np.asarray(grid)))
    grid = np.asarray([grid[i] for i in range(np.shape(grid)[0])])
    
 
    weight = np.ndarray.flatten(weight)
    var = [weighted_var(grid[:, i], weights = weight)for i in range(numLigand)]

    #print(np.shape(var))
    #exit()
    return -1 * (var/tot_var)
def weighted_var(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return variance
def objective_BIN(ALPHA, INPUT = np.genfromtxt('data/kDraws.csv', delimiter = ','), OUTPUT = np.genfromtxt('data/output.csv', delimiter = ','), nbins = 10, LIGAND = np.genfromtxt('data/ligand.csv', delimiter = ',')):
#Objective function for the simulated annealing
#In this case, the objective function is to calculate the first order sensitivity index 
    
    numLigand = np.shape(OUTPUT)[1]
   # var = np.var(OUTPUT, axis = 0)
    tot_var = np.var(OUTPUT, axis = 0)
    
    #Calculate the input which is linear combination of the K's
    input = ALPHA * INPUT
    
    input = np.sum(input, axis = 1)
    #print(np.shape(input))
    indices = np.argsort(input)
    sorted_input = input[indices]
    sorted_output = OUTPUT[indices, :]
    #print(np.shape(sorted_output))
    var_explained = []

    #Calculate Var(E(f|x_i))
    for l in range(numLigand):
        part = partition_array(sorted_output[:, l], nbins)
        expect = np.mean(part, axis = 1) 
        var_explained.append(np.var(expect))
    SI = var_explained/tot_var
  
    return -1 * np.average(SI)
def partition_array(arr, d):
    n = len(arr)
    partition_size = n // d
    partitions = [arr[i:i + partition_size] for i in range(0, n, partition_size)]
    return partitions


if __name__ == "__main__":
    beta = np.genfromtxt('data/beta.csv', delimiter = ',')
    alpha = np.genfromtxt('data/alpha.csv', delimiter = ',')
    beta = np.random.rand(11)

    t = []
    kde_1pc = []
    bin_1pc = []
    kde_2pc = []
    bin_2pc = []
    for i in range(10):
   
        beta = np.random.rand(11)
        obj = objective_KDE(alpha)
       
        obj_bin = objective_BIN(alpha)
  
        kde_1pc.append(obj)
        bin_1pc.append(obj_bin)

        temp2 = objective_BIN_multPC(beta)
        print('enter kde')
        start = time.time()  
        temp = objective_KDE_multPC(beta)
        end = time.time()

        kde_2pc.append(temp)
        bin_2pc.append(temp2)
        elapsed = end - start
        print(elapsed)
      #  print('1pc: ', obj)
        print('2pc: ', temp, temp2)
    diff1 = np.asarray(kde_1pc) - np.asarray(bin_1pc)
    diff2 = np.asarray(kde_2pc) - np.asarray(bin_2pc)
    m1 = np.mean(diff1)
    v1 = np.var(diff1)
    m2 = np.mean(diff2)
    v2 = np.var(diff2)
    print('m1: ', m1)
    print('m2: ', m2)
    print('v1: ', v1)
    print('v2: ', v2)
    # kdraws = np.genfromtxt('data/kdraws.csv', delimiter= ',')
    # alpha = np.genfromtxt('data/alpha.csv', delimiter = ',')
    # O = calcSteadyState(kdraws)
    # alpha = np.random.rand(11)
    # start = time.time()
    # results = minimize(objective_KDE, alpha, method = 'Powell')
    # print(objective_KDE(results.x))
    # np.savetxt('temp_alpha.csv', results.x, delimiter = ',')
    # end = time.time()
    # print(end - start)
