import numpy as np
from scipy.optimize import minimize
from calcSteadyState import calcSteadyState
import math
import matplotlib.pyplot as plt
import time
import objective as obj
from scipy.optimize import dual_annealing
from scipy.optimize import basinhopping
def optimize(file):
    #alpha = np.genfromtxt('test/beta_opt_0.csv', delimiter = ',')
    alpha = np.random.rand(11)
    #alpha = np.linspace(0, 3, 11)
    #print("Bin: ", obj.objective_BIN(alpha))

    start = time.time()
    print('Initial: ', obj.objective_KDE_multPC(alpha))
    end = time.time()
    print('Time for 1 function call: ', end - start)
  #  bound = [[-10, 10] for _ in range(11)]
    result = basinhopping()
    #result = dual_annealing(obj.objective_KDE_multPC, bounds = bound)
    print(np.shape(result))
    print(result)
    #result = minimize(obj.objective_KDE_multPC, alpha, method = 'L-BFGS-B')
    # si = obj.objective(alpha)f
    # print(si)
    maxi = result.x
    print(maxi)
    max_output = obj.objective_KDE_multPC(maxi)
    print('Final: ', max_output)
    #print("Bin: ", obj.objective_BIN(maxi))
    np.savetxt(file, maxi, delimiter = ',')

    return max_output
if __name__ == "__main__":

    for j in range(10):
        start = time.time()
        path = 'data/beta_10kKDE_' + str(j) +'.csv'
    #    beta = np.genfromtxt(path, delimiter = ',')
    

       # path = 'test/alpha_opt_.csv'
        opt = optimize(file = path)
        print(j, " ", np.round(opt, 5))
        end = time.time()
        print('time: ', end - start)
      

  #  alpha = np.genfromtxt('data/alpha.csv', delimiter = ',')

  

    
