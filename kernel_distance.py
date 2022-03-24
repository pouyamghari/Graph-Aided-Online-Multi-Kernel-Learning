import numpy as np
import math
from scipy.stats import norm

def kernel_distance(param_1, param_2, kernel_type_1, kernel_type_2):
    if kernel_type_1 == 'Gaussian' and kernel_type_2 == 'Gaussian':
        dis = ((param_1**1.5)/np.sqrt(math.pi)) + ((param_2**1.5)/np.sqrt(math.pi))
        dis -= np.sqrt(param_1*param_2)/np.sqrt((2*math.pi/param_1)+(2*math.pi/param_2))
        
    elif kernel_type_1 == 'Gaussian' and kernel_type_2 == 'Laplacian':
        dis = ((param_1**1.5)/np.sqrt(math.pi)) + (param_2/(4*(math.pi**2)))
        dis -= ((param_1)/(2*math.pi))*(1+norm.cdf(np.sqrt(param_1)/(np.sqrt(2)*param_2))-norm.cdf(-np.sqrt(param_1)/(np.sqrt(2)*param_2)))
        
    elif kernel_type_1 == 'Laplacian' and kernel_type_2 == 'Gaussian':
        dis = ((param_2**1.5)/np.sqrt(math.pi)) + (param_1/(4*(math.pi**2)))
        dis -= ((param_2)/(2*math.pi))*(1+norm.cdf(np.sqrt(param_2)/(np.sqrt(2)*param_1))-norm.cdf(-np.sqrt(param_2)/(np.sqrt(2)*param_1)))
        
    elif kernel_type_1 == 'Laplacian' and kernel_type_2 == 'Laplacian':
        dis = (param_1/(4*(math.pi**2))) + (param_2/(4*(math.pi**2)))
        dis -= (param_1*param_2)/(2*(math.pi**2)*(param_1+param_2))