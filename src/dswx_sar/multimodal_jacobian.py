'''
Jacobian matrix definition for bimodal and trimodal function
'''

import numpy as np

def jacobian_trimodal(x,
                      m1, s1, a1,
                      m2, s2, a2,
                      m3, s3, a3):
    '''
    Generate Jacobian matrix for trimodal function

    Parameters
    ----------
    x: np.ndarray
        Input of the trimodal function

    mu1, sigma1, amplitude1: float
        mean value, standard deviation of the 1st Gaussian function
    
    mu2, sigma2, amplitude2: float
        mean value, standard deviation of the 2nd Gaussian function

    mu3, sigma3, amplitude3: float
        mean value, standard deviation of the 3rd Gaussian function
    
    Returns
    -------
    J : np.ndarray
        Jacobian matrix
    '''

    params = [m1, s1, a1, m2, s2, a2, m3, s3, a3]
    J = np.zeros((x.size, len(params)))  # 9 parameters in total
    gauss_partials = [
        lambda mu, sig, amp, xi: -amp * (mu - xi) * np.exp(-(mu - xi)**2 / (2*sig**2)) / sig**2,  # dmu
        lambda mu, sig, amp, xi: amp * (mu - xi)**2 * np.exp(-(mu - xi)**2 / (2*sig**2)) / sig**3,  # dsig
        lambda mu, sig, amp, xi: np.exp(-(mu - xi)**2 / (2*sig**2))                           # damp
    ]

    for i in range(3):  # three Gaussians
        for j in range(3):  # s, m, a for each Gaussian
            J[:, 3*i + j] = gauss_partials[j](params[3*i], params[3*i+1], params[3*i+2], x)

    return J


def jacobian_bimodal(x,
                     m1, s1, a1,
                     m2, s2, a2):
    '''
    Generate Jacobian matrix for bimodal function

    Parameters
    ----------
    x: np.ndarray
        Input of the trimodal function

    mu1, sigma1, amplitude1: float
        mean value, standard deviation of the 1st Gaussian function
    
    mu2, sigma2, amplitude2: float
        mean value, standard deviation of the 2nd Gaussian function
    
    Returns
    -------
    J : np.ndarray
        Jacobian matrix
    '''

    params = [m1, s1, a1, m2, s2, a2]
    J = np.zeros((x.size, len(params)))  # 6 parameters in total
    gauss_partials = [
        lambda mu, sig, amp, xi: -amp * (mu - xi) * np.exp(-(mu - xi)**2 / (2*sig**2)) / sig**2,  # dmu
        lambda mu, sig, amp, xi: amp * (mu - xi)**2 * np.exp(-(mu - xi)**2 / (2*sig**2)) / sig**3,  # dsig
        lambda mu, sig, amp, xi: np.exp(-(mu - xi)**2 / (2*sig**2))                           # damp
    ]

    for i in range(2):  # two Gaussians
        for j in range(3):  # s, m, a for each Gaussian
            J[:, 3*i + j] = gauss_partials[j](params[3*i], params[3*i+1], params[3*i+2], x)

    return J
