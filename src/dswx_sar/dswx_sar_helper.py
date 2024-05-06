import numpy as np

def jacobian_trimodal(x,
                      m1, s1, a1,
                      m2, s2, a2,
                      m3, s3, a3):
    
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
    
    params = [m1, s1, a1, m2, s2, a2]
    J = np.zeros((x.size, len(params)))  # 9 parameters in total
    gauss_partials = [
        lambda mu, sig, amp, xi: -amp * (mu - xi) * np.exp(-(mu - xi)**2 / (2*sig**2)) / sig**2,  # dmu
        lambda mu, sig, amp, xi: amp * (mu - xi)**2 * np.exp(-(mu - xi)**2 / (2*sig**2)) / sig**3,  # dsig
        lambda mu, sig, amp, xi: np.exp(-(mu - xi)**2 / (2*sig**2))                           # damp
    ]

    for i in range(2):  # two Gaussians
        for j in range(3):  # s, m, a for each Gaussian
            J[:, 3*i + j] = gauss_partials[j](params[3*i], params[3*i+1], params[3*i+2], x)

    return J

