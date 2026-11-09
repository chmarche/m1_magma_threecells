import numpy as np

P = 0.1            # (MPa)
T = 1373.0         # (K)
X = 6.2790e-04

m = -20.79 - 5030 / T - 1.4 * P / T

D = X * np.exp(m) * (1 + np.exp(56 + m + X * (-34.1 + 44620 / T + 5.73 * P / T)
                                - np.sqrt(X) * (0.091 + 4.77e6 / T ** 2)))
D = D * 1e-12

if __name__ == '__main__':
    print('D =', D)
