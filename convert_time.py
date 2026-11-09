import numpy as np

hours = np.array([14, 14, 15, 15, 16, 16, 17, 17, 18, 19])
minutes = np.array([34, 53, 24, 56, 26, 58, 30, 50, 56, 29])

seconds = hours * 3600 + minutes * 60
time = seconds - seconds[0]

if __name__ == '__main__':
    print('time =', time)
