import numpy as np
from matplotlib import pyplot as plt

def div_sigmoid(x, coe=1):
    x = coe*x
    return (1 - np.exp(-x)) / (1 + np.exp(-x))



if __name__ == '__main__':
    x = np.arange(0, 100, 0.25)
    y = div_sigmoid(x, 0.1)
    plt.plot(x, y)
    plt.show()