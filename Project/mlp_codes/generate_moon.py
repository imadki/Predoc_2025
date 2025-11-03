from sklearn.datasets import make_moons
import numpy as np

X, y = make_moons(100, noise=0.20, random_state=3)
np.savetxt("data/data_X.txt", X)
np.savetxt("data/data_y.txt", y, fmt="%d")
