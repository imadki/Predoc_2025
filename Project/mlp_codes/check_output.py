import numpy as np
import matplotlib.pyplot as plt

# Chargement
X = np.loadtxt("data/data_X.txt")
y = np.loadtxt("data/data_y.txt", dtype=int)
W1 = np.loadtxt("output/W1.txt")
b1 = np.loadtxt("output/b1.txt")
W2 = np.loadtxt("output/W2.txt")
b2 = np.loadtxt("output/b2.txt")

# Remettre en forme
nn_input_dim = 2
nn_hdim = len(b1)
nn_output_dim = len(b2)

W1 = W1.reshape((nn_input_dim, nn_hdim))
W2 = W2.reshape((nn_hdim, nn_output_dim))
b1 = b1.reshape((1, nn_hdim))
b2 = b2.reshape((1, nn_output_dim))

# Forward identique au C
def forward(X):
    a1 = np.tanh(X.dot(W1) + b1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# Grille
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot
plt.figure(figsize=(5,4))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral, edgecolor='k')
plt.title(f"Decision Boundary (hidden={nn_hdim})")
plt.tight_layout()
plt.show()
