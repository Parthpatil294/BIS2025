import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Step 1: Dataset Generators (no sklearn needed)
# =========================================================
def make_moons_manual(n_samples=200, noise=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # outer moon
    outer_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    outer = np.vstack([outer_x, outer_y]).T
    outer_labels = np.zeros(n_samples_out, dtype=int)

    # inner moon
    inner_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_y = -np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
    inner = np.vstack([inner_x, inner_y]).T
    inner_labels = np.ones(n_samples_in, dtype=int)

    X = np.vstack([outer, inner])
    y = np.concatenate([outer_labels, inner_labels])
    X += noise * np.random.randn(*X.shape)
    return X, y.reshape(-1,1)

def make_blobs_manual(n_samples=300, centers=3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    X = []
    y = []
    for i in range(centers):
        cx, cy = np.random.uniform(-5,5), np.random.uniform(-5,5)
        points = np.random.randn(n_samples//centers, 2) + np.array([cx, cy])
        labels = np.full(n_samples//centers, i)
        X.append(points)
        y.append(labels)
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y.reshape(-1,1)

def make_sine_manual(n_samples=200, noise=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.linspace(-2*np.pi, 2*np.pi, n_samples).reshape(-1,1)
    y = np.sin(X) + noise*np.random.randn(*X.shape)
    return X, y

# =========================================================
# Step 2: Neural Network
# =========================================================
def sigmoid(x): return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.num_params = (n_input*n_hidden) + n_hidden + (n_hidden*n_output) + n_output

    def unpack_weights(self, vec):
        idx = 0
        W1 = vec[idx: idx+self.n_input*self.n_hidden].reshape((self.n_input, self.n_hidden)); idx += self.n_input*self.n_hidden
        b1 = vec[idx: idx+self.n_hidden].reshape((1, self.n_hidden)); idx += self.n_hidden
        W2 = vec[idx: idx+self.n_hidden*self.n_output].reshape((self.n_hidden, self.n_output)); idx += self.n_hidden*self.n_output
        b2 = vec[idx: idx+self.n_output].reshape((1, self.n_output))
        return W1, b1, W2, b2

    def forward(self, X, vec):
        W1, b1, W2, b2 = self.unpack_weights(vec)
        a1 = sigmoid(np.dot(X, W1) + b1)
        a2 = sigmoid(np.dot(a1, W2) + b2)
        return a2

    def mse_loss(self, X, y, vec):
        y_pred = self.forward(X, vec)
        return np.mean((y_pred - y)**2)

# =========================================================
# Step 3: Grey Wolf Optimization
# =========================================================
def gwo_optimize(obj_func, dim, pop_size=30, max_iter=200, lb=-1.0, ub=1.0):
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([obj_func(positions[i]) for i in range(pop_size)])
    idx = np.argsort(fitness)
    alpha, beta, delta = positions[idx[:3]]
    alpha_score = fitness[idx[0]]
    fitness_history = [alpha_score]

    for t in range(max_iter):
        a = 2 - 2*(t/(max_iter-1))
        for i in range(pop_size):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1, C1 = 2*a*r1 - a, 2*r2
            D_alpha = np.abs(C1*alpha - positions[i])
            X1 = alpha - A1*D_alpha

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2, C2 = 2*a*r1 - a, 2*r2
            D_beta = np.abs(C2*beta - positions[i])
            X2 = beta - A2*D_beta

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3, C3 = 2*a*r1 - a, 2*r2
            D_delta = np.abs(C3*delta - positions[i])
            X3 = delta - A3*D_delta

            positions[i] = np.clip((X1+X2+X3)/3, lb, ub)

        fitness = np.array([obj_func(positions[i]) for i in range(pop_size)])
        idx = np.argsort(fitness)
        alpha, beta, delta = positions[idx[:3]]
        alpha_score = fitness[idx[0]]
        fitness_history.append(alpha_score)

        if t % 20 == 0 or t == max_iter-1:
            print(f"Iter {t+1}/{max_iter} - Best MSE: {alpha_score:.6f}")

    return alpha, alpha_score, fitness_history

# =========================================================
# Step 4: Choose Dataset
# =========================================================
# Pick one:
# X, y = make_moons_manual(n_samples=200, noise=0.2, random_state=42)
# X, y = make_blobs_manual(n_samples=300, centers=3, random_state=42)
X, y = make_sine_manual(n_samples=200, noise=0.1, random_state=42)

# Plot dataset
plt.scatter(X[:,0], X[:,1] if y.shape[1]>1 else y, c=y.ravel() if len(np.unique(y))>2 else None, cmap="rainbow")
plt.title("Sample Dataset")
plt.show()

# =========================================================
# Step 5: Train with GWO
# =========================================================
n_input = X.shape[1]
n_hidden = 6
n_output = 1  # keep 1 for binary/regression
net = NeuralNetwork(n_input, n_hidden, n_output)

def objective(vec): return net.mse_loss(X, y, vec)

best_vec, best_mse, hist = gwo_optimize(objective, net.num_params, pop_size=40, max_iter=200, lb=-5, ub=5)
print("\nOptimization finished. Best MSE:", best_mse)

# =========================================================
# Step 6: Visualize Results
# =========================================================

y_pred = net.forward(X, best_vec)

if len(np.unique(y)) > 2 and n_input >= 2:
    # --- Multi-class classification (needs at least 2D input) ---
    pred_labels = (y_pred > 0.5).astype(int)  # crude mapping
    plt.scatter(X[:,0], X[:,1], c=pred_labels.ravel(),
                cmap="rainbow", marker="o", edgecolor="k")
    plt.title("Predicted Classes (Multi-class)")

elif n_input == 2:
    # --- Binary classification with 2D inputs ---
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = net.forward(grid, best_vec).reshape(xx.shape)
    plt.contourf(xx, yy, probs, levels=[0,0.5,1], cmap="bwr", alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y.ravel(), cmap="bwr", edgecolor="k")
    plt.title("Decision Boundary (Binary Classification)")

else:
    # --- Regression or 1D input ---
    plt.scatter(X, y, color="blue", s=15, label="True")
    plt.plot(X, y_pred, color="red", label="Prediction")
    plt.title("Regression Fit")
    plt.legend()

plt.show()

# Fitness curve
plt.plot(hist)
plt.title("GWO Optimization Progress (MSE)")
plt.xlabel("Iteration")
plt.ylabel("Best MSE")
plt.grid(True)
plt.show()
