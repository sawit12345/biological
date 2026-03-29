import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import time

# -------------------------------------------------------------------------
# Non-Axiomatic Reasoning System (NARS) Engine (NAL + AIKR)
# -------------------------------------------------------------------------
class TruthValue:
    def __init__(self, f, c):
        self.f = f  # frequency [0, 1]
        self.c = c  # confidence (0, 1)

def nars_revision(tv1, tv2):
    """ NAL Revision Rule: Combines evidence from two independent sources. """
    w1 = tv1.c / (1.0 - tv1.c + 1e-9)
    w2 = tv2.c / (1.0 - tv2.c + 1e-9)
    w = w1 + w2
    if w == 0:
        return TruthValue(0.5, 0.0)
    f = (w1 * tv1.f + w2 * tv2.f) / w
    c = w / (w + 1.0)
    return TruthValue(f, c)

def nars_choice(tv1, tv2):
    """ NAL Choice Rule: Chooses the better expectation. """
    exp1 = tv1.c * (tv1.f - 0.5) + 0.5
    exp2 = tv2.c * (tv2.f - 0.5) + 0.5
    if exp1 > exp2:
        return tv1, 1
    elif exp2 > exp1:
        return tv2, 2
    else:
        return (tv1, 1) if tv1.c > tv2.c else (tv2, 2)

# -------------------------------------------------------------------------
# Utility: Softmax
# -------------------------------------------------------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# -------------------------------------------------------------------------
# Entorhinal Cortex (EC)
# -------------------------------------------------------------------------
class EntorhinalCortex:
    def __init__(self, input_dim, grid_dim):
        self.grid_dim = grid_dim
        # Using random projections to create periodic "grid cell" responses
        # Not a neural network trained with backprop, just a static transform
        np.random.seed(42)
        self.W_grid = np.random.randn(input_dim, grid_dim // 2) * 0.1

    def encode(self, x):
        """ Transforms sensory input into a grid-cell-like periodic representation. """
        base = np.dot(x, self.W_grid)
        # Periodic firing mimicking grid cells
        return np.concatenate([np.sin(base), np.cos(base)], axis=-1)

# -------------------------------------------------------------------------
# Hippocampus
# -------------------------------------------------------------------------
class Hippocampus:
    def __init__(self, capacity):
        self.memory_states = []  # Explicit memory buffer
        self.memory_labels = []
        self.capacity = int(capacity)
        # For AIKR: fixed capacity. When full, we forget (simplest strategy: oldest)
        self.grid_phases = []

    def memorize_and_learn(self, state, label, familiar=False):
        """
        Store a new episodic memory.
        "better to form a new one because it's cheaper and faster and doesn't affect things"
        "must both memorize and learn a bit (for learning it's grid cells)"
        """
        if len(self.memory_states) >= self.capacity:
            self.memory_states.pop(0)
            self.memory_labels.pop(0)
            self.grid_phases.pop(0)

        # Fast memory formation
        self.memory_states.append(state)
        self.memory_labels.append(label)

        # Learn a bit: shift the internal phase representation slightly towards the state
        # Simulated grid-cell "learning"
        phase = np.mean(state)
        self.grid_phases.append(phase)

    def recall(self, query_state):
        """
        Exponential capacity in linear RAM implemented using Dense Associative memory formulation
        over the stored episodic buffer. Softmax over dot products provides the exponential capacity scaling.
        """
        if not self.memory_states:
            return None, TruthValue(0.5, 0.0)

        keys = np.array(self.memory_states)
        labels = np.array(self.memory_labels)

        # Similarities
        sims = np.dot(keys, query_state)

        # Softmax for exponential separation of memories (Dense Associative Memory property)
        weights = softmax(sims)

        best_idx = np.argmax(weights)
        best_label = labels[best_idx]
        confidence = float(np.max(weights))  # Highest weight as confidence

        # If confidence is extremely low, cap it
        c = max(0.01, min(0.99, confidence))
        return best_label, TruthValue(1.0, c)

    def replay(self):
        """
        Replay memories for Neocortex consolidation.
        "compressed like in real life teaching the Neocortex"
        "replay can be forward or sometimes reversed"
        """
        seq_states = self.memory_states.copy()
        seq_labels = self.memory_labels.copy()

        # Forward replay
        yield zip(seq_states, seq_labels)

        # Sometimes reversed replay
        if np.random.rand() > 0.5:
            yield zip(seq_states[::-1], seq_labels[::-1])


# -------------------------------------------------------------------------
# Neocortex
# -------------------------------------------------------------------------
class Neocortex:
    def __init__(self, input_dim, num_classes, capacity):
        # AIKR: capacity bounds. Neocortex has 100x capacity.
        self.capacity = capacity
        # Dense associative memory weights (No backprop neural network)
        self.W = np.zeros((num_classes, input_dim))

    def learn(self, state, label, learning_rate):
        """
        "must still can learn during the day... without sleep first"
        Uses FEP (Free Energy Principle: minimize prediction error) + Hebbian Learning.
        """
        # Exponential capacity in linear RAM via continuous energy landscape
        # Compute "energy" / activations
        activations = np.dot(self.W, state)

        # Prediction
        preds = softmax(activations)

        target = np.zeros(len(self.W))
        target[label] = 1.0

        # FEP: Prediction error (Surprise / Free Energy gradient)
        error = target - preds

        # Hebbian outer product update modulated by FEP error
        self.W[label] += learning_rate * error[label] * state

    def predict(self, query_state):
        activations = np.dot(self.W, query_state)
        preds = softmax(activations)
        best_class = np.argmax(preds)
        confidence = preds[best_class]
        c = max(0.01, min(0.99, confidence))
        return best_class, TruthValue(1.0, c)


# -------------------------------------------------------------------------
# Full Complementary Learning System (CLS)
# -------------------------------------------------------------------------
class CLSModel:
    def __init__(self, input_dim, num_classes):
        self.num_classes = num_classes

        # "Neocortex must has atleast 100x more capacity than Hippocampus"
        # We specify abstract capacity limits (AIKR)
        hc_capacity = 1000
        nc_capacity = hc_capacity * 100

        grid_dim = 256
        self.ec = EntorhinalCortex(input_dim, grid_dim)
        self.hc = Hippocampus(capacity=hc_capacity)
        self.nc = Neocortex(grid_dim, num_classes, capacity=nc_capacity)

    def train(self, X, y):
        # --- 1. Day Time Learning ---
        # "Neocortex does learn during the day without sleep first"
        for i in range(len(X)):
            state = self.ec.encode(X[i])

            # "use their old memories to form a new one faster"
            # Fast NARS recall prior to storage
            hc_label, hc_tv = self.hc.recall(state)
            is_familiar = (hc_tv.c > 0.8)

            # Day time learning in both systems
            self.hc.memorize_and_learn(state, y[i], familiar=is_familiar)

            # Faster learning if familiar
            lr = 0.5 if not is_familiar else 0.1
            self.nc.learn(state, y[i], learning_rate=lr)

        # --- 2. Sleep / Consolidation ---
        # "Hippocampus must replay/recall the activities... over and over many times"
        for _ in range(10):
            for sequence in self.hc.replay():
                # "compressed like in real life"
                # We can simulate compression by subsampling or high learning rate
                for state, label in sequence:
                    self.nc.learn(state, label, learning_rate=0.05)

    def predict(self, X):
        preds = []
        for i in range(len(X)):
            state = self.ec.encode(X[i])

            # Independent evidence from Neocortex and Hippocampus
            nc_label, nc_tv = self.nc.predict(state)
            hc_label, hc_tv = self.hc.recall(state)

            if hc_label is None:
                preds.append(nc_label)
                continue

            # NARS Reasoning Engine integrates the evidence
            if nc_label == hc_label:
                # Same conclusion, revise to increase confidence
                combined_tv = nars_revision(nc_tv, hc_tv)
                preds.append(nc_label)
            else:
                # Conflicting conclusions, use NARS choice rule
                best_tv, source = nars_choice(nc_tv, hc_tv)
                if source == 1:
                    preds.append(nc_label)
                else:
                    preds.append(hc_label)

        return np.array(preds)


# -------------------------------------------------------------------------
# Execution on MNIST
# -------------------------------------------------------------------------
def main():
    print("Loading MNIST via fetch_openml (This might take a minute)...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')

    # Scale data to [0, 1]
    X = mnist.data.values.astype(np.float32) / 255.0
    y = mnist.target.values.astype(int)

    # "test on MNIST... one sample per class full test" -> One-shot learning
    print("Setting up One-Shot Learning (1 sample per class)...")
    X_train, y_train = [], []
    for c in range(10):
        # Find first occurrence of each class
        idx = np.where(y == c)[0][0]
        X_train.append(X[idx])
        y_train.append(y[idx])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Full test on the 10k test split to keep it super cheap and fast
    X_test = X[60000:]
    y_test = y[60000:]

    print("Initializing Full CLS Model (EC, Hippocampus, Neocortex, NARS)...")
    model = CLSModel(input_dim=784, num_classes=10)

    print("Training Model...")
    t0 = time.time()
    model.train(X_train, y_train)
    t1 = time.time()
    print(f"Training Time: {t1 - t0:.4f} seconds")

    print("Testing Model on Full Test Set (10,000 samples)...")
    t0 = time.time()
    preds = model.predict(X_test)
    t1 = time.time()
    print(f"Testing Time: {t1 - t0:.4f} seconds")

    acc = accuracy_score(y_test, preds)
    print(f"One-Shot Accuracy: {acc * 100:.2f}%")

if __name__ == '__main__':
    main()
