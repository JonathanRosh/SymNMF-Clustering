import sys
import pandas as pd
import numpy as np
import symnmf
import kmeans
from sklearn.metrics import silhouette_score


def error_occurred():
    """Prints a generic error message and terminates the program."""
    print("An Error Has Occurred")
    sys.exit(1)


# ==========================================
# --- 1. PARSE ARGUMENTS & VALIDATION ---
# ==========================================
if len(sys.argv) < 3:
    error_occurred()

k = int(sys.argv[1])
file_name = sys.argv[2]

# Read points from the file into a 2D Python list.
X = pd.read_csv(file_name, header=None).values.tolist()
N = len(X)
d = len(X[0])

# Validate inputs.
if k >= N or k <= 0:
    error_occurred()

# ==========================================
# --- 2. C-EXTENSION PIPELINE ---
# ==========================================

# Calculate the sym, ddg and norm matrices.
A = symnmf.sym(X, N, d)
D = symnmf.ddg(A, N)
W = symnmf.norm(D, A, N)

# Calculate the symnmf matrix.
m = np.mean(W)
np.random.seed(1234)
lim = 2 * np.sqrt(m / k)
H = np.random.uniform(0, lim, size=(N, k)).tolist()
final_H = symnmf.symnmf(W, H, N, k)


# ==========================================
# --- 3. CLUSTERING & SCORING ---
# ==========================================

# Derive labels from the symnmf matrix.
symnmf_labels = [row.index(max(row)) for row in final_H]

# Run K-means.
kmeans_labels = kmeans.run_kmeans(X, k)

# Calculate the silhouette scores.
symnmf_score = silhouette_score(X, symnmf_labels)
kmeans_score = silhouette_score(X, kmeans_labels)

print(f"nmf: {symnmf_score:.4f}")
print(f"kmeans: {kmeans_score:.4f}")
