import sys
import pandas as pd
import numpy as np
import symnmf


def error_occurred():
    """Prints a generic error message and terminates the program."""
    print("An Error Has Occurred")
    sys.exit(1)


def print_matrix(matrix):
    """
    Prints a 2D matrix with each element formatted to 4 decimal places,
    separated by commas.
    """
    for row in matrix:
        print(",".join(["{:.4f}".format(num) for num in row]))


# ==========================================
# --- 1. PARSE ARGUMENTS & VALIDATION ---
# ==========================================
if len(sys.argv) < 4:
    error_occurred()

try:
    k = int(sys.argv[1])
except ValueError:
    error_occurred()

goal = sys.argv[2]
file_name = sys.argv[3]

# Read points from the file into a 2D Python list.
X = pd.read_csv(file_name, header=None).values.tolist()
N = len(X)
d = len(X[0])

# Validate inputs.
if k >= N or k <= 0 or goal not in ["sym", "ddg", "norm", "symnmf"]:
    error_occurred()


# ==========================================
# --- 2. C-EXTENSION PIPELINE ---
# ==========================================

# Calculate the sym matrix.
A = symnmf.sym(X, N, d)
if goal == "sym":
    print_matrix(A)
    sys.exit(0)

# Calculate the ddg matrix.
D = symnmf.ddg(A, N)
if goal == "ddg":
    print_matrix(D)
    sys.exit(0)

# Calculate the norm matrix.
W = symnmf.norm(D, A, N)
if goal == "norm":
    print_matrix(W)
    sys.exit(0)


# Calculate the symnmf matrix.
m = np.mean(W)
np.random.seed(1234)
lim = 2 * np.sqrt(m / k)
H = np.random.uniform(0, lim, size=(N, k)).tolist()
if goal == "symnmf":
    final_H = symnmf.symnmf(W, H, N, k)
    print_matrix(final_H)
    sys.exit(0)
