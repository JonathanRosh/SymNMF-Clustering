# 🧬 Symmetric Non-Negative Matrix Factorization (SymNMF)

A clustering project implementing the Symmetric Non-Negative Matrix Factorization (SymNMF) algorithm. This implementation features a performance-focused C engine for matrix operations and a Python interface for data processing and analysis.

## ✨ Features

- ⚙️ **C Engine:** Efficient implementation of similarity, degree, and normalized Laplacian matrices.
- 📈 **Optimization:** Iterative update of the association matrix H until convergence or max iterations (300).
- 🔌 **Python C-API:** Custom bridge allowing Python to utilize C-based matrix calculations.
- 📊 **Comparative Analysis:** Built-in tools to compare SymNMF results with K-Means using the Silhouette Score.
- 🛡️ **Memory Safety:** Strict heap management with zero memory leaks (verified via Valgrind).

## 🛠️ Build and Installation

### C Executable

Compile the pure C interface using the provided Makefile:
`make`

### Python Module

Build the C-extension for use in Python scripts:
`python3 setup.py build_ext --inplace`

## 🚀 Usage

### symnmf.py

The main Python interface for executing the algorithm:
`python3 symnmf.py <k> <goal> <file_name.txt>`

- **Goals:** `symnmf`, `sym`, `ddg`, `norm`.

### symnmf (C Executable)

For direct matrix calculations from the terminal:
`./symnmf <goal> <file_name.txt>`

- **Goals:** `sym`, `ddg`, `norm`.

### analysis.py

Runs both SymNMF and K-Means and reports Silhouette Scores:
`python3 analysis.py <k> <file_name.txt>`

## 📂 Project Structure

- 📄 `symnmf.c` / `symnmf.h`: Core C implementation and header.
- 🔗 `symnmfmodule.c`: Python C-API wrapper.
- 🐍 `symnmf.py`: Python driver and H matrix initialization.
- 🧪 `analysis.py`: Comparison script using sklearn.metrics.
- 🔨 `Makefile` / `setup.py`: Build scripts.

## 👨‍💻 Author

**Yonatan Rosh**
Computer Science, Tel Aviv University
