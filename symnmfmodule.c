#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"
#include <stdlib.h>

/**
 * Converts Python matrix into C matrix.
 */
double **py_to_c_matrix(PyObject *py_matrix, int rows, int cols)
{
    double **c_matrix = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        PyObject *row = PyList_GetItem(py_matrix, i);
        for (int j = 0; j < cols; j++)
        {
            PyObject *item = PyList_GetItem(row, j);
            c_matrix[i][j] = PyFloat_AsDouble(item);
        }
    }
    return c_matrix;
}

/**
 * Converts C matrix into Python matrix.
 */
PyObject *c_to_py_matrix(double **c_matrix, int rows, int cols)
{
    PyObject *py_matrix = PyList_New(rows);
    for (int i = 0; i < rows; i++)
    {
        PyObject *py_row = PyList_New(cols);
        PyList_SetItem(py_matrix, i, py_row);
        for (int j = 0; j < cols; j++)
        {
            PyObject *item = PyFloat_FromDouble(c_matrix[i][j]);
            PyList_SetItem(py_row, j, item);
        }
    }
    return py_matrix;
}

/**
 * Python wrapper for sym_matrix.
 * Expects (X, N, d) and returns the Similarity Matrix, A.
 */
static PyObject *wrap_sym(PyObject *self, PyObject *args)
{
    PyObject *py_X;
    int N;
    int d;

    if (!PyArg_ParseTuple(args, "Oii", &py_X, &N, &d))
        return NULL;

    double **c_X = py_to_c_matrix(py_X, N, d);
    double **c_A = sym_matrix(c_X, N, d);
    PyObject *py_A = c_to_py_matrix(c_A, N, N);
    free_matrix(c_X, N);
    free_matrix(c_A, N);
    return py_A;
}

/**
 * Python wrapper for ddg_matrix.
 * Expects (A, N) and returns the Diagonal Degree Matrix ,D.
 */
static PyObject *wrap_ddg(PyObject *self, PyObject *args)
{
    PyObject *py_A;
    int N;

    if (!PyArg_ParseTuple(args, "Oi", &py_A, &N))
        return NULL;

    double **c_A = py_to_c_matrix(py_A, N, N);
    double **c_D = ddg_matrix(c_A, N);
    PyObject *py_D = c_to_py_matrix(c_D, N, N);
    free_matrix(c_A, N);
    free_matrix(c_D, N);
    return py_D;
}

/**
 * Python wrapper for norm_matrix.
 * Expects (D, A, N) and returns the Normalized Similarity Matrix ,W.
 */
static PyObject *wrap_norm(PyObject *self, PyObject *args)
{
    PyObject *py_D;
    PyObject *py_A;
    int N;

    if (!PyArg_ParseTuple(args, "OOi", &py_D, &py_A, &N))
        return NULL;

    double **c_A = py_to_c_matrix(py_A, N, N);
    double **c_D = py_to_c_matrix(py_D, N, N);
    double **c_W = norm_matrix(c_D, c_A, N);
    PyObject *py_W = c_to_py_matrix(c_W, N, N);
    free_matrix(c_A, N);
    free_matrix(c_D, N);
    free_matrix(c_W, N);
    return py_W;
}

/**
 * Python wrapper for the core symnmf_algorithm.
 * Expects (W, H, N, k) and returns the optimized H matrix.
 */
static PyObject *wrap_symnmf(PyObject *self, PyObject *args)
{
    PyObject *py_W;
    PyObject *py_H;
    int N;
    int k;

    if (!PyArg_ParseTuple(args, "OOii", &py_W, &py_H, &N, &k))
        return NULL;

    double **c_W = py_to_c_matrix(py_W, N, N);
    double **c_H = py_to_c_matrix(py_H, N, k);
    double **c_Final_H = symnmf_algorithm(c_W, c_H, N, k);
    PyObject *py_Final_H = c_to_py_matrix(c_Final_H, N, k);
    free_matrix(c_W, N);
    free_matrix(c_H, N);
    return py_Final_H;
}

static PyMethodDef symnmfMethods[] = {
    {"sym", wrap_sym, METH_VARARGS, "Calculate Similarity Matrix"},
    {"ddg", wrap_ddg, METH_VARARGS, "Calculate Diagonal Degree Matrix"},
    {"norm", wrap_norm, METH_VARARGS, "Calculate Normalized Similarity Matrix"},
    {"symnmf", wrap_symnmf, METH_VARARGS, "Calculate optimized H matrix"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf", // Name of the Python module
    NULL,
    -1,
    symnmfMethods};

PyMODINIT_FUNC PyInit_symnmf(void)
{
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m)
    {
        return NULL;
    }
    return m;
}