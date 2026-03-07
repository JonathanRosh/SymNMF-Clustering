#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/**
 * Helper to print error and exit.
 */
void error_occurred()
{
    printf("An Error Has Occurred\n");
    exit(1);
}

/**
 * Helper to allocate a 2D matrix.
 */
double **allocate_matrix(int rows, int cols)
{
    double **matrix = (double **)malloc(rows * sizeof(double *));
    int i;
    if (!matrix)
        error_occurred();
    for (i = 0; i < rows; i++)
    {
        matrix[i] = (double *)malloc(cols * sizeof(double));
        if (!matrix[i])
            error_occurred();
    }
    return matrix;
}

/**
 * Helper to free aloccated memory of a matrix.
 */
void free_matrix(double **matrix, int rows)
{
    int i;
    for (i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

/**
 * Analyzes file to find N and d.
 */
void analyze_file(FILE *fp, int *N, int *d)
{
    int c, last_c;
    *N = 0;
    *d = 1;
    while ((c = fgetc(fp)) != EOF)
    {
        last_c = c;
        if (c == '\n')
            (*N)++;
        if (c == ',' && *N == 0)
            (*d)++;
    }
    if (last_c != '\n' && last_c != 0)
        (*N)++;
    if (*N == 0 || *d == 0)
        error_occurred();
}

/**
 * Reads points from file into a 2D array.
 */
double **read_points(char *filename, int *N, int *d)
{
    FILE *fp = fopen(filename, "r");
    char garbage;
    int i, j;
    double **X;
    if (!fp)
        error_occurred();

    analyze_file(fp, N, d);

    X = allocate_matrix(*N, *d);

    rewind(fp);
    for (i = 0; i < (*N); i++)
    {
        for (j = 0; j < (*d); j++)
        {
            if (fscanf(fp, "%lf", &X[i][j]) != 1)
                error_occurred();
            fscanf(fp, "%c", &garbage);
        }
    }
    fclose(fp);
    return X;
}

/**
 * Calculates Squared Euclidean Distance between two vectors.
 */
double sq_euclidean_dist(double *p1, double *p2, int d)
{
    double distance = 0;
    int i;
    for (i = 0; i < d; i++)
    {
        distance += ((p1[i] - p2[i]) * (p1[i] - p2[i]));
    }
    return distance;
}

/**
 * Calculates the Similarity Matrix, A.
 */
double **sym_matrix(double **X, int N, int d)
{
    double **A = allocate_matrix(N, N);
    double dist, val;
    int i, j;
    for (i = 0; i < N; i++)
    {
        A[i][i] = 0;
        for (j = i + 1; j < N; j++)
        {
            dist = sq_euclidean_dist(X[i], X[j], d);
            val = exp((-0.5) * (dist));
            A[i][j] = val;
            A[j][i] = val;
        }
    }
    return A;
}

/**
 * Calculates the Diagonal Degree Matrix, D.
 */
double **ddg_matrix(double **A, int N)
{
    double **D = allocate_matrix(N, N);
    int i, j;
    for (i = 0; i < N; i++)
    {
        D[i][i] = 0;
        for (j = 0; j < N; j++)
        {
            D[i][i] += A[i][j];
            if (i != j)
                D[i][j] = 0;
        }
    }
    return D;
}

/**
 * Calculates the Normalized Similarity Matrix ,W.
 */
double **norm_matrix(double **D, double **A, int N)
{
    double **W = allocate_matrix(N, N);
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (D[i][i] == 0 || D[j][j] == 0)
            {
                W[i][j] = 0;
            }
            else
            {
                W[i][j] = A[i][j] / (sqrt((D[i][i] * D[j][j])));
            }
        }
    }
    return W;
}

/**
 * Multiplies two matrices by dot product.
 */
double **multiply_matrices(double **mat1, int rows1, int cols1, double **mat2, int cols2)
{
    double **matrix = allocate_matrix(rows1, cols2);
    double dot_product;
    int i, j, k;
    for (i = 0; i < rows1; i++)
    {
        for (j = 0; j < cols2; j++)
        {
            dot_product = 0.0;
            for (k = 0; k < cols1; k++)
            {
                dot_product += (mat1[i][k] * mat2[k][j]);
            }

            matrix[i][j] = dot_product;
        }
    }

    return matrix;
}

/**
 * Transposing a given matrix.
 */
double **transpose_matrix(double **matrix, int rows, int cols)
{
    double **t_matrix = allocate_matrix(cols, rows);
    int i, j;
    for (i = 0; i < cols; i++)
    {
        for (j = 0; j < rows; j++)
        {
            t_matrix[i][j] = matrix[j][i];
        }
    }

    return t_matrix;
}

/**
 * Core SymNMF Algorithm logic.
 */
double **symnmf_algorithm(double **W, double **H, int N, int k)
{
    int max_iter = 300;
    double epsilon = 0.0001;
    double **WH, **Ht, **HHt, **HHtH, distance;
    int iter = 0;
    int i, j;
    double **H_new = allocate_matrix(N, k);
    while (iter < max_iter)
    {
        WH = multiply_matrices(W, N, N, H, k);
        Ht = transpose_matrix(H, N, k);
        HHt = multiply_matrices(H, N, k, Ht, N);
        HHtH = multiply_matrices(HHt, N, N, H, k);

        distance = 0.0;
        for (i = 0; i < N; i++)
            for (j = 0; j < k; j++)
            {
                H_new[i][j] = H[i][j] * (0.5 + 0.5 * (WH[i][j] / HHtH[i][j]));
                distance += ((H_new[i][j] - H[i][j]) * (H_new[i][j] - H[i][j]));
            }

        free_matrix(WH, N);
        free_matrix(Ht, k);
        free_matrix(HHt, N);
        free_matrix(HHtH, N);

        for (i = 0; i < N; i++)
            for (j = 0; j < k; j++)
                H[i][j] = H_new[i][j];

        if (distance < epsilon)
            break;

        iter++;
    }
    free_matrix(H_new, N);
    return H;
}

/**
 * Prints a matrix to the console with 4 decimal places.
 */
void print_matrix(double **matrix, int rows, int cols)
{
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            printf("%.4f", matrix[i][j]);
            if (j < cols - 1)
            {
                printf(",");
            }
        }
        printf("\n");
    }
}

/**
 * Main entry point for the pure C executable.
 * Parses CMD arguments, executes the requested matrix calculation
 * (sym, ddg, or norm), prints the result, and safely frees memory.
 */
int main(int argc, char **argv)
{
    int N, d;
    char *goal, *filename;
    double **X, **A, **D, **W;
    if (argc < 3)
        error_occurred();

    goal = argv[1];
    filename = argv[2];

    if (strcmp(goal, "sym") != 0 &&
        strcmp(goal, "ddg") != 0 &&
        strcmp(goal, "norm") != 0)
        error_occurred();

    X = read_points(filename, &N, &d);
    A = sym_matrix(X, N, d);
    free_matrix(X, N);
    if (strcmp(goal, "sym") == 0)
    {
        print_matrix(A, N, N);
        free_matrix(A, N);
        return 0;
    }

    D = ddg_matrix(A, N);
    if (strcmp(goal, "ddg") == 0)
    {
        print_matrix(D, N, N);
        free_matrix(A, N);
        free_matrix(D, N);
        return 0;
    }

    W = norm_matrix(D, A, N);
    print_matrix(W, N, N);
    free_matrix(A, N);
    free_matrix(D, N);
    free_matrix(W, N);
    return 0;
}