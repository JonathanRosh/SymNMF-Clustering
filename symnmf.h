#ifndef SYMNMF_H
#define SYMNMF_H

double **allocate_matrix(int rows, int cols);
double **sym_matrix(double **X, int N, int D);
double **ddg_matrix(double **A, int N);
double **norm_matrix(double **D, double **A, int N);
double **symnmf_algorithm(double **W, double **H, int N, int k);
void free_matrix(double **matrix, int rows);

#endif