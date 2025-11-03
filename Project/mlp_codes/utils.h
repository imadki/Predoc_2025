#ifndef UTILS_H
#define UTILS_H

double randn();
void matmul(double *A, double *B, double *C, int n, int m, int p);
void add_bias(double *Z, double *b, int n, int p);
void softmax(double *Z, double *P, int n, int p);
int count_lines(const char *filename);
void load_X(const char *filename, double *X, int num_examples, int input_dim);
void load_y(const char *filename, int *y, int num_examples);

#endif
