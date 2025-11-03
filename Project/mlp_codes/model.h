#ifndef MODEL_H
#define MODEL_H

// ----------------------
// Variables globales
// ----------------------
extern int num_examples;
extern int nn_input_dim;
extern int nn_output_dim;
extern double reg_lambda;
extern double epsilon;
extern double *X;
extern int *y;

// Fonctions principales
double calculate_loss(double *W1, double *b1, double *W2, double *b2, int nn_hdim);
void build_model(int nn_hdim, int num_passes, int print_loss);

#endif
