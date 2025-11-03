#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ----------------------
// Variables globales
// ----------------------
int num_examples;
int nn_input_dim;
int nn_output_dim;
double reg_lambda = 0.01;
double epsilon = 0.01;
double *X; // (num_examples × nn_input_dim)
int *y;    // (num_examples)


// ===================================================
// 1. calculate_loss
// ===================================================
double calculate_loss(double *W1, double *b1, double *W2, double *b2, int nn_hdim)
{
    // Allocation des tableaux intermédiaires
    double *z1 = calloc(num_examples * nn_hdim, sizeof(double));
    double *a1 = calloc(num_examples * nn_hdim, sizeof(double));
    double *z2 = calloc(num_examples * nn_output_dim, sizeof(double));
    double *probs = calloc(num_examples * nn_output_dim, sizeof(double));

    // Forward
    matmul(X, W1, z1, num_examples, nn_input_dim, nn_hdim);
    add_bias(z1, b1, num_examples, nn_hdim);
    for (int i = 0; i < num_examples * nn_hdim; i++)
        a1[i] = tanh(z1[i]);
    matmul(a1, W2, z2, num_examples, nn_hdim, nn_output_dim);
    add_bias(z2, b2, num_examples, nn_output_dim);

    // Softmax
    softmax(z2, probs, num_examples, nn_output_dim);

    // Cross-entropy loss
    double data_loss = 0.0;
    for (int i = 0; i < num_examples; i++) {
        int label = y[i];
        double p = probs[i*nn_output_dim + label];
        data_loss += -log(p);
    }

    // Regularisation
    double reg = 0.0;
    for (int i = 0; i < nn_input_dim * nn_hdim; i++)
        reg += W1[i]*W1[i];
    for (int i = 0; i < nn_hdim * nn_output_dim; i++)
        reg += W2[i]*W2[i];
    data_loss += reg_lambda / 2.0 * reg;

    double loss = data_loss / num_examples;

    return loss;
}


// ===================================================
// 2. build_model
// ===================================================
void build_model(int nn_hdim, int num_passes, int print_loss)
{
    srand(0);

    // Initialisation des poids
    double *W1 = malloc(nn_input_dim * nn_hdim * sizeof(double));
    double *b1 = calloc(nn_hdim, sizeof(double));
    double *W2 = malloc(nn_hdim * nn_output_dim * sizeof(double));
    double *b2 = calloc(nn_output_dim, sizeof(double));


    for (int i = 0; i < nn_input_dim * nn_hdim; i++)
        W1[i] = randn() / sqrt(nn_input_dim);
    for (int i = 0; i < nn_hdim * nn_output_dim; i++)
        W2[i] = randn() / sqrt(nn_hdim);

    // ------------------------------
    // Boucle d'entraînement principale
    // ------------------------------
    for (int it = 0; it < num_passes; it++) {

        // ------------------------------
        // Forward propagation
        // ------------------------------
        double *z1 = calloc(num_examples * nn_hdim, sizeof(double));
        double *a1 = calloc(num_examples * nn_hdim, sizeof(double));
        double *z2 = calloc(num_examples * nn_output_dim, sizeof(double));
        double *probs = calloc(num_examples * nn_output_dim, sizeof(double));

        matmul(X, W1, z1, num_examples, nn_input_dim, nn_hdim);
        add_bias(z1, b1, num_examples, nn_hdim);
        for (int i = 0; i < num_examples * nn_hdim; i++)
            a1[i] = tanh(z1[i]);
        matmul(a1, W2, z2, num_examples, nn_hdim, nn_output_dim);
        add_bias(z2, b2, num_examples, nn_output_dim);
        softmax(z2, probs, num_examples, nn_output_dim);

        // ------------------------------
        // Backpropagation
        // ------------------------------
        double *delta3 = malloc(num_examples * nn_output_dim * sizeof(double));
        for (int i = 0; i < num_examples * nn_output_dim; i++)
            delta3[i] = probs[i];
        for (int i = 0; i < num_examples; i++)
            delta3[i*nn_output_dim + y[i]] -= 1.0;

        double *dW2 = calloc(nn_hdim * nn_output_dim, sizeof(double));
        double *db2 = calloc(nn_output_dim, sizeof(double));
        double *delta2 = calloc(num_examples * nn_hdim, sizeof(double));
        double *dW1 = calloc(nn_input_dim * nn_hdim, sizeof(double));
        double *db1 = calloc(nn_hdim, sizeof(double));
	
        // dW2 = a1.T * delta3
        for (int j = 0; j < nn_hdim; j++)
            for (int k = 0; k < nn_output_dim; k++) {
                double sum = 0.0;
                for (int n = 0; n < num_examples; n++)
                    sum += a1[n*nn_hdim + j] * delta3[n*nn_output_dim + k];
                dW2[j*nn_output_dim + k] = sum;
            }

        // db2
        for (int k = 0; k < nn_output_dim; k++) {
            double sum = 0.0;
            for (int n = 0; n < num_examples; n++)
                sum += delta3[n*nn_output_dim + k];
            db2[k] = sum;
        }

        // delta2 = delta3 * W2.T * (1 - a1^2)
        for (int n = 0; n < num_examples; n++)
            for (int j = 0; j < nn_hdim; j++) {
                double sum = 0.0;
                for (int k = 0; k < nn_output_dim; k++)
                    sum += delta3[n*nn_output_dim + k] * W2[j*nn_output_dim + k];
                delta2[n*nn_hdim + j] = sum * (1.0 - a1[n*nn_hdim + j]*a1[n*nn_hdim + j]);
            }

        // dW1 = X.T * delta2
        for (int i = 0; i < nn_input_dim; i++)
            for (int j = 0; j < nn_hdim; j++) {
                double sum = 0.0;
                for (int n = 0; n < num_examples; n++)
                    sum += X[n*nn_input_dim + i] * delta2[n*nn_hdim + j];
                dW1[i*nn_hdim + j] = sum;
            }

        // db1
        for (int j = 0; j < nn_hdim; j++) {
            double sum = 0.0;
            for (int n = 0; n < num_examples; n++)
                sum += delta2[n*nn_hdim + j];
            db1[j] = sum;
        }

        // Regularisation
        for (int i = 0; i < nn_hdim * nn_output_dim; i++)
            dW2[i] += reg_lambda * W2[i];
        for (int i = 0; i < nn_input_dim * nn_hdim; i++)
            dW1[i] += reg_lambda * W1[i];

        // ------------------------------
        // Mise à jour des paramètres (gradient descent)
        // ------------------------------
        for (int i = 0; i < nn_input_dim * nn_hdim; i++)
            W1[i] -= epsilon * dW1[i];
        for (int i = 0; i < nn_hdim; i++)
            b1[i] -= epsilon * db1[i];
        for (int i = 0; i < nn_hdim * nn_output_dim; i++)
            W2[i] -= epsilon * dW2[i];
        for (int i = 0; i < nn_output_dim; i++)
            b2[i] -= epsilon * db2[i];

        // Affichage périodique de la loss
        if (print_loss && it % 1000 == 0) {
            double loss = calculate_loss(W1, b1, W2, b2, nn_hdim);
            printf("Loss après %d itérations: %.6f\n", it, loss);
        }
    }
    
    // Sauvegarde des poids et biais
    FILE *fw1 = fopen("output/W1.txt", "w");
    FILE *fb1 = fopen("output/b1.txt", "w");
    FILE *fw2 = fopen("output/W2.txt", "w");
    FILE *fb2 = fopen("output/b2.txt", "w");

    for (int i = 0; i < nn_input_dim * nn_hdim; i++)
      fprintf(fw1, "%lf\n", W1[i]);
    for (int i = 0; i < nn_hdim; i++)
      fprintf(fb1, "%lf\n", b1[i]);
    for (int i = 0; i < nn_hdim * nn_output_dim; i++)
      fprintf(fw2, "%lf\n", W2[i]);
    for (int i = 0; i < nn_output_dim; i++)
      fprintf(fb2, "%lf\n", b2[i]);

    fclose(fw1); fclose(fb1);
    fclose(fw2); fclose(fb2);
    printf("Poids sauvegardés dans W1.txt, b1.txt, W2.txt, b2.txt\n");

}
