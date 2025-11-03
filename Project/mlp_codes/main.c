#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    // Dimensions du réseau
    nn_input_dim = 2;       // deux features
    nn_output_dim = 2;      // deux classes
    int nn_hdim = 10;       // neurones cachés

    // Compte les exemples
    num_examples = count_lines(file_y);
    printf("Chargement de %d échantillons.\n", num_examples);

    // Allocation mémoire
    X = malloc(num_examples * nn_input_dim * sizeof(double));
    y = malloc(num_examples * sizeof(int));

    // Chargement des données
    load_X(file_X, X, num_examples, nn_input_dim);
    load_y(file_y, y, num_examples);
    
    // Entraînement du modèle
    build_model(nn_hdim, 20000, 1);

    // Libération
    free(X);
    free(y);
    return 0;
}
