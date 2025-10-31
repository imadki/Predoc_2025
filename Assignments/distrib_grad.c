#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_FEATURES 5
#define N_SAMPLES 1000
#define LEARNING_RATE 0.01
#define EPOCHS 10000
#define STOP_THRESHOLD 1e-2

// ---------- Structures ----------

typedef struct {
  double x[N_FEATURES];
  double y;
} Sample;

typedef struct {
  double* data;
  int size;
} Vector;

// ---------- Initialization & Free ----------

Vector init_vector(int size) {
  Vector vec;
  vec.size = size;
  vec.data = (double*)malloc(size * sizeof(double));
  for (int i = 0; i < size; i++)
    vec.data[i] = 0.0;
  return vec;
}

void free_vector(Vector* vec) {
  free(vec->data);
  vec->data = NULL;
  vec->size = 0;
}

// ---------- Generating data ----------

void generate_data(Sample* samples, int n) {
  srand(42); // fixed seed
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < N_FEATURES; j++) {
      samples[i].x[j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    samples[i].y = 2.0 * samples[i].x[0] - 1.0 * samples[i].x[1] + 0.1 * ((double)rand() / RAND_MAX);
  }
}

// ---------- Compute the gradient and the loss ----------

double compute_gradient_and_loss(Sample* data, Vector weights, Vector* gradient, int n) {
  for (int j = 0; j < N_FEATURES; j++)
    gradient->data[j] = 0.0;
  
  double loss = 0.0;
  for (int i = 0; i < n; i++) {
    double pred = 0.0;
    for (int j = 0; j < N_FEATURES; j++)
      pred += weights.data[j] * data[i].x[j];
    
    double error = pred - data[i].y;
    loss += error * error;
    
    for (int j = 0; j < N_FEATURES; j++)
      gradient->data[j] += error * data[i].x[j] / n;
  }
  return loss / n; // MSE
}

// ---------- Main ----------

int main() {

  //local_n = 
  
  /* 1. Define an MPI derived type for struct Sample using MPI_Type_create_struct */
  // TODO: create_sample_type(&mpi_sample_type);
  

  /* 2. Generate data on rank 0 and scatter to all processes */
  Sample* data = NULL;
  data = malloc(N_SAMPLES * sizeof(Sample));
  generate_data(data, N_SAMPLES);

  //Sample* local_data = 
  // TODO: MPI_Scatter full_data → local_data using the derived type


  printf("======>here\n"); 
  
  /* 3. Initialize weights on rank 0, then broadcast them to all ranks */
  Vector weights = init_vector(N_FEATURES);
  srand(123); // different seed for the weights
  for (int i = 0; i < N_FEATURES; i++)
    weights.data[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

  // TODO: Broadcast weights.data
  

  Vector grad = init_vector(N_FEATURES);
  
  // Vector local_grad = 

  //startime = 
  for (int epoch = 1; epoch <= EPOCHS; epoch++) {

    /* Change this call by local_data, local_loss and local_grad */
    double loss = compute_gradient_and_loss(data, weights, &grad, N_SAMPLES);

    /* 4. Aggregate gradients and loss across processes */
    // TODO: Communicate the global gradient for gradient
    // TODO: Communicate the loss

    /* 5. Update weights on rank 0 */
    for (int j = 0; j < N_FEATURES; j++)
      weights.data[j] -= LEARNING_RATE * grad.data[j];


    /* Only rank 0 should print this */
    if (epoch%10 == 0)
      printf("Epoch %d | Loss (MSE): %.6f | w[0]: %.4f, w[1]: %.4f\n",
	     epoch, loss, weights.data[0], weights.data[1]);

    /* Early stopping based on loss - Only rank 0 should stop the loop */
    if (loss < STOP_THRESHOLD) {
      printf("Early stopping at epoch %d — loss %.6f < %.1e\n", epoch, loss, STOP_THRESHOLD);
      break;
    }
  }

  //endtime = 
  /* Rank 0 print the Training time */

  
  /* Free all pointers */
  free(data);
  free_vector(&weights);
  free_vector(&grad);

  return 0;
}
