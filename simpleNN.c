#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
// Include my header for all the maths functions... 
// I need to extend to experiment with Gompertz functions (partly implemented) 
// and bicubic splines to provide even more flexibility

#include "nnMaths.h" // My maths functions
#include "nnFileHandler.h" // My somewhat clunky file loader

// Set up the NN layout

#define INPUT_NODES 2
#define HIDDEN_NODES 4
#define OUTPUT_NODES 1
#define LEARNING_RATE 0.1 // how fast will nn learn? But like control systems... if this is too high we can end up with over shoot and oscilations.
#define EPOCHS 10000 // more epocs the better the NN learns stuff 

/* For my reference, (in case I break it.. ) this set of #defines will give me:

#define INPUT_NODES 2
#define HIDDEN_NODES 4
#define OUTPUT_NODES 1
#define LEARNING_RATE 0.1
#define EPOCHS 10000

Input Layer         Hidden Layer         Output Layer
+------------+     +------------+       +-------------+
|  Input 1   | --> | Hidden 1   | ----> |  Output     |
| Two things | --> | Hidden 1   | ----> | One thing   |
+------------+     +------------+       +-------------+
                   | Hidden 2   |
                   +------------+
                   | Hidden 3   |
                   +------------+
                   | Hidden 4   |
                   +------------+

*/



// Struct for weights and biases
typedef struct {
    // Weight - input to hidden
    double weights_ih[INPUT_NODES][HIDDEN_NODES];
    // hidden layer Bias
    double bias_h[HIDDEN_NODES];

    // Weights - hidden to output
    double weights_ho[HIDDEN_NODES][OUTPUT_NODES];
    // Output layer biases
    double bias_o[OUTPUT_NODES];
} NeuralNetwork;

// Initialize weights and biases
void initialize_network(NeuralNetwork *nn) {
    srand(time(NULL)); // Seed the rand for weights

    // Init input-to-hidden weights + biases
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            nn->weights_ih[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Random between -1 and 1
        }
    }
    for (int j = 0; j < HIDDEN_NODES; j++) {
        nn->bias_h[j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    //  Init hidden-to-output weights + biases
    for (int i = 0; i < HIDDEN_NODES; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            nn->weights_ho[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    for (int j = 0; j < OUTPUT_NODES; j++) {
        nn->bias_o[j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

// Forward propagation
void forward_propagate(NeuralNetwork *nn, const double *input, double *hidden_output, double *output) {
    // Calculate hidden layer output
    for (int j = 0; j < HIDDEN_NODES; j++) {
        double sum = nn->bias_h[j];
        for (int i = 0; i < INPUT_NODES; i++) {
            sum += input[i] * nn->weights_ih[i][j];
        }
        hidden_output[j] = sigmoid(sum); // Using sigmoid from nnMaths.h
    }

    // Calculate output
    for (int k = 0; k < OUTPUT_NODES; k++) {
        double sum = nn->bias_o[k];
        for (int j = 0; j < HIDDEN_NODES; j++) {
            sum += hidden_output[j] * nn->weights_ho[j][k];
        }
        output[k] = sigmoid(sum); // Using sigmoid from nnMaths.h
    }
}

// Backpropagation for training the NN (https://www.geeksforgeeks.org/backpropagation-in-neural-network/)
void train_network(NeuralNetwork *nn, const double *input, const double *target) {
    double hidden_output[HIDDEN_NODES];
    double output[OUTPUT_NODES];

    // 1. Forward prop
    forward_propagate(nn, input, hidden_output, output);

    // 2. Calculate output error and delta
    double output_errors[OUTPUT_NODES];
    double output_deltas[OUTPUT_NODES];
    for (int k = 0; k < OUTPUT_NODES; k++) {
        output_errors[k] = target[k] - output[k];
        output_deltas[k] = output_errors[k] * sigmoid_derivative(output[k]); // Using sigmoid_derivative from nnMaths.h
    }

    // 3. Calculate hidden layer error and delta
    double hidden_errors[HIDDEN_NODES];
    double hidden_deltas[HIDDEN_NODES];
    for (int j = 0; j < HIDDEN_NODES; j++) {
        hidden_errors[j] = 0.0;
        for (int k = 0; k < OUTPUT_NODES; k++) {
            hidden_errors[j] += output_deltas[k] * nn->weights_ho[j][k];
        }
        hidden_deltas[j] = hidden_errors[j] * sigmoid_derivative(hidden_output[j]); // Using sigmoid_derivative from nnMaths.h
    }
// Update process
    // 4. hidden-to-output weights and biases - update
    for (int j = 0; j < HIDDEN_NODES; j++) {
        for (int k = 0; k < OUTPUT_NODES; k++) {
            nn->weights_ho[j][k] += LEARNING_RATE * output_deltas[k] * hidden_output[j];
        }
    }
    for (int k = 0; k < OUTPUT_NODES; k++) {
        nn->bias_o[k] += LEARNING_RATE * output_deltas[k];
    }

    // 5. input-to-hidden weights and biases - update
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            nn->weights_ih[i][j] += LEARNING_RATE * hidden_deltas[j] * input[i];
        }
    }
    for (int j = 0; j < HIDDEN_NODES; j++) {
        nn->bias_h[j] += LEARNING_RATE * hidden_deltas[j];
    }
}

int main() {
    NeuralNetwork nn;
    initialize_network(&nn);

    Dataset training_dataset;
    Dataset target_dataset;

    // --- Training Phase ---
    printf("--- Training Phase ---\n");

    // Load training data
    if (load_csv_data("training_data.csv", INPUT_NODES, &training_dataset) != 0) {
        fprintf(stderr, "Failed to load training data. Exiting.\n");
        return 1;
    }

    // Load target data
    if (load_csv_data("target_data.csv", OUTPUT_NODES, &target_dataset) != 0) {
        fprintf(stderr, "Failed to load target data. Exiting.\n");
        free_dataset(&training_dataset); // Free training data if target data loading failed
        return 1;
    }

    // Ensure the number of training examples matches
    if (training_dataset.num_rows != target_dataset.num_rows) {
        fprintf(stderr, "Error: Number of rows in training and target data CSVs do not match.\n");
        free_dataset(&training_dataset);
        free_dataset(&target_dataset);
        return 1;
    }

    printf("Training the neural network...\n");

    // Training loop - now uses dynamically loaded data
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < training_dataset.num_rows; i++) {
            train_network(&nn, training_dataset.data[i], target_dataset.data[i]);
        }
        // Optional: Print error every N epochs
        if (epoch % (EPOCHS / 10) == 0) {
            double total_error = 0.0;
            for (int i = 0; i < training_dataset.num_rows; i++) {
                double hidden_output[HIDDEN_NODES];
                double output[OUTPUT_NODES];
                forward_propagate(&nn, training_dataset.data[i], hidden_output, output);
                total_error += 0.5 * pow(target_dataset.data[i][0] - output[0], 2); // Assumes single output
            }
            printf("Epoch %d, Total Error: %f\n", epoch, total_error);
        }
    }

    printf("\nTraining complete. Testing with training data:\n");

    // Test the trained network with its own training data
    for (int i = 0; i < training_dataset.num_rows; i++) {
        double hidden_output[HIDDEN_NODES];
        double output[OUTPUT_NODES];
        forward_propagate(&nn, training_dataset.data[i], hidden_output, output);
        printf("Input: [%.0f, %.0f], Expected: %.0f, Predicted: %f (Rounded: %.0f)\n",
               training_dataset.data[i][0], training_dataset.data[i][1], target_dataset.data[i][0], output[0], round(output[0]));
    }

    // Free training and target dataset memory
    free_dataset(&training_dataset);
    free_dataset(&target_dataset);

    // --- Inference Phase ---
    printf("\n--- Inference Phase ---\n");

    // We'll directly use "data.csv" as the filename
    const char *inference_filepath = "data.csv"; 

    Dataset inference_dataset;

    // Load inference data
    // The number of columns must match INPUT_NODES
    if (load_csv_data(inference_filepath, INPUT_NODES, &inference_dataset) != 0) {
        fprintf(stderr, "Failed to load inference data from '%s'. Exiting inference phase.\n", inference_filepath);
        return 1; // Exit program if inference data can't be loaded
    }

    printf("\nMaking predictions on new data from '%s':\n", inference_filepath);

    // Iterate through inference data and make predictions
    double hidden_output_inf[HIDDEN_NODES];
    double output_inf[OUTPUT_NODES]; // Array to hold predictions for current input

    for (int i = 0; i < inference_dataset.num_rows; i++) {
        forward_propagate(&nn, inference_dataset.data[i], hidden_output_inf, output_inf);

        // Print input data
        printf("Input: [");
        for (int j = 0; j < INPUT_NODES; j++) {
            printf("%.2f", inference_dataset.data[i][j]);
            if (j < INPUT_NODES - 1) {
                printf(", ");
            }
        }
        printf("], Predicted Output: [");

        // Print predicted output (assuming single output for simplicity in this example)
        for (int k = 0; k < OUTPUT_NODES; k++) {
            printf("%f (Rounded: %.0f)", output_inf[k], round(output_inf[k]));
            if (k < OUTPUT_NODES - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }

    // Free inference dataset memory
    free_dataset(&inference_dataset);

    return 0;
}