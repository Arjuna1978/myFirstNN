#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "nnMaths.h" // My maths functions
#include "nnFileHandler.h" // My somewhat clunky file loader

// --- General Neural Network Configuration ---
#define INPUT_NODES 2
#define OUTPUT_NODES 1
#define MAX_HIDDEN_LAYERS 3 // Maximum number of hidden layers supported by the architecture (for array sizing)
#define NUM_HIDDEN_LAYERS 1 // Actual number of hidden layers to use in this network (must be <= MAX_HIDDEN_LAYERS)

// --- Hidden layer Configuration ---

#define HIDDEN_NODES_LAYER_0 4 // Set to 0 if not used or specific value if used
#define HIDDEN_NODES_LAYER_1 4 // Set to 0 if not used or specific value if used
#define HIDDEN_NODES_LAYER_2 4 // Set to 0 if not used or specific value if used

// Array to hold the actual sizes of the active hidden layers
// This is initialized in main based on the above HIDDEN_NODES_LAYER_X defines
static int hidden_layer_sizes[MAX_HIDDEN_LAYERS];

// Macro to find the largest layer and set that as a fixed size to make memory allocation easier... there may be a better way... not sure
#define MAX_NODES_PER_HIDDEN_LAYER \
    (HIDDEN_NODES_LAYER_0 > HIDDEN_NODES_LAYER_1 ? \
        (HIDDEN_NODES_LAYER_0 > HIDDEN_NODES_LAYER_2 ? HIDDEN_NODES_LAYER_0 : HIDDEN_NODES_LAYER_2) : \
        (HIDDEN_NODES_LAYER_1 > HIDDEN_NODES_LAYER_2 ? HIDDEN_NODES_LAYER_1 : HIDDEN_NODES_LAYER_2))

#define LEARNING_RATE 0.1
#define EPOCHS 10000

/*
Example Architecture (if NUM_HIDDEN_LAYERS = 2, HIDDEN_NODES_LAYER_0 = 4, HIDDEN_NODES_LAYER_1 = 3):

Input Layer         Hidden Layer 0         Hidden Layer 1         Output Layer
+------------+     +------------+         +------------+       +-------------+
|  Input 1   | --> | Hidden 0_1 | ------> | Hidden 1_1 | ----> |  Output     |
|  Input 2   | --> | Hidden 0_2 | ------> | Hidden 1_2 | ----> |  Output     |
+------------+     +------------+         +------------+       +-------------+
                   | Hidden 0_3 |           | Hidden 1_3 |
                   +------------+           +------------+
                   | Hidden 0_4 |
                   +------------+
*/

// Struct for weights and biases
typedef struct {
    // Weights  input --> 1st hidden layer
    double weights_ih[INPUT_NODES][MAX_NODES_PER_HIDDEN_LAYER];
    // Biases for each hidden layer (MAX_HIDDEN_LAYERS x max_nodes_in_any_hidden_layer)
    double bias_h[MAX_HIDDEN_LAYERS][MAX_NODES_PER_HIDDEN_LAYER];

    // Weights hidden layer --> hidden layer
    // MAX_HIDDEN_LAYERS-1 because there are one fewer sets of inter-hidden weights than layers
    double weights_hh[MAX_HIDDEN_LAYERS - 1][MAX_NODES_PER_HIDDEN_LAYER][MAX_NODES_PER_HIDDEN_LAYER];

    // Weights Lst hidden layer -->  output
    double weights_ho[MAX_NODES_PER_HIDDEN_LAYER][OUTPUT_NODES];

    // Output layer biases
    double bias_o[OUTPUT_NODES];
} NeuralNetwork;

// Initialize weights and biases for all layers
void initialize_network(NeuralNetwork *nn) {
    srand(time(NULL)); // Seed the random number generator

    // Initialize Input to First Hidden Layer weights and biases
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < hidden_layer_sizes[0]; j++) {
            nn->weights_ih[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    for (int j = 0; j < hidden_layer_sizes[0]; j++) {
        nn->bias_h[0][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    // Initialize weights and biases for subsequent hidden layers
    for (int l = 0; l < NUM_HIDDEN_LAYERS - 1; l++) { // Loop for inter-hidden layer weights
        for (int i = 0; i < hidden_layer_sizes[l]; i++) {
            for (int j = 0; j < hidden_layer_sizes[l+1]; j++) {
                nn->weights_hh[l][i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }
        }
        for (int j = 0; j < hidden_layer_sizes[l+1]; j++) {
            nn->bias_h[l+1][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }

    // Initialize Last Hidden Layer to Output weights and biases
    for (int i = 0; i < hidden_layer_sizes[NUM_HIDDEN_LAYERS - 1]; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            nn->weights_ho[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    for (int j = 0; j < OUTPUT_NODES; j++) {
        nn->bias_o[j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

// Forward propagation for a multi-layered network
void forward_propagate(NeuralNetwork *nn, const double *input,
                       double hidden_layer_outputs[MAX_HIDDEN_LAYERS][MAX_NODES_PER_HIDDEN_LAYER],
                       double *output) {

    // Calculate first hidden layer output
    double current_input[INPUT_NODES];
    for (int i = 0; i < INPUT_NODES; i++) {
        current_input[i] = input[i];
    }

    // From input to first hidden layer
    for (int j = 0; j < hidden_layer_sizes[0]; j++) {
        double sum = nn->bias_h[0][j];
        for (int i = 0; i < INPUT_NODES; i++) {
            sum += current_input[i] * nn->weights_ih[i][j];
        }
        hidden_layer_outputs[0][j] = sigmoid(sum);
    }

    // Through subsequent hidden layers
    for (int l = 0; l < NUM_HIDDEN_LAYERS - 1; l++) {
        for (int j = 0; j < hidden_layer_sizes[l+1]; j++) {
            double sum = nn->bias_h[l+1][j];
            for (int i = 0; i < hidden_layer_sizes[l]; i++) {
                sum += hidden_layer_outputs[l][i] * nn->weights_hh[l][i][j];
            }
            hidden_layer_outputs[l+1][j] = sigmoid(sum);
        }
    }

    // From last hidden layer to output layer
    for (int k = 0; k < OUTPUT_NODES; k++) {
        double sum = nn->bias_o[k];
        for (int j = 0; j < hidden_layer_sizes[NUM_HIDDEN_LAYERS - 1]; j++) {
            sum += hidden_layer_outputs[NUM_HIDDEN_LAYERS - 1][j] * nn->weights_ho[j][k];
        }
        output[k] = sigmoid(sum);
    }
}

// Backpropagation to train  NN
void train_network(NeuralNetwork *nn, const double *input, const double *target) {
    double hidden_layer_outputs[MAX_HIDDEN_LAYERS][MAX_NODES_PER_HIDDEN_LAYER];
    double output[OUTPUT_NODES];

    // 1. Forward propagate
    forward_propagate(nn, input, hidden_layer_outputs, output);

    // 2. Calculate output layer error and delta
    double output_errors[OUTPUT_NODES];
    double output_deltas[OUTPUT_NODES];
    for (int k = 0; k < OUTPUT_NODES; k++) {
        output_errors[k] = target[k] - output[k];
        output_deltas[k] = output_errors[k] * sigmoid_derivative(output[k]);
    }

    // 3. Calculate hidden layer errors and deltas (backwards)
   
    // Error & Delta fo last hidden layer
    double hidden_errors[MAX_HIDDEN_LAYERS][MAX_NODES_PER_HIDDEN_LAYER];
    double hidden_deltas[MAX_HIDDEN_LAYERS][MAX_NODES_PER_HIDDEN_LAYER];

    // For Lst hidden layer --> Output
    for (int j = 0; j < hidden_layer_sizes[NUM_HIDDEN_LAYERS - 1]; j++) {
        hidden_errors[NUM_HIDDEN_LAYERS - 1][j] = 0.0;
        for (int k = 0; k < OUTPUT_NODES; k++) {
            hidden_errors[NUM_HIDDEN_LAYERS - 1][j] += output_deltas[k] * nn->weights_ho[j][k];
        }
        hidden_deltas[NUM_HIDDEN_LAYERS - 1][j] = hidden_errors[NUM_HIDDEN_LAYERS - 1][j] * sigmoid_derivative(hidden_layer_outputs[NUM_HIDDEN_LAYERS - 1][j]);
    }

    // Hidden Layer <---> Hidden Layer---> 1st layer to calculate dltats and errors
    for (int l = NUM_HIDDEN_LAYERS - 2; l >= 0; l--) { 
        for (int j = 0; j < hidden_layer_sizes[l]; j++) {
            hidden_errors[l][j] = 0.0;
            for (int k = 0; k < hidden_layer_sizes[l+1]; k++) {
                hidden_errors[l][j] += hidden_deltas[l+1][k] * nn->weights_hh[l][j][k];
            }
            hidden_deltas[l][j] = hidden_errors[l][j] * sigmoid_derivative(hidden_layer_outputs[l][j]);
        }
    }

    // Update weights & biases

    // 4. Update Lst Hidden Layer --> Output weights & biases
    for (int j = 0; j < hidden_layer_sizes[NUM_HIDDEN_LAYERS - 1]; j++) {
        for (int k = 0; k < OUTPUT_NODES; k++) {
            nn->weights_ho[j][k] += LEARNING_RATE * output_deltas[k] * hidden_layer_outputs[NUM_HIDDEN_LAYERS - 1][j];
        }
    }
    for (int k = 0; k < OUTPUT_NODES; k++) {
        nn->bias_o[k] += LEARNING_RATE * output_deltas[k];
    }

    // 5. Update weights and biases Hidden Layer <---> Hidden Layer
    for (int l = NUM_HIDDEN_LAYERS - 2; l >= 0; l--) {
        for (int i = 0; i < hidden_layer_sizes[l]; i++) {
            for (int j = 0; j < hidden_layer_sizes[l+1]; j++) {
                nn->weights_hh[l][i][j] += LEARNING_RATE * hidden_deltas[l+1][j] * hidden_layer_outputs[l][i];
            }
        }
        for (int j = 0; j < hidden_layer_sizes[l+1]; j++) {
            nn->bias_h[l+1][j] += LEARNING_RATE * hidden_deltas[l+1][j];
        }
    }

    // 6. Update Input <---> 1st Layer weights and biases
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < hidden_layer_sizes[0]; j++) {
            nn->weights_ih[i][j] += LEARNING_RATE * hidden_deltas[0][j] * input[i];
        }
    }
    for (int j = 0; j < hidden_layer_sizes[0]; j++) {
        nn->bias_h[0][j] += LEARNING_RATE * hidden_deltas[0][j];
    }
}

int main() {
    // Initialize hidden_layer
    hidden_layer_sizes[0] = HIDDEN_NODES_LAYER_0;
    if (MAX_HIDDEN_LAYERS > 1) hidden_layer_sizes[1] = HIDDEN_NODES_LAYER_1;
    if (MAX_HIDDEN_LAYERS > 2) hidden_layer_sizes[2] = HIDDEN_NODES_LAYER_2;

    // Validate NUM_HIDDEN_LAYERS
    if (NUM_HIDDEN_LAYERS > MAX_HIDDEN_LAYERS) {
        fprintf(stderr, "Error: NUM_HIDDEN_LAYERS (%d) cannot be greater than MAX_HIDDEN_LAYERS (%d).\n", NUM_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS);
        return 1;
    }
    if (NUM_HIDDEN_LAYERS < 1) { // A network typically needs at least one hidden layer to be 'hidden'
        fprintf(stderr, "Error: NUM_HIDDEN_LAYERS must be at least 1.\n");
        return 1;
    }


    NeuralNetwork nn;
    initialize_network(&nn);

    Dataset training_dataset;
    Dataset target_dataset;

    // --- Training Phase ---
    printf("--- Starting  Training Phase ---\n");

     printf("--- Laoding Input side Training Data  ---\n");

    // Load training data
    if (load_csv_data("training_data.csv", INPUT_NODES, &training_dataset) != 0) {
        fprintf(stderr, "Failed to load training data. Exiting.\n");
        return 1;
    }

    printf("--- Loading output side Training Phase ---\n");
    // Load target data
    if (load_csv_data("target_data.csv", OUTPUT_NODES, &target_dataset) != 0) {
        fprintf(stderr, "Failed to load target data. Exiting.\n");
        free_dataset(&training_dataset);
        return 1;
    }

    printf("--- Checking data sizes to match the NN---\n");
    // Ensure the number of training examples matches
    if (training_dataset.num_rows != target_dataset.num_rows) {
        fprintf(stderr, "Error: Number of rows in training and target data CSVs do not match.\n");
        free_dataset(&training_dataset);
        free_dataset(&target_dataset);
        return 1;
    }

    printf("Starting training the neural network...\n");

    // Training loop - now uses dynamically loaded data
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
         printf("Epoch %d, training started \n", epoch);
        for (int i = 0; i < training_dataset.num_rows; i++) {
            train_network(&nn, training_dataset.data[i], target_dataset.data[i]);
        }
        // Optional: Print error every N epochs
        if (epoch % (EPOCHS / 10) == 0) {
            double total_error = 0.0;
            double temp_hidden_outputs[MAX_HIDDEN_LAYERS][MAX_NODES_PER_HIDDEN_LAYER];
            double temp_output[OUTPUT_NODES];
            for (int i = 0; i < training_dataset.num_rows; i++) {
                forward_propagate(&nn, training_dataset.data[i], temp_hidden_outputs, temp_output);
                for (int k = 0; k < OUTPUT_NODES; k++) {
                    total_error += 0.5 * pow(target_dataset.data[i][k] - temp_output[k], 2);
                }
            }
            printf("Epoch %d, Total Error: %f\n", epoch, total_error);
        }
    }

    printf("\nTraining complete. Testing with training data:\n");

    // Test the trained network with its own training data
    for (int i = 0; i < training_dataset.num_rows; i++) {
        double temp_hidden_outputs[MAX_HIDDEN_LAYERS][MAX_NODES_PER_HIDDEN_LAYER];
        double temp_output[OUTPUT_NODES];
        forward_propagate(&nn, training_dataset.data[i], temp_hidden_outputs, temp_output);

        printf("Input: [");
        for (int j = 0; j < training_dataset.num_cols; j++) {
            printf("%.0f", training_dataset.data[i][j]);
            if (j < training_dataset.num_cols - 1) {
                printf(", ");
            }
        }
        printf("], Expected: [");
        for (int k = 0; k < target_dataset.num_cols; k++) {
            printf("%.0f", target_dataset.data[i][k]);
            if (k < target_dataset.num_cols - 1) {
                printf(", ");
            }
        }
        printf("], Predicted: [");
        for (int k = 0; k < OUTPUT_NODES; k++) {
            printf("%f (Rounded: %.0f)", temp_output[k], round(temp_output[k]));
            if (k < OUTPUT_NODES - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }

    // Free training and target dataset memory
    free_dataset(&training_dataset);
    free_dataset(&target_dataset);

    // --- Inference Phase ---
    printf("\n--- Inference Phase ---\n");

    const char *inference_filepath = "data.csv";
    Dataset inference_dataset;

    if (load_csv_data(inference_filepath, INPUT_NODES, &inference_dataset) != 0) {
        fprintf(stderr, "Failed to load inference data from '%s'. Exiting inference phase.\n", inference_filepath);
        return 1;
    }

    printf("\nMaking predictions on user provided data from '%s':\n", inference_filepath);

    for (int i = 0; i < inference_dataset.num_rows; i++) {
        double hidden_output_inf[MAX_HIDDEN_LAYERS][MAX_NODES_PER_HIDDEN_LAYER];
        double output_inf[OUTPUT_NODES];

        printf("Input: [");
        for (int j = 0; j < inference_dataset.num_cols; j++) {
            printf("%.2f", inference_dataset.data[i][j]);
            if (j < inference_dataset.num_cols - 1) {
                printf(", ");
            }
        }
        printf("], ");

        forward_propagate(&nn, inference_dataset.data[i], hidden_output_inf, output_inf);

        printf("Predicted Output: [");
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