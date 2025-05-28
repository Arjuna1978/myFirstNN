#include "nnFileHandler.h"
#include <stdlib.h> 
#include <string.h>


//Config
#define MAX_LINE_LENGTH 1024 // Maximum characters in a line of CSV
#define DELIMITER ","        // CSV delimiter

// Count columns in a line
static int count_columns(const char *line) {
    if (!line || strlen(line) == 0) return 0;
    int count = 1; // At least one column
    for (int i = 0; line[i] != '\0'; i++) {
        if (line[i] == DELIMITER[0]) {
            count++;
        }
    }
    return count;
}

int load_csv_data(const char *filepath, int expected_cols, Dataset *dataset) {
    FILE *file = fopen(filepath, "r");
    if (!file) {
        perror("Error opening file");
        return -1;
    }

    char line[MAX_LINE_LENGTH];
    int num_rows = 0;
    int actual_cols = -1; // To store the actual number of columns detected

    // First pass: Count rows and verify column count
    while (fgets(line, sizeof(line), file)) {
        // Remove newline character if present
        line[strcspn(line, "\n")] = 0;

        if (strlen(line) == 0) continue; // Skip empty lines

        int current_cols = count_columns(line);
        if (actual_cols == -1) {
            actual_cols = current_cols; // Set actual_cols from the first non-empty line
        } else if (actual_cols != current_cols) {
            fprintf(stderr, "Error: Inconsistent number of columns in CSV file '%s'. Expected %d, found %d.\n",
                    filepath, actual_cols, current_cols);
            fclose(file);
            return -1;
        }
        num_rows++;
    }

    if (num_rows == 0) {
        fprintf(stderr, "Error: CSV file '%s' is empty or contains no valid data.\n", filepath);
        fclose(file);
        return -1;
    }
    if (actual_cols == -1) { // Should not happen if num_rows > 0 and file is not empty
        fprintf(stderr, "Error: Could not determine column count in CSV file '%s'.\n", filepath);
        fclose(file);
        return -1;
    }

    if (actual_cols != expected_cols) {
        fprintf(stderr, "Error: Column count mismatch in CSV file '%s'. Expected %d, found %d.\n",
                filepath, expected_cols, actual_cols);
        fclose(file);
        return -1;
    }

    dataset->num_rows = num_rows;
    dataset->num_cols = actual_cols;

    // Allocate memory for rows (pointers to arrays)
    dataset->data = (double **)malloc(num_rows * sizeof(double *));
    if (!dataset->data) {
        perror("Memory allocation failed for dataset rows");
        fclose(file);
        return -1;
    }

    // Allocate memory for columns in each row
    for (int i = 0; i < num_rows; i++) {
        dataset->data[i] = (double *)malloc(actual_cols * sizeof(double));
        if (!dataset->data[i]) {
            perror("Memory allocation failed for dataset columns");
            // Clean up previously allocated rows
            for (int j = 0; j < i; j++) {
                free(dataset->data[j]);
            }
            free(dataset->data);
            fclose(file);
            return -1;
        }
    }

    // Second pass: Read and parse data
    fseek(file, 0, SEEK_SET); // Reset file pointer to the beginning
    int row_idx = 0;
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = 0; // Remove newline

        if (strlen(line) == 0) continue; // Skip empty lines

        char *token;
        char *rest = line; // Use a mutable copy for strtok_r if preferred

        for (int col_idx = 0; col_idx < actual_cols; col_idx++) {
            token = strtok_r(rest, DELIMITER, &rest); // strtok_r for thread safety
            if (!token) {
                fprintf(stderr, "Error: Missing data at row %d, column %d in '%s'.\n", row_idx, col_idx, filepath);
                free_dataset(dataset); // Clean up
                fclose(file);
                return -1;
            }
            dataset->data[row_idx][col_idx] = strtod(token, NULL);
        }
        row_idx++;
    }

    fclose(file);
    return 0; // Success
}

void free_dataset(Dataset *dataset) {
    if (dataset && dataset->data) {
        for (int i = 0; i < dataset->num_rows; i++) {
            free(dataset->data[i]);
        }
        free(dataset->data);
        dataset->data = NULL; // Clear pointer after freeing
        dataset->num_rows = 0;
        dataset->num_cols = 0;
    }
}