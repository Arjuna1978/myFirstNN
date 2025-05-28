#ifndef NN_FILE_HANDLER_H
#define NN_FILE_HANDLER_H

#include <stdio.h> 

// Structure to hold data
typedef struct {
    double **data;     // Pointer to the 2D array of data
    int num_rows;      // Number of rows (training examples)
    int num_cols;      // Number of columns (features/outputs)
} Dataset;

/**
 This function reads a CSV file, parses its contents, with memory safety managed.
 
 @param filepath The path to the CSV file.
 @param expected_cols The expected number of columns in the CSV.
 The function will error if the actual column count differs.
 @param dataset A pointer to a Dataset struct to be populated.
 @return 0 on success, -1 if the shit hits the fan.
 **/

int load_csv_data(const char *filepath, int expected_cols, Dataset *dataset);

/** 
  Frees the allocated memory for a Dataset.
 It is crucial to call this function when the dataset is no longer needed
 to prevent memory leaks. Really need to call this or we will have leaky code
 
 @param dataset A pointer to the Dataset struct whose memory should be freed.
 **/
void free_dataset(Dataset *dataset);

#endif // NN_FILE_HANDLER_H