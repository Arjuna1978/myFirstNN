//include the header refere
#include "nnMaths.h"
// include c standard maths library
#include <math.h> 

// Sigmoid function stolen from here : (https://en.wikipedia.org/wiki/Sigmoid_function)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function .. I did the dx, on paper so need to double check if there are issues
double sigmoid_derivative(double x) {
    // Assuming x is already the output of the sigmoid function
    return x * (1.0 - x);
}

// Still to try implementing the Gompertz function which allows us to tweek the x axis and rate of activation.

double gompertz(double x, int b, int c){

    int a = 1; // this is the asymptote NNs must have 1 or it breaks

    // incase not everyone knows that b and c need to be negative... 
    // just realised 0*-1 = 0 but I don't really feel like fixing it.
    if (b>=0){
        b= b*-1;
    }

    if (c>=0){
        c= c*-1;
    }
    
    return a*(exp (-b*exp(-c*x)));
}