# Simple Feed-Forward Neural Network in C

This project implements a basic feed-forward neural network (FNN) in C. 
I have made it capable of learning simple patterns 1d patterns eg this variable results in that output
I have given a XOR logic gate as an example. 

The network uses a single hidden layer, sigmoid activation functions, and trains using the backpropagation algorithm. Training and target data are loaded from CSV files, providing flexibility for different datasets.

## NN architechture

* INPUT_NODES 2

* HIDDEN_NODES 4

* OUTPUT_NODES 1


## Project Structure

The project consists of the following files:

* `simpleNN.c`: The main neural network
* `nnMaths.h`: All my maths 
* `nnMaths.c`: All my maths
* `nnFileHandler.h`:My file handler
* `nnFileHandler.c`: My file handler
* `training_data.csv`: CSV file containing xOr training data (that will work)
* `target_data.csv`: CSV file containing target info

## Dependencies

I've used GCC and standard libraries so if you have GCC and standard C libraries installed this should compile

## Compliling

I made a bash file for people who are lazy like me and can't seem to get linkings right... so to compile just run:
bash compile.sh 

## Compliling
Make sure you have both csv files in the same directory... the code won't go searching

Once you compile just run:
simpleNN

# Have fun!!
