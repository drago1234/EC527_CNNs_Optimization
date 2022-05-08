#ifndef PARAMETERS_H
#define PARAMETERS_H

typedef struct {
    int n_hidden;                       // Ex: 3
    int* hidden_layers_size;            // Ex: 4,5,5
    int* hidden_activation_functions;   // (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    double learning_rate;
    int n_iterations_max;               // Ex: 10000
    int momentum;
    int output_layer_size;
    int output_activation_function;     // (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    double** data_train;
    double** data_test;
    int feature_size;
    int train_sample_size;
    int test_sample_size;
    double*** weight;           // a pointer, point to a 2D array
} parameters;

#endif