/*
Author: Manohar Mukku
Date: 21.07.2018
Desc: Feedforward propagation
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include "forward_propagation.h"
#include "time.h"
#include "pthread.h"
#define max(x, y) (x > y ? x : y)
#define Num_threads 4

int clock_gettime(clockid_t clk_id, struct timespec *tp);

double interval_forward(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

/*
struct forward_work
{
    double *Arr;
    double **Brr;
    double *Resrr;
    int Str_n;
    int Str_p;
    int thread_id;
};
*/
/*
void *work(void *i){
    
    struct forward_work *work_in;
    int n, p, T_ID;
    double *a, *result;
    double **b;
    work_in = (struct forward_work*) i;
    a = work_in->Arr;
    result = work_in->Resrr;
    b = work_in->Brr;
    n = work_in->Str_n;
    p = work_in->Str_p;
    T_ID = work_in->thread_id;
    int begin, end;
    begin = T_ID * p / Num_threads;
    end = (T_ID + 1) * p / Num_threads;
    if(T_ID == Num_threads -1){end = p;}
    int j, k;
    int aa=0;
    //printf("%d=%d\n", aa, T_ID);
    for (j = begin; j < end; j++) {
        result[j] = 0.0;
        //aa=aa+1;
        //printf("%d=%d\n", aa, T_ID);
        for (k = 0; k < n; k++){
            //aa=aa+1;
            //printf("j=%d=%d\n", j, T_ID);
            result[j] += (a[k] * b[k][j]);  
        }
            
    }
    //printf("%d=%d\n", aa++, T_ID);
    pthread_exit(NULL);
}*/


/*void mat_mul(double* a, double** b, double* result, int n, int p) {
    // matrix a of size 1 x n (array)
    // matrix b of size n x p
    // matrix result of size 1 x p (array)
    // result = a * b


    pthread_t id[Num_threads];
    struct forward_work  *str_work;
    str_work = (struct forward_work *)calloc(1, sizeof(struct forward_work));
    str_work->Arr = a;
    str_work->Brr = b;
    str_work->Resrr = result;
    str_work->Str_n = n;
    str_work->Str_p = p;

    for(long t = 0; t < Num_threads; t++){
        str_work->thread_id = t;
        if(pthread_create(&id[t], NULL, work, (void *)(str_work))){
            printf("ERROR creating the thread\n");
            exit(19);
        }
    }
}*/

void mat_mul(double* a, double** b, double* result, int n, int p) {
    int j, k;
    for (j = 0; j < p; j++) {
        result[j] = 0.0;
        for (k = 0; k < n; k++)
            result[j] += (a[k] * b[k][j]);
    }
}

void identity(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = input[i]; // Identity function
}

void sigmoid(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = 1.0 / (1.0 + exp(-input[i])); // Sigmoid function
}

void tan_h(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = tanh(input[i]); // tanh function
}

void relu(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = max(0.0, input[i]); // ReLU function
}

void softmax(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++)
        sum += exp(input[i]);

    for (i = 0; i < n; i++) 
        output[i+1] = exp(input[i]) / sum; // Softmax function
}

void forward_propagation(parameters* param, int training_example, int n_layers, int* layer_sizes, double** layer_inputs, double** layer_outputs, double*** weight_forward) {
    // Fill the input layer's input and output (both are equal) from data matrix with the given training example
    int i;
    struct timespec time_start, time_stop;
    layer_outputs[0][0] = 1; // Bias term of input layer
    for (i = 0; i < param->feature_size-1; i++)
        layer_outputs[0][i+1] = layer_inputs[0][i] = param->data_train[training_example][i];

    // Perform forward propagation for each hidden layer
    // Calculate input and output of each hidden layer
    for (i = 1; i < n_layers-1; i++) {
        // Compute layer_inputs[i]

        mat_mul(layer_outputs[i-1], weight_forward[i-1], layer_inputs[i], layer_sizes[i-1]+1, layer_sizes[i]);


        // Compute layer_outputs[i]
        // Activation functions (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
        switch (param->hidden_activation_functions[i-1]) {
            case 1: // identity
                identity(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                break;
            case 2: // sigmoid
                sigmoid(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                break;
            case 3: // tanh
                tan_h(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                break;
            case 4: // relu
                relu(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                break;
            case 5: // softmax
                softmax(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                break;
            default:
                printf("Forward propagation: Invalid hidden activation function\n");
                exit(0);
                break;
        }
    }

    // Fill the output layers's input and output
    

    mat_mul(layer_outputs[n_layers-2], weight_forward[n_layers-2], layer_inputs[n_layers-1], layer_sizes[n_layers-2]+1, layer_sizes[n_layers-1]);



    // Activation functions (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    switch (param->output_activation_function) {
        case 1: // identity
            identity(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
            break;
        case 2: // sigmoid
            sigmoid(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
            break;
        case 3: // tanh
            tan_h(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
            break;
        case 4: // relu
            relu(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
            break;
        case 5: // softmax
            softmax(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
            break;
        default:
            printf("Forward propagation: Invalid hidden activation function\n");
            exit(0);
            break;
    }
}
