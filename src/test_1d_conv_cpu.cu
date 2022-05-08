#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cstdio>
#include <cstdlib>
#include <math.h>

#include "cuPrintf.cu"
#include "cuPrintf.cuh"

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans)                           \
	{                                                 \
		gpuAssert((ans), (char *)__FILE__, __LINE__); \
	}
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
				cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

#ifdef __APPLE__
/* Shim for Mac OS X (use at your own risk ;-) */
#include "apple_pthread_barrier.h"
#endif /* __APPLE__ */

#define CPNS 2.9 /* Cycles per nanosecond -- Adjust to your computer, for example a 3.2 GhZ GPU, this would be 3.2 */

// Things for running on GPU
#define PRINT_TIME 1			 // Whether we want to measure time cost (1/0)
// #define NUM_THREADS_PER_BLOCK 16 // Number of threads per block
#define TOL 0.05

#define ITERATIONS 2000
#define MINVAL 0.0
#define MAXVAL 10.0

// Things for 1D_Conv
// #define N_ARR_LEN 10000000  // array/vector size for output (P), must be multiple of NUM_THREADS_PER_BLOCK ==> Otherwise, you will have unmatched result...
#define MASK_WIDTH 3  // array size for mask (M)
#define GHOST_WIDTH 1 // padding cell, e.g., if GHOST_WIDTH = 1, then each side of row will have 1 padding

// dim3 dimGrid( ceil(SM_ARR_LEN/NUM_THREADS_PER_BLOCK), ceil(SM_ARR_LEN/NUM_THREADS_PER_BLOCK));   // Shape of grid
// dim3 dimBlock(NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK);    // Each block has shape of <16, 16>, so 256 threads/block

/* Prototypes */
void initializeArray1D(float *arr, int len, int seed);
void initializeArray1D_ignore_halo_cell(float *arr, int len, int HALO_CELL, int seed);
void print_1D_array(float *arr, int arrlen);
void print_2D_array(float *arr, int arrlen);
void conv_1D(float *N, float *M, float *P, int mask_width, int N_rowlen);

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:

		struct timespec {
		  time_t   tv_sec;   // seconds
		  long     tv_nsec;  // and nanoseconds
		};
 */

double interval(struct timespec start, struct timespec end)
{
	struct timespec temp;
	temp.tv_sec = end.tv_sec - start.tv_sec;
	temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	if (temp.tv_nsec < 0)
	{
		temp.tv_sec = temp.tv_sec - 1;
		temp.tv_nsec = temp.tv_nsec + 1000000000;
	}
	return (((double)temp.tv_sec) + ((double)temp.tv_nsec) * 1.0e-9);
}
/*
	 This method does not require adjusting a #define constant

  How to use this method:

	  struct timespec time_start, time_stop;
	  clock_gettime(CLOCK_REALTIME, &time_start);
	  // DO SOMETHING THAT TAKES TIME
	  clock_gettime(CLOCK_REALTIME, &time_stop);
	  measurement = interval(time_start, time_stop);
 */

/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
	double meas = 0; int i, j;
	struct timespec time_start, time_stop;
	double quasi_random = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
	j = 100;
	while (meas < 1.0) {
		for (i=1; i<j; i++) {
		/* This iterative calculation uses a chaotic map function, specifically
			the complex quadratic map (as in Julia and Mandelbrot sets), which is
			unpredictable enough to prevent compiler optimisation. */
		quasi_random = quasi_random*quasi_random - 1.923432;
		}
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
		meas = interval(time_start, time_stop);
		j *= 2; /* Twice as much delay next time, until we've taken 1 second */
	}
	return quasi_random;
}

// nvcc -g -G -lrt -lm  src/test_1d_conv.cu -o test_1d_conv
/*****************************************************************************/

int main(int argc, char *argv[])
{
	int i;
	// 1D_Conv variables
	int HALO_CELL = ceil(MASK_WIDTH / 2); /* 2 extra rows/columns for "ghost zone". */

	// Length for input array N
	int N_ARR_LEN = 1000000000;
	if (argc > 2) {
		N_ARR_LEN  = atoi(argv[2]);
	}
	int P_ARR_LEN = N_ARR_LEN;			  // They should be equal, 1 for start, 1 for end

	// GPU Timing variables
	struct timespec time_start, time_stop;
	printf("\n\n");
	printf("Size of HALO_CELL: %d\n", HALO_CELL);
	printf("Length of input array(N): %d\n", N_ARR_LEN);
	printf("Length of mask array(M): %d\n", MASK_WIDTH);
	printf("Length of output array(P): %d\n", P_ARR_LEN);

	/* ======================Memory Allocation and initialization(CPU & GPU) =============== */

	// Intialize arrays on host memory
	printf("\nInitializing the input arrays, h_input...\n");
	float *h_input = (float *)calloc(N_ARR_LEN, sizeof(float));
	for (i = 0; i < N_ARR_LEN; i++){
		// h_input 			= {0, 1, 2, 3, 4, 5, 0};
		if((i<HALO_CELL) || (i>= (N_ARR_LEN-HALO_CELL))){
			h_input[i] = 0;
		}else{
			h_input[i] = rand() % 100;
		}
		if (i > N_ARR_LEN - 5)
			printf("h_input[%d] = %.2f\n", i, h_input[i]);
	}

	// Intialize value for h_mask
	float *h_mask = (float *)calloc(MASK_WIDTH, sizeof(float));
	int random_init = 1; 
	if (random_init){
		for(int i = 0; i<MASK_WIDTH; i++){
			h_mask[i] = rand() % 10;
		}
	}else{
		h_mask[0] = 0.3;
		h_mask[1] = 0.2;
		h_mask[2] = 0.8;
	}
	// Check initialized values
	printf("Checking Intialized value for h_mask\n");
	for (i = 0; i < MASK_WIDTH; i++){
		printf("h_mask[%d] = %.2f\n", i, h_mask[i]);
	}
	printf("\t... done\n\n");

	// Allocate arrays on host memory (calloc will use zero-initialization)
	float *h_output_gold = (float *)calloc(P_ARR_LEN, sizeof(float)); // result computed in CPU
	float *h_output_data = (float *)calloc(P_ARR_LEN, sizeof(float)); // result computed in GPU

	/* ====================== Running code on CPU =============== */
	// Warmup
	printf("Warmup tests \n\n");
	double wakeup_answer = wakeup_delay();
	printf("Wakeup delay computed: %g \n", wakeup_answer);

	int NUM_TESTS = 4;
	int test_N_lenth[] = {1024, 100000, 10000000, 1000000000};
	double time_stamp[NUM_TESTS];
	printf("Running code in CPU \n");
	for(int i=0; i<NUM_TESTS; i++){
		clock_gettime(CLOCK_REALTIME, &time_start);
		conv_1D(h_input, h_mask, h_output_gold, MASK_WIDTH, test_N_lenth[i]);
		clock_gettime(CLOCK_REALTIME, &time_stop);
		time_stamp[i] = interval(time_start, time_stop);
	}
	printf("Finished running 1D conv in CPU \n");

	printf("All times are in cycles (if CPNS is set correctly in code)\n");
	printf("\n");

  	/* output times */
  	printf("N_lenth, Mask_length, output_length, 1D conv time(msec)\n");
    for (int i = 0; i < NUM_TESTS; i++) {
		printf("%7d, \t%12d, \t%13d, \t%13.8g", test_N_lenth[i], MASK_WIDTH, P_ARR_LEN, (double)CPNS * 1.0e3 * time_stamp[i]);
      	// printf("%d,  ", test_N_lenth[i] );
        // printf("%ld", (long int)((double)(CPNS) * 1.0e9 * time_stamp[j][i]));
      printf("\n");
    }

	printf("\n");
	/* ====================== Running code on GPU =============== */
	printf("==========> All CPU tests are done! Now, running GPU code!\n");

	free(h_input);
	free(h_mask);
	free(h_output_data);
	free(h_output_gold);

	return 0;
} /* end main */

/************************ Some helper function  ******/
void initializeArray1D(float *arr, int len, int seed)
{
	int i;
	float randNum;
	srand(seed);
	if (len > 0)
	{
		for (i = 0; i < len; i++)
		{
			// Randomly initialize each cell, but leave halo cell to zero
			randNum = (float)rand();
			arr[i] = randNum;
		}
	}
}

void initializeArray1D_ignore_halo_cell(float *arr, int len, int HALO_CELL, int seed)
{
	int i;
	float randNum;
	srand(seed);
	if (len > 0)
	{
		for (i = HALO_CELL; i < len - HALO_CELL; i++)
		{
			// Randomly initialize each cell, but leave halo cell to zero
			randNum = (float)rand();
			arr[i] = randNum;
		}
	}
}

/* print all elements of an array */
void print_1D_array(float *arr, int arrlen)
{
	int i;
	printf("row length = %d\n", arrlen);
	for (i = 0; i < arrlen; i++)
	{
		printf("%.4f ", arr[i]);
	}
	printf("\n");
}

/* print all elements of an matrix */
void print_2D_matrix(float *arr, int row_len)
{
	int i, j;
	printf("row length = %d\n", row_len);
	for (i = 0; i < row_len; i++)
	{
		for (j = 0; j < row_len; j++)
			printf("%.4f ", arr[i * row_len + j]);
		printf("\n");
	}
}

/************************************/

/* matrix-array multiplication */
void conv_1D(float *N, float *M, float *P, int mask_width, int N_rowlen)
{
	// matrix a of size 1 x n (array)
	// matrix b of size n x p
	// matrix result of size 1 x p (array)
	// result = a * b
	int i;
	float Pvalue;
	// Return directly, if threadIdx exceed the size of P
	int halo_width = (mask_width - 1) / 2; // assume mask_width is odd number
	/* Return immediately, if we are in the ghost cell, e.g., if N_lenth=5, mask_width=3, and ghost_width=1, then we only compute value for P[1-3], and it require to read value from input array N[0-4]
	For example
		N = [0,	1,	2,	3,	4]
		M =   [.3, .2, .8]
		P = [0,	x,	x,	x,	0]
	*/
	for (i = halo_width; i < N_rowlen-halo_width; i++){
		Pvalue = 0;
		for (int j = 0; j < mask_width; j++){
			Pvalue += N[i - halo_width + j] * M[j];
		}
		P[i] = Pvalue;
	}
}


