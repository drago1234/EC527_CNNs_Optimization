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
__global__ void cuda_conv_1D_single_block(float *N, float *M, float *P, int mask_width, int N_rowlen);
__global__ void cuda_conv_1D_multi_block(float *N, float *M, float *P, int mask_width, int N_rowlen);
__global__ void cuda_conv_1D_multi_block_with_mask_in_constant_memory(float *N, float *P, int mask_width, int N_rowlen);
__global__ void cuda_conv_1D_tiled_and_shared_memory_kernel(float *N, float *P, int mask_width, int N_rowlen);
__global__ void cuda_conv_1D_tiled_and_shared_memory_kernel2(float *N, float *P, int mask_width, int N_rowlen);

/* Things to put into device constant memory */
__constant__ float d_mask_constant[MASK_WIDTH];

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
	int start_point = 0;
	// 1D_Conv variables
	int HALO_CELL = ceil(MASK_WIDTH / 2); /* 2 extra rows/columns for "ghost zone". */

	int taskid = 6; // 1(1D single block) --> 2(1D multi-block) --> 3(1D tiled algo with shared memory)
	if (argc > 1) {
		taskid  = atoi(argv[1]);
	}

	// Length for input array N 
	int N_ARR_LEN = 1024;	//1000000000
	if (argc > 2) {
		N_ARR_LEN  = atoi(argv[2]);
	}
	int P_ARR_LEN = N_ARR_LEN;			  // They should be equal, 1 for start, 1 for end

	// Number of threads for each block
	int NUM_THREADS_PER_BLOCK = 16;
	if (argc > 3) {
		N_ARR_LEN  = atoi(argv[3]);
	}

	// GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;
	struct timespec time_start, time_stop;
	double elapsed_cpu;
	printf("\n\n");
	printf("Size of HALO_CELL: %d\n", HALO_CELL);
	printf("Length of input array(N): %d\n", N_ARR_LEN);
	printf("Length of mask array(M): %d\n", MASK_WIDTH);
	printf("Length of output array(P): %d\n", P_ARR_LEN);

	/* ======================Memory Allocation and initialization(CPU & GPU) =============== */
	size_t alloc_size = (N_ARR_LEN) * sizeof(float);

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

	// Allocate memory on GPU device
	float *d_input, *d_mask, *d_output_data; // Arrays on GPU global memory
	cudaMalloc((void **)&d_input, N_ARR_LEN * sizeof(float));
	cudaMalloc((void **)&d_mask, MASK_WIDTH * sizeof(float));
	cudaMalloc((void **)&d_output_data, P_ARR_LEN * sizeof(float));

	/* ====================== Running code on CPU =============== */
	// Warmup
	printf("Warmup tests \n\n");
	double wakeup_answer = wakeup_delay();
	printf("Wakeup delay computed: %g \n", wakeup_answer);

	printf("Running code in CPU \n");
	clock_gettime(CLOCK_REALTIME, &time_start);
	conv_1D(h_input, h_mask, h_output_gold, MASK_WIDTH, N_ARR_LEN);
	clock_gettime(CLOCK_REALTIME, &time_stop);
	elapsed_cpu = interval(time_start, time_stop);
	printf("Finished running 1D conv in CPU \n");

	printf("All times are in cycles (if CPNS is set correctly in code)\n");
	printf("\n");
	printf("N_lenth, Mask_length, output_length, 1D conv time(msec)\n");
	printf("%7d, \t%12d, \t%13d, \t%13.8g", N_ARR_LEN, MASK_WIDTH, P_ARR_LEN, (double)CPNS * 1.0e3 * elapsed_cpu);
	printf("\n");


	/* ====================== Running code on GPU =============== */
	printf("==========> All CPU tests are done! Now, running GPU code!\n");
	// Select GPU
	CUDA_SAFE_CALL(cudaSetDevice(0));
#if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);
#endif

	/* ====================== Transfer data to GPU/Device =============== */
	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_input, h_input, N_ARR_LEN * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_mask, h_mask, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
	// CUDA_SAFE_CALL(cudaMemcpy(d_output_data, h_output_data, P_ARR_LEN * sizeof(float), cudaMemcpyHostToDevice));

	// Transfer M to device constant memory
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_mask_constant, h_mask, MASK_WIDTH * sizeof(float))); // cudaMemcpyToSymbol(dest, src, size)

	// Launch the kernel
	cudaPrintfInit();

	// For 1D CNN
	int THREADS = NUM_THREADS_PER_BLOCK;
	int GRID = (P_ARR_LEN + THREADS - 1) / THREADS;
	// Amount of space per-block for shared memory
	// This is padded by the overhanging radius on either side
	size_t SHMEM = (THREADS + HALO_CELL*2) * sizeof(int);

	// For 2D CNN
	dim3 dimGrid(ceil(P_ARR_LEN / NUM_THREADS_PER_BLOCK), 1); // Shape of grid = # of elements in a row divided by the number of threads per block row
	dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1);

	printf("==============>Running taskid #: %d on GPU!\n", taskid);
	printf("1(1D single block) --> 2(1D multi-block) --> 3(1D mulit-block with mask in constant memory) --> 4(tiled algo with Strategy 1) --> 5(1D tiled strategy 2)\n");
	switch (taskid){
	case 1:
		// single block, each thread compute single output
		cuda_conv_1D_single_block<<<1, P_ARR_LEN>>>(d_input, d_mask, d_output_data, MASK_WIDTH, N_ARR_LEN);
		break;
	case 2:
		cuda_conv_1D_multi_block<<<dimGrid, dimBlock>>>(d_input, d_mask, d_output_data, MASK_WIDTH, N_ARR_LEN);
		break;
	case 3:
		cuda_conv_1D_multi_block_with_mask_in_constant_memory<<<dimGrid, dimBlock>>>(d_input, d_output_data, MASK_WIDTH, N_ARR_LEN);
		break;
	case 4:
		cuda_conv_1D_tiled_and_shared_memory_kernel<<<GRID, THREADS, SHMEM>>>(d_input, d_output_data, MASK_WIDTH, N_ARR_LEN);
		break;
	case 5:
		cuda_conv_1D_tiled_and_shared_memory_kernel2<<<GRID, THREADS>>>(d_input, d_output_data, MASK_WIDTH, N_ARR_LEN);
		break;
	default:
		printf("ERROR: You hit an error, no such taskid # %d!n", taskid);
		break;
	}
	cudaPrintfDisplay(stdout, true); cudaPrintfEnd();

	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());

	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_input, d_input, N_ARR_LEN * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_mask, d_mask, MASK_WIDTH * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_output_data, d_output_data, P_ARR_LEN * sizeof(float), cudaMemcpyDeviceToHost));

#if PRINT_TIME
	// Stop and destroy the timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif

	// Comparing the result we obtained from kernel function and regular SOR:
	int errCount = 0;
	int zeroCount = 0;
	float *relative_error = (float *)malloc(P_ARR_LEN * sizeof(float));
	for (i = 0; i < P_ARR_LEN; i++){
		relative_error[i] = abs(h_output_gold[i] - h_output_data[i]) / h_output_gold[i] * 100;
		if (relative_error[i] > TOL){
			errCount++;
			if (errCount < 15) {printf("UNMATCHED RESULT in %d:\t%.4f\t%.4f\t%.2f %%\n", i, h_output_gold[i], h_output_data[i], relative_error[i]); }
		}
		if (h_output_data[i] == 0.0){
			zeroCount++;
			if (zeroCount < 15) {printf("ZERO RESULT in %d:\t%.4f\t%.4f\t%.2f %%\n", i, h_output_gold[i], h_output_data[i], relative_error[i]); }
		}
	}

	// double error_rate = errCount/(arrLen*arrLen) * 100;
	if (errCount > 0){
		printf("\n@ERROR: TEST FAILED: %d/%d results did not match\n", errCount, P_ARR_LEN);
	}else if (zeroCount > 0){
		printf("\n@ERROR: TEST FAILED: %d/%d results (from GPU) are zero\n", zeroCount, P_ARR_LEN);
	}else{
		printf("\nTEST PASSED: All results matched\n");
	}

	printf("\n");
	printf("i:\th_output_gold[i]\th_output_data[i]\n");
	start_point = 0;
	int end_point = P_ARR_LEN < 50 ? P_ARR_LEN : 50;
	for (i = start_point; i < end_point; i++)
	{
		printf("%d:\t%.4f\t%.4f\n", i, h_output_gold[i], h_output_data[i]);
	}

	// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(d_input));
	CUDA_SAFE_CALL(cudaFree(d_mask));
	CUDA_SAFE_CALL(cudaFree(d_output_data));

	// free(h_input);
	// free(h_mask);
	// free(h_output_data);
	// free(h_output_gold);

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


/************ CUDA implementation
 * The idea for this implementation, is to chopping off the entire matrix by block/tile, so that each thread is responsible for a single block.
 */
__global__ void cuda_conv_1D_single_block(float *N, float *M, float *P, int mask_width, int N_rowlen){
	/*
	Input parameter:
		float *N: pointer to input array N
		float *M: pointer to input mask M
		float *P: pointer to output array P
		int Mask_Width: size of mask, e.g., (1, len(M))
		int Width: size of input and output array Width, e.g., (1, n=len(N))
	*/
	int i = threadIdx.x; // i is for output array P[halo_width, P_ARR_LEN-halo_width]
	float Pvalue = 0;
	int j;

	// Return directly, if threadIdx exceed the size of P
	int halo_width = (mask_width - 1) / 2; // assume mask_width is odd number
	/* Return immediately, if we are in the ghost cell, e.g., if N_lenth=5, mask_width=3, and ghost_width=1, then we only compute value for P[1-3], and it require to read value from input array N[0-4]
	For example
		N = [0,	1,	2,	3,	4]
		M =   [.3, .2, .8]
		P = [0,	x,	x,	x,	0]
	*/
	if (i < halo_width || i > (N_rowlen - 1 - halo_width)) return;	// 1 off for idx
	for (j = 0; j < mask_width; j++){
		Pvalue += N[i - halo_width + j] * M[j];
	}
	P[i] = Pvalue;
}

__global__ void cuda_conv_1D_multi_block(float *N, float *M, float *P, int mask_width, int N_rowlen)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; // i is [0, P_ARR_LEN-1]
	float Pvalue = 0;

	// Return directly, if threadIdx exceed the size of P
	int halo_width = (mask_width - 1) / 2; // assume mask_width is odd number
	/* Return immediately, if we are in the ghost cell, e.g., if N_lenth=5, mask_width=3, and ghost_width=1, then we only compute value for P[1-3], and it require to read value from input array N[0-4]
	For example
		N = [0,	1,	2,	3,	4]
		M =   [.3, .2, .8]
		P = [0,	x,	x,	x,	0]
	*/
	if (i < halo_width || i > (N_rowlen - 1 - halo_width)) return;	// 1 off for idx
	for (int j = 0; j < mask_width; j++){
		Pvalue += N[i - halo_width + j] * M[j];
	}
	P[i] = Pvalue;
}

__global__ void cuda_conv_1D_multi_block_with_mask_in_constant_memory(float *N, float *P, int mask_width, int N_rowlen)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; // i is [0, P_ARR_LEN-1]
	float Pvalue = 0;

	// Return directly, if threadIdx exceed the size of P
	int halo_width = (mask_width - 1) / 2; // assume mask_width is odd number
	/* Return immediately, if we are in the ghost cell, e.g., if N_lenth=5, mask_width=3, and ghost_width=1, then we only compute value for P[1-3], and it require to read value from input array N[0-4]
	For example
		N = [0,	1,	2,	3,	4]
		M =   [.3, .2, .8]
		P = [0,	x,	x,	x,	0]
	*/
	if (i < halo_width || i > (N_rowlen - 1 - halo_width)) return;	// 1 off for idx
	for (int j = 0; j < mask_width; j++){
		Pvalue += N[i - halo_width + j] * d_mask_constant[j];
	}
	P[i] = Pvalue;
}


// Strategy 1:
__global__ void cuda_conv_1D_tiled_and_shared_memory_kernel(float *N, float *P, int mask_width, int N_rowlen){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int s_array[];  

	int halo_width = (mask_width - 1) / 2; 	// assume mask_width is odd number
	// Skip the first and last cell for output, e.g, P[0, ..., P_ARR_LEN-1] -->Because those are ghost cell, no input for them to compute
  	// if (tid < halo_width || tid > (N_rowlen - 1 - halo_width)) return;

	// Load the lower elements first starting at the halo
	s_array[threadIdx.x] = N[tid]; 

	// Maximum Size of the shared memory array
	int n_padded = blockDim.x + 2 * halo_width;
	// Offset for the second set of loads in shared memory
	int s_offset = threadIdx.x + blockDim.x;
	// Global offset for the array in DRAM
	int g_offset = tid + blockDim.x;
	// Load in the remaining upper elements
	if (s_offset < n_padded) {
	  s_array[s_offset] = N[g_offset];
	}
	
	__syncthreads();
	// Temp value for calculation
	float Pvalue = 0;
	// Go over each element of the mask
	for (int j = 0; j < mask_width; j++){
		Pvalue += s_array[threadIdx.x + j] * d_mask_constant[j];
	}
	if (tid >= halo_width || tid < (N_rowlen -  halo_width)){
		P[tid+halo_width] = Pvalue;
	}
}

// Strategy 2:
#define TILE_WIDTH 4
__global__ void cuda_conv_1D_tiled_and_shared_memory_kernel2(float *N, float *P, int mask_width, int N_rowlen){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// Instantiate shared array s_array
	__shared__ float s_array[TILE_WIDTH];

	// Load data with corresponding idx from N
	s_array[threadIdx.x] = N[i];

	// Making sure all threads have finished loading data into shared memory.
	__syncthreads();

	int halo_width = (mask_width - 1) / 2; 	// assume mask_width is odd number
	/* Return immediately, if we are in the ghost cell, e.g., if N_lenth=5, mask_width=3, and ghost_width=1, then we only compute value for P[1-3], and it require to read value from input array N[0-4]
	For example
		N = [0,	1,	2,	3,	4]
		M =   [.3, .2, .8]
		P = [0,	x,	x,	x,	0]
	*/
	if (i < halo_width || i > (N_rowlen - 1 - halo_width)) return;	// 1 off for idx

	int this_tile_start_point = blockIdx.x * blockDim.x;
	int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
	int N_start_point = i - halo_width;			// the first idx we should start to read from N
	
	float Pvalue = 0;
	/* Go through each item in mask. If the corresponding item needed from input array is in ghost cell, then we will read it from global memory, otherwise, we will read it from shared memory! */
	for (int j = 0; j < mask_width; j++){
		int N_index = N_start_point + j;	//	the corresponding idx of M[j] for N[N_index]
		// Check the boundary
		if (N_index >= 0 && N_index < N_rowlen){
			// Decide whether should read from global memory or shared memory?
			int reading_from_s_array = ((N_index >= this_tile_start_point) && (N_index < next_tile_start_point));
			if (reading_from_s_array){
				Pvalue += s_array[threadIdx.x - halo_width + j] * d_mask_constant[j];
			}else {
				Pvalue += N[N_index] * d_mask_constant[j];
			}
		}
	}
	P[i] = Pvalue;
}


