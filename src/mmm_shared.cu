#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#include <cstdio>
#include <cstdlib>
#include <math.h>

#include "cuPrintf.cu"
#include "cuPrintf.cuh"

#define CPNS 2.9    /* Cycles per nanosecond -- Adjust to your computer, for example a 3.2 GhZ GPU, this would be 3.2 */

// Things for running on GPU
#define SM_ARR_LEN         	2048     // array/vector size
#define TILE_WIDTH			16
#define BLOCK_SIZE   		TILE_WIDTH 		// Number of threads per block
#define NUM_BLOCKS         	ceil(SM_ARR_LEN/(float)BLOCK_SIZE)       // This is equivalent to (N + M -1), where N is the number of elements, and M is the number of thread within a block. This will ensure that we have enough threads to cover all the elements.
#define PRINT_TIME         1        // Whether we want to measure time cost (1/0)

#define IDENT 0
#define TOL 0.01

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
										cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void initializeArray1D(float *arr, int len, int seed);
void mmm_ijk(float *a, float *b, float *c, int rowlen);		// best order for MMM
__global__ void cuda_mmm_ijk(float *d_a, float *d_b, float *d_data, int rowlen);
__global__ void cuda_mmm_with_shared_memory(float *a0, float *b0, float *c0, int rowlen);

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
	if (temp.tv_nsec < 0) {
		temp.tv_sec = temp.tv_sec - 1;
		temp.tv_nsec = temp.tv_nsec + 1000000000;
	}
	return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
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


/*****************************************************************************/
int main(int argc, char *argv[])
{
	int i;
	int start_point = 4000;
	// GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;
	// CPU Timing variables
	struct timespec time_start, time_stop;
	double elapsed_cpu;

	int rowlen = 0;   // N x N matrix
	if (argc > 1) {
		rowlen  = atoi(argv[1]);
	}
	else {
		rowlen = SM_ARR_LEN;
	}
	printf("Length of the row = %d\n", rowlen);

	size_t alloc_size = (rowlen*rowlen) * sizeof(float);
	printf("%d element bing allocated, and %ld size in byte\n", rowlen*rowlen, alloc_size);
		
	// Allocate arrays on host CPU memory
	float *h_a              		= (float *) malloc(alloc_size);
	float *h_b              		= (float *) malloc(alloc_size);
	float *h_data              	= (float *) malloc(alloc_size);
	float *h_data_gold          	= (float *) malloc(alloc_size);
	/* h_data is the data will be transferred from GPU calculation, and h_data_gold is the ground truth resulted computed from CPU. */

	// Allocate on device GPU memory
	float *d_a, *d_b, *d_data;		// Arrays on GPU global memoryc
	cudaMalloc((void **)&d_a, alloc_size);
	cudaMalloc((void **)&d_b, alloc_size);
	cudaMalloc((void **)&d_data, alloc_size);
	
	// Intialize arrays on host CPU memory
	printf("\nInitializing the arrays on CPU...");
	// Arrays are initialized with a known seed for reproducability
	initializeArray1D(h_a, rowlen*rowlen, 1111);
	initializeArray1D(h_b, rowlen*rowlen, 2222);
	initializeArray1D(h_data, rowlen*rowlen, 3333);
	for (i = 0; i < rowlen * rowlen; i++){
		h_data_gold[i] = h_data[i];
	}
	printf("\t... done\n\n");

	// Intialize arrays on GPU memory

	
	// Launch the MMM function on CPU
	printf("Dense MMM tests \n\n");
	double wakeup_answer = wakeup_delay();
	printf("Wakeup delay computed: %g \n", wakeup_answer);

	printf("Running MMM in CPU \n");
	clock_gettime(CLOCK_REALTIME, &time_start);
	mmm_ijk(h_a, h_b, h_data_gold, rowlen);
	clock_gettime(CLOCK_REALTIME, &time_stop);
	elapsed_cpu = interval(time_start, time_stop);
	printf("Finished running MMM in CPU \n");
	// Report the measurement: time elapsed in ns
	printf("All times are in cycles (if CPNS is set correctly in code)\n");
	printf("\n");
	printf("array_len, CPU time\n");
	printf("%4d, %10.4g", rowlen, (double)CPNS * 1.0e9 * elapsed_cpu);
	printf("\n");


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
	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, alloc_size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, alloc_size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, alloc_size, cudaMemcpyHostToDevice));

	// Launch the kernel
	cudaPrintfInit();
	dim3 dimGrid(NUM_BLOCKS, NUM_BLOCKS);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	printf("==============>Running the result for on GPU\n");
	// cuda_mmm_ijk<<<dimGrid, dimBlock>>>(d_a, d_b, d_data, rowlen);
	cuda_mmm_with_shared_memory<<<dimGrid, dimBlock>>>(d_a, d_b, d_data, rowlen);
	cudaPrintfDisplay(stdout, true); cudaPrintfEnd();

	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());

	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_data, d_data, alloc_size, cudaMemcpyDeviceToHost));

	#if PRINT_TIME
		// Stop and destroy the timer
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_gpu, start, stop);
		printf("\nGPU time: %f (msec)\n", elapsed_gpu);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	#endif

	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, alloc_size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, alloc_size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, alloc_size, cudaMemcpyHostToDevice));

	#if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);
	#endif

	// Launch the kernel
	cuda_mmm_with_shared_memory<<<dimGrid, dimBlock>>>(d_a, d_b, d_data, rowlen);

	#if PRINT_TIME
		// Stop and destroy the timer
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_gpu, start, stop);
		printf("\nGPU time(without data transfer, execution only): %f (msec)\n", elapsed_gpu);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	#endif
	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_data, d_data, alloc_size, cudaMemcpyDeviceToHost));

	// Comparing the result we obtained from kernel function and regular SOR:
	int errCount = 0; 
	int zeroCount = 0;
	float *relative_error = (float *)malloc(alloc_size);
	for (i=0; i<rowlen*rowlen; i++){
		relative_error[i] = abs(h_data_gold[i] - h_data[i]) / h_data_gold[i];
		if (relative_error[i] > TOL){
			// printf("FAILED in %d:\t%.8f\t%.8f\t%.8f %%\n", i, h_data_gold[i], h_data[i], relative_error[i]);
			errCount++;
		}
		if (h_data[i] == 0.0) {
			zeroCount++;
		}
	}

	// double error_rate = errCount/(rowlen*rowlen) * 100;
	if (errCount > 0) {
		printf("\n@ERROR: TEST FAILED: %d/%d results did not match\n", errCount, rowlen*rowlen);
	}else if (zeroCount > 0){
		printf("\n@ERROR: TEST FAILED: %d/%d results (from GPU) are zero\n", zeroCount, rowlen*rowlen);
	}else {
		printf("\nTEST PASSED: All results matched\n");
	}

	start_point = 1000;
	// for(i = start_point; i < start_point+50; i++) {
	//   //idx, gt_value, computed_value, relative error
	//   printf("%d:\t%.8f\t%.8f\t%.8f %%\n", i, h_data_gold[i], h_data[i], relative_error[i]);
	// } 
	for(i = start_point; i < start_point+128; i++) {
		printf("%d:\t%.8f\t%.8f\n", i, h_data_gold[i], h_data[i]);
	}
	printf("\n\n");

	// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(d_a));
	CUDA_SAFE_CALL(cudaFree(d_b));
	CUDA_SAFE_CALL(cudaFree(d_data));
	free(h_a);
	free(h_b);
	free(h_data);
	free(h_data_gold);

	return 0;
} /* end main */


/*********************************/
void initializeArray1D(float *arr, int len, int seed) {
	int i;
	float randNum;
	srand(seed);

	for (i = 0; i < len; i++) {
		randNum = (float) rand();
		// randNum = 1;
		arr[i] = randNum;
	}
}


/* mmm */
void mmm_ijk(float *a0, float *b0, float *c0, int rowlen){
	long int i, j, k;
	float sum = 0;
	for (i = 0; i < rowlen; i++) {
		for (j = 0; j < rowlen; j++) {
		sum = IDENT;
		for (k = 0; k < rowlen; k++) {
			sum += a0[i*rowlen+k] * b0[k*rowlen+j];
		}
		c0[i*rowlen+j] += sum;
		}
	}
}

/**************
 * This version will allow each thread to operate on multiple elements --> It's gonna be slower if we are running each thread on 1 element
 */

__global__ void cuda_mmm_with_shared_memory(float *Md, float *Nd, float *Pd, int Width){
	// extern __shared__ int s_data[]; // allocated shared memory during kernel launch

	// Declared the shared variable for each block 
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	// Use automatic variables resided in register for faster access (Only accessible by thread)
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	//Determine the row and column indexes of the d_P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;	// The accumulate the intermediate value
	//Loop over the d_M and d_N tiles required to compute d_P element 
	// The ph indicates the number of phases that have already been done for the dot product. This number is determined by the  Width/TILE_WIDTH, where Width is the number of elements in a matrix row, and TILE_WIDTH is the number of element in a block row. --> So, in another word, the size of ph is the number of block in a grid.
	for(int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ph++){
		// Collaborative loading of d__M ad d_N tiles into shared memory 
		// 因为我们要考虑the boundary case, the case where we don't have a squared matrix, so we need to check if the row and column index for Md is valid.
		if ((Row < Width) && (ph * TILE_WIDTH+tx) <Width){
			Mds[ty][tx] = Md[Row * Width + ph*TILE_WIDTH + tx];
		}
		if ((ph*TILE_WIDTH+ty)<Width && Col < Width){
			Nds[ty][tx] = Nd[(ph*TILE_WIDTH + ty)*Width + Col];
		}

		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; k++){
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}
	if ((Row < Width) && (Col<Width)){
		Pd[Row*Width + Col] = Pvalue;
	}
}


__global__ void cuda_mmm_ijk(float *a0, float *b0, float *c0, int rowlen){
	long int i, j, k;

	int resultId;
	float sum = 0;
	// int row_a, col_a, col_b = rowlen;

	// Block index 
	const int bId_x = blockIdx.x;
	const int bId_y = blockIdx.y;
	// Local thread index
	const int local_tid_x = threadIdx.x;
	const int local_tid_y = threadIdx.y;

	// Number of element need to be evaluated for a rows and columns for each block
	const int rows_per_block = rowlen / gridDim.x;
	const int cols_per_block = rowlen / gridDim.y;

	// Number of element need to be evaluated for a rows and columns for each thread
	const int rows_per_thread = rows_per_block / blockDim.x;
	const int cols_per_thread = cols_per_block / blockDim.y;

	// Row and columns start/ending indices for current block
	const int blockStartId_row = bId_x * rows_per_block;
	const int blockEndId_row = (bId_x+1) * rows_per_block - 1;
	const int blockStartId_col = bId_y * cols_per_block;
	const int blockEndId_col = (bId_y+1) * cols_per_block - 1;

	// Row and columns start/ending indices for current threads
	const int threadStartId_row = blockStartId_row + local_tid_x * rows_per_thread;
	const int threadEndId_row = blockEndId_row + (local_tid_x+1) * rows_per_thread -1;
	const int threadStartId_col = blockStartId_col + local_tid_y * cols_per_thread;
	const int threadEndId_col = blockEndId_col + (local_tid_y+1) * cols_per_thread -1;


	// Who I am??
	for (i = threadStartId_row; i <= threadEndId_row; i++) {
		for (j = threadStartId_col; j <= threadEndId_col; j++) {
			sum = IDENT;
			resultId = i*rowlen + j;
			for (k = 0; k < rowlen; k++) {
				sum += a0[i*rowlen+k] * b0[k*rowlen+j];
			}
			c0[resultId] = sum;
		}
	}
}
