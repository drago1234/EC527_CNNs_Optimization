#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cstdio>
#include <cstdlib>
#include <math.h>

#include "cuPrintf.cu"
#include "cuPrintf.cuh"

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

#ifdef __APPLE__
/* Shim for Mac OS X (use at your own risk ;-) */
# include "apple_pthread_barrier.h"
#endif /* __APPLE__ */

#define CPNS 2.9    /* Cycles per nanosecond -- Adjust to your computer, for example a 3.2 GhZ GPU, this would be 3.2 */

// Things for SOR
#define GHOST 2   /* 2 extra rows/columns for "ghost zone". */
#define MINVAL   0.0
#define MAXVAL  10.0
#define TOL 0.05
#define OMEGA 1       // TO BE DETERMINED

// Things for running on GPU
#define NUM_THREADS_PER_BLOCK   256 // Number of threads per block
#define NUM_BLOCKS         16       // Number of block in a grid
#define PRINT_TIME         1        // Whether we want to measure time cost (1/0)
#define SM_ARR_LEN         128     // array/vector size
#define ITERATIONS         2000
// dim3 dimGrid( ceil(SM_ARR_LEN/NUM_THREADS_PER_BLOCK), ceil(SM_ARR_LEN/NUM_THREADS_PER_BLOCK));   // Shape of grid
// dim3 dimBlock(NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK);    // Each block has shape of <16, 16>, so 256 threads/block 

/* Prototypes */
void SOR(int rowlen, float *data);
__global__ void cuda_single_block_SOR(int rowlen, float *data, double change);
__global__ void cuda_multi_block_SOR(int rowlen, float *data, double change);
void initializeArray1D(float *arr, int len, int seed);
void initializeArray2D(float *arr, int len, int seed);


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
      clock_gettime(CLOCK_REALTIME, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_REALTIME, &time_stop);
      measurement = interval(time_start, time_stop);
 */


/* -=-=-=-=- End of time measurement declarations =-=-=-=- */



/*****************************************************************************/
int main(int argc, char *argv[])
{
  int i;
  int start_point = 4000;
  double change = 1.0e10;   /* start w/ something big */
  // GPU Timing variables
  cudaEvent_t start, stop;
  float elapsed_gpu;
  struct timespec time_start, time_stop;
  double elapsed_cpu;

  int arrLen = 0;   // N x N matrix
  if (argc > 1) {
      arrLen  = atoi(argv[1]);
  }
  else {
      arrLen = SM_ARR_LEN;
  }
  printf("Length of the row = %d\n", arrLen);

  size_t alloc_size = (arrLen*arrLen) * sizeof(float);
  printf("%d element bing allocated, and %ld size in byte\n", arrLen*arrLen, alloc_size);
	
  // Allocate arrays on host memory
  float *h_data_gold          = (float *) malloc(alloc_size);
  float *h_data               = (float *) malloc(alloc_size);

  // Allocate GPU memory
  float *d_data;    // Arrays on GPU global memoryc
  cudaMalloc((void **)&d_data, alloc_size);
  
  // Intialize arrays on host memory
  printf("\nInitializing the arrays ...");
  // Arrays are initialized with a known seed for reproducability
  initializeArray1D(h_data, arrLen*arrLen, 2453);
  for (i = 0; i < arrLen * arrLen; i++){
    h_data_gold[i] = h_data[i];
  }
  printf("\t... done\n\n");

  // Launch the SOR function
  printf("Running SOR in CPU \n");
  clock_gettime(CLOCK_REALTIME, &time_start);
  SOR(arrLen, h_data_gold);
  clock_gettime(CLOCK_REALTIME, &time_stop);
  elapsed_cpu = interval(time_start, time_stop);
  printf("Finished running SOR in CPU \n");

  printf("All times are in cycles (if CPNS is set correctly in code)\n");
  printf("\n");
  printf("array_len, SOR time, SOR iters\n");
  printf("%4d, %10.4g, %d", arrLen, (double)CPNS * 1.0e9 * elapsed_cpu, ITERATIONS);
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
  CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, alloc_size, cudaMemcpyHostToDevice));

  // Launch the kernel
  cudaPrintfInit();
  dim3 dimGrid(ceil(SM_ARR_LEN/16), ceil(SM_ARR_LEN/16));   // Shape of grid = # of elements in a row divided by the number of threads per block row
  dim3 dimBlock(16, 16);    // Each block has shape of <16, 16>, so 256 threads/block
  // So each grid has shape (128 x 128), and each block has shape (16, 16)
  
  int work_on_part_2 = 0;
  if (work_on_part_2){    // Running code for lab7 part 2
    printf("==============>Running the result for part2!\n");
    // CUDA_SOR<<<1, NUM_THREADS_PER_BLOCK>>>(arrLen, d_data);
    cuda_single_block_SOR<<<1, dimBlock>>>(arrLen, d_data, change);   // 2D block
  }else{
    int iters = 0;
    while (iters < ITERATIONS) {
      iters++;
      cuda_multi_block_SOR<<<dimGrid, dimBlock>>>(arrLen, d_data, change);
      // cuda_SOR<<<dimGrid, dimBlock>>>(arrLen, dev_data, change);
    }
    printf("    SOR() done after %d iters\n", iters);
  }
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

  // Comparing the result we obtained from kernel function and regular SOR:
  int errCount = 0;
  int zeroCount = 0;
  float *relative_error = (float *)malloc(alloc_size);
  for (i=0; i<arrLen*arrLen; i++){
    if (abs(h_data_gold[i] - h_data[i]) / h_data_gold[i] > TOL){
      // printf("FAILED in %d:\t%.8f\t%.8f\t%.8f %%\n", i, h_data_gold[i], h_data[i], relative_error[i]);
      errCount++;
    }
    if (h_data[i] == 0.0) {
      zeroCount++;
    }
  }

  // double error_rate = errCount/(arrLen*arrLen) * 100;
  if (errCount > 0) {
    printf("\n@ERROR: TEST FAILED: %d/%d results did not match\n", errCount, arrLen*arrLen);
  }else if (zeroCount > 0){
    printf("\n@ERROR: TEST FAILED: %d/%d results (from GPU) are zero\n", zeroCount, arrLen*arrLen);
  }else {
    printf("\nTEST PASSED: All results matched\n");
  }

  printf("\n\n");
  start_point = 125;
  // for(i = start_point; i < start_point+50; i++) {
  //   //idx, gt_value, computed_value, relative error
  //   printf("%d:\t%.8f\t%.8f\t%.8f %%\n", i, h_data_gold[i], h_data[i], relative_error[i]);
  // } 
  for(i = start_point; i < start_point+128; i++) {
    printf("%d:\t%.8f\t%.8f\n", i, h_data_gold[i], h_data[i]);
  }

  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_data));
  free(h_data);
  return 0;
} /* end main */



/*********************************/
void initializeArray1D(float *arr, int len, int seed) {
  int i;
  float randNum;
  srand(seed);

  for (i = 0; i < len; i++) {
    randNum = (float) rand();
    arr[i] = randNum;
  }
}

void initializeArray2D(float *arr, int len, int seed) {
  int i;
  srand(seed);

  if (len > 0){
    for (i = 0; i < len*len; i++) {
      arr[i] = (float) rand();
    }
  }
}

/************************************/

/* SOR */
void SOR(int rowlen, float *data){
  long int i, j, iters;
  double change = 1.0e10;   /* start w/ something big */

  printf("Hello from SOR in CPU");
  for(iters = 0; iters < ITERATIONS; iters++){
    for (i = 1; i <= rowlen-1; i++) {
      for (j = 1; j <= rowlen-1; j++) {
        if (i>=1 && i<=rowlen-2 && j>=1 && j<=rowlen-2){
          change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                            data[(i+1)*rowlen+j] +
                                            data[i*rowlen+j+1] +
                                            data[i*rowlen+j-1]);
          data[i*rowlen+j] -= change * OMEGA;
        }
      }
    }
  }
  printf("    SOR() done after %ld iters\n", iters);
}




/************
 * The idea for this implementation, is to chopping off the entire matrix by block/tile, so that each thread is responsible for a single block.
*/
__global__ void cuda_single_block_SOR(int rowlen, float *data, double change){
  // cuPrintf("Hello from cuda_single_block_SOR!\n");
  long int i, j, iters;

  // Number of element need to be evaluated for a rows and columns for each block
  const int rows_per_thread = SM_ARR_LEN / blockDim.x;
  const int cols_per_thread = SM_ARR_LEN / blockDim.y;

  // Row and columns start/ending indices for current threads
  const int threadStartId_row = threadIdx.x * rows_per_thread ;
  const int threadEndId_row =  (threadIdx.x+1) * rows_per_thread - 1;
  const int threadStartId_col = threadIdx.y * cols_per_thread;
  const int threadEndId_col = (threadIdx.y+1) * cols_per_thread - 1;
  for(iters=0; iters < ITERATIONS; iters++){
    for (i = threadStartId_row; i <= threadEndId_row; i++) {
      for (j = threadStartId_col; j <= threadEndId_col; j++) {
        if (i>=1 && i<=rowlen-2 && j>=1 && j<=rowlen-2){  // We need to check the boundary for each thread
          change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                    data[(i+1)*rowlen+j] +
                                    data[i*rowlen+j+1] +
                                    data[i*rowlen+j-1]);
          // __syncthreads();    // Must SYNC between read nad writes. Otherwise there is a race condition, and some threads may read data that has alread been written. ==> Otherwise, you won't get correct result!!
          data[i*rowlen+j] -= change * OMEGA;
          __syncthreads();
        }
      }
    }
  }
}


/* SOR 
  rowlen: the row length of matrix
  data: a pointer to data matrix
  Note: Image, kernel is the code that each threads will run. So, we don't really need two for loop --> You need to think of the programming model in a parallel version, not serial version, and that's how the code can be more scalable.
*/
__global__ void cuda_multi_block_SOR(int rowlen, float *data, double change){
  // cuPrintf("Hello from cuda_multi_block_SOR!\n");
  long int i, j;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  j = threadIdx.y + blockIdx.y * blockDim.y;
  /* Where I am? 
  Imagine that you want to flatten everything into 1D array.
    blockDim.x*blockDim.y*blockIdx.x --> This one bring you to the beginning of that block.
    blockDim.x*threadIdx.y --> This one bring you to the row of this block.
    threadIdx.x --> This one bring you to the col of this block
  */
  // long int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*blockIdx.x;
  // long int tid_top = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*blockIdx.x;
  // if (tid>=1 && tid <=(rowlen*rowlen)-2){  
  if (i>=1 && i<=rowlen-2 && j>=1 && j<=rowlen-2){ // We need to check the boundary for each thread
    change = data[i*rowlen+j] - .25 * (data[(i-1)*rowlen+j] +
                                      data[(i+1)*rowlen+j] +
                                      data[i*rowlen+j+1] +
                                      data[i*rowlen+j-1]);
    
    // cuPrintf("[i*rowlen+j], [data[(i-1)*rowlen+j], [(i+1)*rowlen+j], [i*rowlen+j+1], [i*rowlen+j-1]\n");
    // cuPrintf("[%d], [%d], [%d], [%d], [%d]\n", i*rowlen+j, (i-1)*rowlen+j, (i+1)*rowlen+j, i*rowlen+j+1, i*rowlen+j-1);
    // cuPrintf("[%d], [%d], [%d], [%d], [%d]\n", data[i*rowlen+j], data[(i-1)*rowlen+j], data[(i+1)*rowlen+j], data[i*rowlen+j+1], data[i*rowlen+j-1]);
    // cuPrintf("[i, j]: [%d, %d]\n", i, j);
    data[i*rowlen+j] -= change * OMEGA;
    // cuPrintf("data: %f\n", data[i*rowlen+j]);
  }
  //   }
  // }
}





