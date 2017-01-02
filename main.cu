#include "header.h"

/**
 * printResults 
 * This is a helper function that prints each codone with its occurences. 
 * It prints as a table format of 4 columns and 16 rows.  
 * 
 * @param cpuArrayPtr [Pointer the the array with the final counts]
 */
void printResults(int *cpuArrayPtr){	

	int i;
	printf("%s\n", TABLE_HEADER); 
	for (i = 0; i < 64; i++)
	{	
		printf("|%.4s %7d |      ", codones[i], cpuArrayPtr[i]);

		if ((i+1)%4 == 0){
			printf("\n");
		}

	}
	printf("%s\n",TABLE_HEADER); 
}



/**
 * main 
 * The following program does not take any input through  the command line
 * It determines the number of occurrences of codones in a file in parallel
 * and in lineal. For both approaches the files with the information has to be 
 * loaded from Hard Drive to RAM. Also, an array of size 64 has to be allocated
 * in memory to hold the final count of each 64 possible codone combinations
 * given by sequences of 3 letter from the following list: A,C,G,T
 *
 * Parallel Approach: 1.Copy file from RAM to cuda 2.Allocate enough memory 
 * in CUDA to store an array of size (# of total threads * 64 * size of char)
 * 3.Allocate an array of size 64 in CUDA to keep final counts. 4.Launching 
 * the correct number of threads, each thread will take 64 codones, evaluate
 * their values and increment the count on the thread's corresponding section
 * of the bog array. 5.After all threads are done, launch another 64 threads to 
 * vector add all codone occurrences found. 6.Copy final array back to CPU. 
 * 7.Deallocate memory used in CUDA
 *
 * Lineal Approach: Ready 1 codone at a time from file already uploaded in 
 * RAM and update the results array by incrementing the index related to the
 * found codone. 
 * 
 * @param  argc [not used]
 * @param  argv [not used]
 * @return      [0 if no errors]
 */
int main(int argc, char const *argv[])
{
	/* This group of variables is used to calculate execution time */
	clock_t start, end;
	double gpu_time_used = 0.0;
	double cpu_time_used = 0.0;

	/*Variable declarations */
	long length = getFileLength(stdin); 
	long numCondones = length/3;
	long maxNumThreads = (numCondones/64) + 1;
	long numCores = (maxNumThreads/1024) + 1;

	/*Create a 64 int array in CPU for final answers */
	int *cpuArrayPtr = allocateFinalArray64CPU();


	/*Load codones file from Hard Drive to RAM*/
	char *ramPtr = loadFileToRAM(stdin, length);


	/*Print details about the file, threas and cores */
	printf("\n\n%s%ld\n%s%Lf\n%s%ld\n%s\n%s%ld\n\n",
		  "Length in characters:", length, "File size in MB: ", 
		   printFileSize(length), "Theads needed: ", maxNumThreads, 
		   "Threads used per core: 1024", "Cores used: ", numCores);

/***************************************************************************
*****************************PARALLEL CODE START****************************
****************************************************************************/

	start = clock();

	/*Create a 64 int array in CUDA for final answers*/
	int *responseArrayptrCUDA = createFinalArray64CUDA();

	/* Move file from RAM to CUDA*/
	char *cudaMemPtr = loadFileToCUDA(ramPtr, length);

	/* Create a global GPU array. Allocate the needed space in cuda so that
	each thread has 64 spaces to record codone matches*/
	int *threadsTable = allocateArrayInCuda(maxNumThreads);

	/*Use GPU to count codone occurences*/
	parseArray<<< numCores, 1024>>>(cudaMemPtr, threadsTable, maxNumThreads, 
									length);

	/*Use GPU to vector add all codone occurences*/
	tableVectorAdd<<<1, 64>>>(threadsTable, maxNumThreads, 
							 responseArrayptrCUDA);

	/*Copy results array from GPU to CPU*/
	cudaMemcpy(cpuArrayPtr, responseArrayptrCUDA, 64*sizeof(int),
			   cudaMemcpyDeviceToHost);

	/*Free cuda memory that contains file*/
	cudaFree(&cudaMemPtr); 

	/*Free cuda memory that contains big threads table*/
	cudaFree(&threadsTable);

	/*Free cuda memory that contains 64 array with final results*/	
	cudaFree(&responseArrayptrCUDA);


	end = clock();

	gpu_time_used = (double)(end - start)/CLOCKS_PER_SEC;
	gpu_time_used = gpu_time_used * 1000;

/***************************************************************************
*****************************PARALLEL CODE END******************************
****************************************************************************/


	/*Print results from parallel program*/
	printf("%s\n", TABLE_HEADER);
	printf("%s\n","Solving problem in parallel......" );
	printf("%s%lf%s\n", "Total time used for GPU: ", gpu_time_used, "ms" );
	printResults(cpuArrayPtr);


	/*clear 64 array after being used by CUDa and before using it in lineal*/
	memset(cpuArrayPtr, 0, 64*sizeof(int));


/***************************************************************************
*****************************LINEAL CODE STARTS******************************
****************************************************************************/
	
	start = clock();

	/*Count the number of codone occurances in a file*/
	parseFileCPU( ramPtr, numCondones, cpuArrayPtr);
	
	end = clock();

	cpu_time_used = (double)(end - start)/CLOCKS_PER_SEC;
	cpu_time_used = cpu_time_used * 1000;

/***************************************************************************
*****************************LINEAL CODE END*********************************
****************************************************************************/
	

	/*Print results from lineal program*/
	printf("\n%s\n", TABLE_HEADER);
	printf("%s\n","Solving problem lineal......" );
	printf("%s%lf%s\n", "Total time used for CPU: ", cpu_time_used, " ms" );
	printResults(cpuArrayPtr);

	/*Free final array 64 in CPU*/
	free(cpuArrayPtr);

	return 0;

} // end of main






