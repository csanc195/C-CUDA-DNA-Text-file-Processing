/*This file includes all memory helper functions*/

/**
 * allocateFinalArray64CPU 
 * This is a helper function to allocate an array of 64 spaces in CPU for 
 * final codone counts
 * @return [Pointer to created array]
 */
int *allocateFinalArray64CPU(){
	int *arrayPtr = (int*) calloc(64, sizeof(int));
	return arrayPtr;
}


/**
 * loadFileToRAM 
 * This helper function loads file to RAM as a Zero terminated string.
 * @param  fIn        [file provided through the command line]
 * @param  fileLength [Length of file in bytes]
 * @return            [Pointer to the memory where file is stores in RAM]
 */
char *loadFileToRAM(FILE * fIn, long fileLength){
	
	char * memoryPtr;
	char current;

	memoryPtr = (char*) calloc(1, fileLength * sizeof(char) + 1);

	//fill memory
	char *tempPtr = memoryPtr;
	while((current = fgetc(fIn)) != EOF){
		*tempPtr = current;
		tempPtr++;
	}
	return memoryPtr;
}


/**
 * loadFileToCUDA 
 * This helper function allocates the needed space in CUDA and fill that 
 * memory with the contents of the codone file currently in CPU RAM. 
 * @param  memPtr     [Pointer to the file in RAM]
 * @param  fileLength [File Length]
 * @return            [Pointer to CUDA memory allocated with file's content]
 */
char* loadFileToCUDA(char *memPtr, long fileLength){

	char *gpuA;
	cudaMalloc(&gpuA, fileLength*sizeof(char));

	char *tempPtr = gpuA;
	/* Fill CUDA memory with the contents of RAM */
	cudaMemcpy(tempPtr , memPtr, fileLength*sizeof(char), cudaMemcpyHostToDevice);

	return gpuA;
}


/**
 * getFileLength 
 * This helper function is used to determine the amount of characters on
 * the file. Leading and trailing character will be counted as well. ex "S"
 * prints as having length 3.
 * 
 * @param  fIn [file provided through the command line]
 * @return     [file Lenght in bytes]
 */
long getFileLength(FILE *fIn){
	long fileLength = 0;
	fseek( fIn, 0L , SEEK_END);
	fileLength = ftell(fIn);
	rewind(fIn);
	return fileLength;
}


/**
 * createFinalArray64CUDA
 * This helper function allocates memory in CUDA to store the final
 * codones count. After memory is allocated all its indexes are set to '0'
 * @return [Pointer to allocated CUDA memory]
 */
int *createFinalArray64CUDA(){
	int *gpuPtr;
	cudaMalloc((int**)&gpuPtr, 64 * sizeof(int));
	cudaMemset(gpuPtr, 0, 64 * sizeof(int));

	return gpuPtr;
}


/**
 * allocateArrayInCuda 
 * This helper function allocates memory in CUDA. It allocates enough 
 * lineal memory so that each thread has 64 spaces to count the occurrences  
 * of each codones it was responsible for. 
 * @param  numThreads [Maximum  number of threads needed to find all codones
 *                    in file. Each thread ell be responsible for analyzing  
 *                    64 codone sequentially]
 * @return            [Pointer to the allocated memory]
 */
int *allocateArrayInCuda(long numThreads){
	
	long size = numThreads*64*sizeof(int);
	int *gpuA;

	cudaError_t myCudaError = cudaMalloc((int **)&gpuA,size );

	if (myCudaError == 0){
		cudaMemset(gpuA, 0, size);
		return gpuA;
	}
	else{
		printf("%s%d\n", "Error while allocation memory in Cuda", myCudaError);
	}

	return 0;
}


long double printFileSize(long numOfBytes){
	 return (double)((numOfBytes)/1000000.00); //formula to convert from bits to MB
}
