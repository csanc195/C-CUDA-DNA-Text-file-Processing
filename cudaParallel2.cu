/*This file contains all GPU functions*/


/**evaluateLetter  
 * This helper function evaluates the letter value knowing that A=0,C=1,G=2,T=4
 * Using formula: letterValue * (4 ^ position).
 *  
 * So GTA = (2*4^2) + (3* 4^1) + (0*4^0) = 44 will increment array[64] by 1.
 * 
 * @param  letter [Upper case character: A|C|G|T ]
 * @param  pos    [position of the char in the codone Ex: in ACT pos(A) = 2 ]
 * @return        [returns the value of the letter after polynomial evaluation]
 */
__device__ int evaluateLetter(char letter, int pos){
	int letterValue;

	if(letter == 'A'){
		letterValue = 0;
	} else if(letter == 'C'){
		letterValue = 1;
	}else if(letter == 'G'){
		letterValue = 2;
	}else if(letter == 'T'){
		letterValue = 3;
	}

	return letterValue * powf(4, pos);
}

/* */

/**
 * calculateIndexValue
 * This function takes a 3 letter string (1 codone) and returns its polynomial 
 * value. The value obtained is used as the index of the 64 final
 * array to increment the number of times the specific codone is found. This
 * function is a device level function because it can be called by any block.
 *
 * Given example codone GTA
 * 
 * @param  letter1 [First letter of the codone: G]
 * @param  letter2 [Second letter of the codone: T]
 * @param  letter3 [Third letter of the codone: A]
 * @return         [The value of the codone as a number. It's used as the index
 *                  for the final array with the occurrences for each codone]
 */
__device__ int calculateIndexValue(char letter1, char letter2, char letter3){
	int indexValue;

	indexValue = evaluateLetter(letter1, 2) + 
				 evaluateLetter(letter2, 1) + 
			     evaluateLetter(letter3, 0);

	return indexValue;
}


/**
 * parseArray 
 * This function will be executed by all threads launched to manipulate the 
 * codone file already copied to the GPU memory. Each thread will be 
 * responsible for processing 64 codones to avoid occupying too much space in
 * memory. For each codone, there will be a call to a helper function to 
 * determine the codone's value and using that value increment the counter for
 * the specified codone on the array index belonging to the current thread.
 * 
 * @param cudaStrPtr         [Pointer to cuda memory where file is copied]
 * @param cudaCodoneOccTable [Pointer to the cuda array where each thread will 
 *                           will have 64 spots to increment the counter for
 *                           the found codone]
 * @param maxNumThreads      [Max number of threads the entire program needs. 
 *                           Using this value we will control when to stop the
 *                           threas for a specific core]
 * @param lenght             [Number of chars of file contains. This value is
 *                           used to determine of the current thread needs to
 *                           handle less that 64 codones because it's getting
 *                           closed to EOF].
 */
__global__ void parseArray(char *cudaStrPtr, int *cudaCodoneOccTable, 
							long maxNumThreads, long lenght){

	long threadNumIndex = (blockIdx.x*blockDim.x) + threadIdx.x;
	long charIndex = threadNumIndex * 192; //64 codones * 3 letters each
	long currentLocalIndex = charIndex;

	if (threadNumIndex < maxNumThreads){
		int count = 0;
		while ((count < 192) && (currentLocalIndex < lenght)){

			int tableIndex = calculateIndexValue(cudaStrPtr[currentLocalIndex], 
											cudaStrPtr[currentLocalIndex + 1], 
										    cudaStrPtr[currentLocalIndex + 2]);
			int position = (threadNumIndex*64) + tableIndex;
			cudaCodoneOccTable[position] += 1;
			currentLocalIndex += 3;
			count +=3;
		}
	}
}

	
/**
 * tableVectorAdd
 * This function performs vector addition to calculate the number of occurrences
 * of all codones. It stores the final count on a new array of size 64 (64 
 * is the max number of possible combinations of 3 letters using A C G T).
 * This function will be used by 64 threads, each thread will add contents of 
 * the table codoneTable. 
 * 
 * @param tablePtr     [Pointer to the table with codone occurrences]
 * @param numRows      [This is a number of threads used to count the 
 *                     occurrences of all codones. Each thread has 64 available
 *                     spaces to write it's findings]
 * @param finalArray64 [Pointer to the final array of size 64 that will 
 *                     contain the final count for each codone]
 */
__global__ void tableVectorAdd(int *tablePtr, long numRows, int * finalArray64){

	long threadNum = (blockIdx.x*blockDim.x) + threadIdx.x;
	long i, position; 
	long sum = 0;

	for (i = 0; i < numRows; ++i)
	{
		position = (i*64) + threadNum;
		sum += tablePtr[position];
	}

	finalArray64[threadNum] = sum;
}

