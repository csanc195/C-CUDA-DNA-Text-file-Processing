/*This file contains the lineal program */


/**
 * evaluateLetterCPU  
 * This function evaluates the letter value knowing that A=0,C=1,G=2,T=4
 * Using formula: letterValue * (4 ^ position).
 *  
 * So GTA = (2*4^2) + (3* 4^1) + (0*4^0) = 44 will increment array[64] by 1.
 * 
 * @param  letter [Upper case character: A|C|G|T ]
 * @param  pos    [position of the char in the codone Ex: in ACT pos(A) = 2 ]
 * @return        [returns the value of the letter after polynomial evaluation]
 */
int evaluateLetterCPU(char letter, int pos){
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



/**
 * calculateCodoneValueCPU 
 * This function takes a 3 letter string (1 codone) and returns its polynomial 
 * value. The value obtained is used as the index of the 64 array to increment 
 * the number of times the specific codone is found. 
 *
 * Given example codone GTA
 * 
 * @param  letter1 [First letter of the codone: G]
 * @param  letter2 [Second letter of the codone: T]
 * @param  letter3 [Third letter of the codone: A]
 * @return         [The value of the codone as a number. It's used as the index
 *                  for the final array with all occurrences for each codone]
 */
int calculateCodoneValueCPU(char letter1, char letter2, char letter3){
 	int indexValue;

	indexValue = evaluateLetterCPU(letter1, 2) + 
				 evaluateLetterCPU(letter2, 1) + 
				 evaluateLetterCPU(letter3, 0);

	return indexValue;
}



/**
 * parseFileCPU 
 * This function processes a file that contains x number of codones. Every 
 * time a codone is found its value is analyzed by helper a function. 
 * The conode value is used to locate the final results array index and 
 * increment it's value by 1 just like a counter.  
 * 
 * @param fileCPUPtr      [Pointer to the codones file in RAM]
 * @param numCodones      [Number of codones to be processed]
 * @param resultsArrayPtr [Pointer to the results array that contains the 
 *                        number of times each codone is found]
 */
void parseFileCPU(char *fileCPUPtr, long numCodones, int *resultsArrayPtr){

	int count = 0;
	while(count < numCodones){

		int codoneIndex = 3 * count;
		int codoneValue = calculateCodoneValueCPU(fileCPUPtr[codoneIndex], 
												 fileCPUPtr[codoneIndex + 1], 
												 fileCPUPtr[codoneIndex + 2]); 
		resultsArrayPtr[codoneValue]++;
		count++;
	}
}