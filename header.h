#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>
#include <sys/time.h>
#include <math.h>
#include "cuPrintf.cu"
#include "cudaParallel2.cu"
#include "memoryHelpers.cu"
#include "lineal.cu"


#define TABLE_HEADER "+------------------------------------------------------------------------+"

const char * codones[64] = {"AAA","AAC","AAG","AAT","ACA","ACC","ACG","ACT","AGA",
     				  "AGC","AGG","AGT","ATA","ATC","ATG","ATT","CAA","CAC",
					  "CAG","CAT","CCA","CCC","CCG","CCT","CGA","CGC","CGG",
					  "CGT","CTA","CTC","CTG","CTT","GAA","GAC","GAG","GAT",
					  "GCA","GCC","GCG","GCT","GGA","GGC","GGG","GGT","GTA",
					  "GTC","GTG","GTT","TAA","TAC","TAG","TAT","TCA","TCC",
					  "TCG","TCT","TGA","TGC","TGG","TGT","TTA","TTC","TTG",
					  "TTT"};  
