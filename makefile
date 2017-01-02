CC = nvcc

FILES = main.cu memoryHelpers.cu cudaParallel.cu
#FLAGS = -Wall -w
OBJ = main.o memoryHelpers.o cudaParallel.o
OUT_EXEC = exec.out

build:$(OBJ)
	$(CC) $(FILES) -o $(OUT_EXEC)

clean:	
	rm -f *.o

rebuild: clean	build