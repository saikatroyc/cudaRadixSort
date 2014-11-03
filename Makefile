
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = radix
OBJ	        = main.o support.o

default: $(EXE)

test-mode: NVCC_FLAGS += -DTEST_MODE
test-mode: default

main.o: main.cu kernel_radix.cu kernel_prescan.cu defs.h support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

