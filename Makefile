CUDA_INSTALL_PATH = /opt/cuda
CC = g++
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common
NMAX            ?= 964
NCPU            ?= 402
R_EXCH			?= 2
GRANULARITY     ?= 1
PRECALC			?= 1
RECALC_LEVEL    ?= 1
KERNEL_TYPE     ?= 2
DEFS = -DNMAX=$(NMAX) -DNCPU=$(NCPU) -DR_EXCH=$(R_EXCH) -DGRANULARITY=$(GRANULARITY) -DPRECALC=$(PRECALC) -DRECALC_LEVEL=$(RECALC_LEVEL) -DKERNEL_TYPE=$(KERNEL_TYPE)
OPTFLAG = -Xcompiler -O2 -Xcompiler -fopenmp -Xcompiler -fomit-frame-pointer -Xcompiler -ftree-vectorize -Xcompiler -ftree-vectorizer-verbose=1 -Xcompiler -funroll-loops -Xcompiler -fassociative-math -Xcompiler -fno-signed-zeros -Xcompiler -fno-trapping-math -Xcompiler -fno-signaling-nans -Xcompiler -ffast-math -Xcompiler -fassociative-math -Xcompiler -msse2 -Xcompiler -msse -Xcompiler -mfpmath=sse
NVFLAGS = ${DEFS} ${NVDEFS} -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing ${OPTFLAG} --ptxas-options=-v --use_fast_math -g ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = ${PROFLAG} -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lgomp -lstdc++ -lcudart -lrt
NVCODE_DP = -gencode=arch=compute_20,code=\"compute_20\"
# -gencode=arch=compute_30,code=\"compute_30\" -gencode=arch=compute_35,code=\"compute_35\"
CLEAN_FILES = lmsor_rb_omp_gpu lmsor_rb_omp_gpu.o

all: lmsor_rb_omp_gpu lmsor_rb_omp_gpu_sse2

lmsor_rb_omp_gpu_sse2: lmsor_rb_omp_gpu_sse2.o
	${NVCC} -o $@ $^ ${LFLAGS}

lmsor_rb_omp_gpu: lmsor_rb_omp_gpu.o
	${NVCC} -o $@ $^ ${LFLAGS}
	
lmsor_rb_omp_gpu_sse2.o: lmsor_rb_omp_gpu.cu
	${NVCC} ${NVCODE_DP} ${NVFLAGS} -DMODIFIED_SOR -DBLOCK_PARTITIONING -D_INTRINSIC_SSE2_ -c $< -o $@

lmsor_rb_omp_gpu.o: lmsor_rb_omp_gpu.cu
	${NVCC} ${NVCODE_DP} ${NVFLAGS} -DMODIFIED_SOR -DBLOCK_PARTITIONING -c $< -o $@

clean:
	\rm -f $(CLEAN_FILES)

rebuild: clean all

