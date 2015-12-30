/* Local Modified SOR_5pt (LMSOR_RB)  Missirlis and Tzaferis (3.11.2002)*/
/* red\black ordering */
/* optimum  w1_ij, w2_ij */
/* Solving of the second order Convection Diffusion PDE */
/*---------------------------------------------------*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include "lcutil.h"
#include "timestamp.h"
#include "omp.h"

#if (defined __SSE2__) || (_M_IX86_FP == 2)
#include <emmintrin.h>
#define __SIMD_SUPPORTED__
#endif

#if defined(_INTRINSIC_SSE2_) && (!PRECALC || !defined(__SIMD_SUPPORTED__))
#error "Can not use _INTRINSIC_SSE2_ without PRECALC == 1 and __SIMD_SUPPORTED__"
#endif

#if R_EXCH % 2 != 0
#error "Number of rows exchanged must be multiple of 2"
#endif

#ifdef _MSC_VER
#define isnan(x) _isnan(x)  // VC++ uses _isnan() instead of isnan()
#endif

#ifdef _OPENMP
#ifndef BLOCK_PARTITIONING
#error "Only block partitioning is implemented"
#endif
#endif

extern __shared__ double d_shmem_base_ptr[];

/*
 RECALC_LEVEL:
 0: No redundant calculations
 1: Recalculate r,l,b,t
 2: Recalculate all factors (including w)
 */
#ifndef RECALC_LEVEL
#define RECALC_LEVEL 0
#endif
/*
 KERNEL_TYPE:
 1: Global memory only
 2: Texture memory (for neighbor elements)
 3: Shared memory data sharing (for neighbor elements)
 */
#ifndef KERNEL_TYPE
#define KERNEL_TYPE 1
#endif
/*
 PREF_SHARED_MEM:
 true:  Prefer shared memory (48KB shared memory/16KB L1 cache)
 false: Prefer cache memory (16KB shared memory/48KB L1 cache)
 */
#ifndef PREF_SHARED_MEM
#define PREF_SHARED_MEM true
#endif

/*
 #ifndef MODIFIED_SOR
 #error "Only modified SOR is implemented"
 #endif
 */

#define PI 3.141592653589793

/*
 #define _DBLS_PER_CACHE_LINE_ (64/sizeof(double)) // count of doubles that fit in a cache line
 #define _ALIGN_SZ_(x,y) ((x+y-1)/y)*y
 */

//redblack arrays on host
double ro_u_host[2][NCPU][NMAX / 2];
double ro_w1_host[2][NCPU][NMAX / 2];
double ro_ffh_host[2][NCPU][NMAX / 2];
double ro_ggh_host[2][NCPU][NMAX / 2];
double ro_l_host[2][NCPU][NMAX / 2];
double ro_r_host[2][NCPU][NMAX / 2];
double ro_t_host[2][NCPU][NMAX / 2];
double ro_b_host[2][NCPU][NMAX / 2];

/* large matrices moved to global space so not to overload stack */
typedef double lattice[NMAX];
lattice *w1 = NULL, *w2 = NULL; //, *m_low, *m_up;
lattice *u = NULL;
lattice *l = NULL, *r = NULL;
lattice *b = NULL, *t = NULL;
lattice *ffh = NULL, *ggh = NULL;
lattice *w1_host = NULL;
lattice *w1_dev = NULL;
lattice *u_host = NULL;
lattice *u_dev = NULL;
lattice *l_host = NULL;
lattice *l_dev = NULL;
lattice *r_host = NULL;
lattice *r_dev = NULL;
lattice *b_host = NULL;
lattice *b_dev = NULL;
lattice *t_host = NULL;
lattice *t_dev = NULL;
lattice *ffh_host = NULL;
lattice *ffh_dev = NULL;
lattice *ggh_host = NULL;
lattice *ggh_dev = NULL;

#if KERNEL_TYPE==2
typedef texture<int2, 2, cudaReadModeElementType> texture2Ddouble;
texture2Ddouble texURed, texUBlack;
__inline__ __device__ double tex2Ddouble(const texture2Ddouble tx, const int x,
		const int y) {
	int2 v = tex2D(tx, x, y);
	return __hiloint2double(v.y, v.x);
}
cudaError_t bindTextureDP(const texture2Ddouble &texData, double *dataD, int X,
		int Y, int pitch) {
	static const cudaChannelFormatDesc channelDesc =
	cudaCreateChannelDesc<int2>();
	return cudaBindTexture2D(NULL, texData, dataD, channelDesc, X, Y, pitch);
}

cudaError_t unbindTextureDP(const texture2Ddouble &texData) {
	return cudaUnbindTexture(texData);
}
#elif KERNEL_TYPE==3
//
#else
//
#endif

double FF(int epil, double r1, double x1, double y1);
double GG(int epil, double r1, double x1, double y1);
double initial_guess(double x1, double y1);
void min_max_MAT(double MAT[][NMAX], int n, double *min_MAT, double *max_MAT);
void min_max(double value, double *min_, double *max_);

void mypause(void) {
	printf("Press \"Enter\"\n");
	timestamp ts = getTimestamp();
	do {
		getchar();
	} while (getElapsedtime(ts) < 100.0f);
}

long int pow(int b, int e) {
	long int r = 1;
	for (int i = 0; i < e; i++)
		r *= b;
	return r;
}

double pow2(double v) {
	return v * v;
}

void validate_w(double w) {
	if (isnan(w)) {
		printf("ERROR: Invalid w value\n");
		exit(1);
	}
}

int getFPDevSize(void) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999)
		return 0;

	return (deviceProp.major == 1 && deviceProp.minor < 3) ? 1 : 2;
}

template<int exponent>
__device__ inline double pow_ce(double v) {
	double res = 1.0;
	for (int i = 0; i < exponent; i++)
		res *= v;
	return res;
}

__device__ inline double dsquare(double v) {
	return v * v;
}

// atomic exchange of a double precision value
__device__ double atomicExch_(double *address, double val) {
	unsigned long long int ival = *(unsigned long long int*) (&val);
	unsigned long long int ires = atomicExch((unsigned long long int*) address,
			ival);
	return *(double*) (&ires);
}

// atomic add for double precision implementation
__device__ double atomicMax_(double *address, double val) {
	double oldval;
	while (val != CUDART_MIN_DENORM) {
		oldval = atomicExch_(address, CUDART_MIN_DENORM);
		oldval = val = max(val, oldval);
		val = atomicExch_(address, val);
	}
	return oldval;
}
__device__ double atomicAdd_(double *address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

#ifdef __DEVICE_EMULATION__
#error "DOES NOT SUPPORT DEVICE EMULATION!"
#endif
// Performs reduction in shared memory of double precision elements
inline __device__ void shmem_reduction(unsigned int tid,
		volatile double *sdata) {
	const unsigned int BLOCK_SIZE = blockDim.x * blockDim.y;
	if (BLOCK_SIZE > 1024) {
		if (tid < min(1024, BLOCK_SIZE - 1024)) {
			sdata[tid] += sdata[tid + 1024];
		}
		__syncthreads();
	}
	if (BLOCK_SIZE > 512) {
		if (tid < min(512, BLOCK_SIZE - 512)) {
			sdata[tid] += sdata[tid + 512];
		}
		__syncthreads();
	}
	if (BLOCK_SIZE > 256) {
		if (tid < min(256, BLOCK_SIZE - 256)) {
			sdata[tid] += sdata[tid + 256];
		}
		__syncthreads();
	}
	if (BLOCK_SIZE > 128) {
		if (tid < min(128, BLOCK_SIZE - 128)) {
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if (BLOCK_SIZE > 64) {
		if (tid < min(64, BLOCK_SIZE - 64)) {
			sdata[tid] += sdata[tid + 64];
		}
		__syncthreads();
	}
	if (tid < 32) {
		if (BLOCK_SIZE > 32) {
			sdata[tid] += sdata[tid + 32];
		}
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}
}

/* performs error value reduction */__global__ void kreduceDP(double *g_idata,
		double *g_odata, unsigned int n) {
	const unsigned int BLOCK_SIZE = blockDim.x * blockDim.y;
	volatile double *shmdata = d_shmem_base_ptr;
	// perform first level of reduction
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;
	unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;
	shmdata[tid] = 0.0;
	while (i < n) {
		shmdata[tid] += g_idata[i];
		if (i + BLOCK_SIZE < n)
			shmdata[tid] += g_idata[i + BLOCK_SIZE];
		i += gridSize;
	}
	__syncthreads();
	shmem_reduction(tid, shmdata);
	if (tid == 0)
		atomicAdd_(g_odata, shmdata[0]);
}

template<int granularity, int calc_red>
__global__ void klmsorDP(const unsigned int N, const unsigned int pitch,
		const double * __restrict__ dSrcU, double * __restrict__ dDstU,
#if RECALC_LEVEL==0
		const double * __restrict__ dL, const double * __restrict__ dR,
		const double * __restrict__ dB, const double * __restrict__ dT,
#else                // 1 or 2
		const double * __restrict__ dFFh, const double * __restrict__ dGGh,
#endif
#if RECALC_LEVEL!=2  // 0 or 1
		const double * __restrict__ dW,
#endif
		double * __restrict__ sqrerrorD) {
	double *shm_sqrerr = d_shmem_base_ptr;
#if KERNEL_TYPE==3
	double *shm_SrcU = &shm_sqrerr[ blockDim.x*blockDim.y ];
	const int shm_width = blockDim.x + 2;
#elif KERNEL_TYPE==2
	const texture2Ddouble texSrcU = calc_red ? texUBlack : texURed;
#endif
	int iy = (blockIdx.y * blockDim.y + threadIdx.y) * granularity + 1;
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
	double sqrerror = 0.0;
	int ptrid = ix + iy * pitch;

#if KERNEL_TYPE==3
	// fetch U data into shared memory
	int shm_index = shm_width*granularity*threadIdx.y+threadIdx.x+1+shm_width;
	if( ix<N/2 && iy<N - NCPU -granularity ) {
		// fetch inner points
#pragma unroll
		for(int i=0; i<granularity; i++)
		shm_SrcU[shm_index+i*shm_width] = dSrcU[ptrid+i*pitch];
		// fetch upper & lower halo points
		if( threadIdx.y==0 ) {
			shm_SrcU[shm_index-shm_width] = dSrcU[ptrid-pitch];
			const int yoffs = (int)min(N-NCPU-1-iy, granularity*blockDim.y);
			shm_SrcU[shm_index+yoffs*shm_width] = dSrcU[ptrid+yoffs*pitch];
		}
		// fetch left & right halo points
		if( threadIdx.x==0 ) {
			int xoffs = (calc_red+iy)%2 ? -1 : (int)blockDim.x;
#pragma unroll
			for(int i=0; i<granularity; i++) {
				shm_SrcU[shm_index+i*shm_width+xoffs] = dSrcU[ptrid+i*pitch+xoffs];
				xoffs = blockDim.x-1-xoffs;
			}
		}
	}
	__syncthreads();
#endif

	int line_indicator =
	granularity % 2 == 0 ? (1 - calc_red) : (calc_red + iy) % 2;
	if (ix < N / 2 && iy < N - NCPU - granularity) {
#pragma unroll
		for (int i = 0; i < granularity; i++) {
			if (ix >= line_indicator && ix < N / 2 - (1 - line_indicator)) {
				double oldv = dDstU[ptrid];
#if RECALC_LEVEL==0
				double r = dR[ptrid];
				double l = dL[ptrid];
				double t = dT[ptrid];
				double b = dB[ptrid];
#else
				const double quarter = 0.25;
				double r = (1.-dFFh[ptrid])*quarter;
				double l = (1.+dFFh[ptrid])*quarter;
				double t = (1.-dGGh[ptrid])*quarter;
				double b = (1.+dGGh[ptrid])*quarter;
#endif
#if RECALC_LEVEL==2
				// w value recalculation
				const double h = 1./(N-1.0);
				int eigen_case = 0;
				if (r*l*t*b>=0) {
					if ( r*l>=0 && t*b>=0 )
					eigen_case = 1;
					else
					if ( r*l<=0 && t*b<=0 )
					eigen_case = 2;
				} else {
					if ( r*l>0 && r+l>=0)
					eigen_case = 3;
				}
				double w, m_up, m_low;
				switch(eigen_case) {
					case 1:
					// case 1 (Im(m)=0) real
					m_up = 2.f*(__fsqrt_rn(r*l)+__fsqrt_rn(t*b))*__cosf(PI*h);
					m_low = 2.f*(__fsqrt_rn(r*l)+__fsqrt_rn(t*b))*__cosf(PI*(1.-h)/2.);
					w = 2. * ( calc_red ?
							__frcp_rn(1.f-m_up*m_low+__fsqrt_rn((1.f-m_up)*(1.f-m_low))) :
							__frcp_rn(1.f+m_up*m_low+__fsqrt_rn((1.f-m_up)*(1.f-m_low))) );
					break;
					case 2:
					// case 2 (Re(m)=0) imaginary
					m_up = 2.*(__fsqrt_rn(-r*l)+__fsqrt_rn(-t*b))*__cosf(PI*h);
					m_low = 2.*(__fsqrt_rn(-r*l)+__fsqrt_rn(-t*b))*__cosf(PI*(1.-h)/2.);
					w = 2. * ( calc_red ?
							__frcp_rn(1.f-m_up*m_low+__fsqrt_rn((1.f+pow_ce<2>(m_up))*(1.f+pow_ce<2>(m_low)))) :
							__frcp_rn(1.f+m_up*m_low+__fsqrt_rn((1.f+pow_ce<2>(m_up))*(1.f+pow_ce<2>(m_low)))) );
					break;
					case 3:
					// case 3a  complex
					w = 2. * __frcp_rn( (1.+__powf((1.f-__powf((r+l), 2.f/3.f)),-1.f/2.f)*fabs(t-b)) );
					break;
					default:
					// case 3b complex
					w = 2. * __frcp_rn( (1.f+__powf((1.f-__powf((t+b),2.f/3.f)),-1.f/2.f)*fabs(r-l)) );
					break;
				}
#else
				double w = dW[ptrid];
#endif

				// compute new element value
#if KERNEL_TYPE==2
				double newv = ( 1. - w )*oldv + w*( l*tex2Ddouble(texSrcU, ix, iy-1) +
						r*tex2Ddouble(texSrcU, ix, iy+1) +
						b*tex2Ddouble(texSrcU, ix-line_indicator, iy) +
						t*tex2Ddouble(texSrcU, ix+1-line_indicator, iy) );
#elif KERNEL_TYPE==3
				double newv = ( 1. - w )*oldv + w*( l*shm_SrcU[shm_index-shm_width] +
						r*shm_SrcU[shm_index+shm_width] +
						b*shm_SrcU[shm_index-line_indicator] +
						t*shm_SrcU[shm_index+1-line_indicator] );
#else
				double newv =
				(1. - w) * oldv
				+ w
				* (l * dSrcU[ptrid - pitch]
						+ r * dSrcU[ptrid + pitch]
						+ b
						* dSrcU[ptrid
						- line_indicator]
						+ t
						* dSrcU[ptrid + 1
						- line_indicator]);
#endif
				dDstU[ptrid] = newv;
				sqrerror += dsquare(oldv - newv);
			}
			line_indicator = 1 - line_indicator;
			ptrid += pitch;
#if KERNEL_TYPE==3
			shm_index += shm_width;
#elif KERNEL_TYPE==2
			iy++;
#endif
		}
	}
	shm_sqrerr[tid] = sqrerror;

	// error reduction
	__syncthreads();
	shmem_reduction(tid, shm_sqrerr);
	if (tid == 0)
	sqrerrorD[blockIdx.y * gridDim.x + blockIdx.x] = shm_sqrerr[0];
}

__global__ void kreorderDP(double *dataD, int element_pitch, int N,
		double *reorder_bufferD) {
	double *buffer = d_shmem_base_ptr;
	const int tid = threadIdx.x;
	const int halfblock = blockDim.x / 2;
	int line = blockIdx.x;
	int black_thread = (tid >= halfblock);
	while (line < (N - NCPU)) {
		int am_i_black = (line + tid) % 2;
		int base = 0;
		while (base < N) {
			if (base + tid < N) {
				int newpos = tid >> 1;
				if (am_i_black)
					newpos += halfblock;
				buffer[newpos] = dataD[element_pitch * line + base + tid];
			}
			__syncthreads();
			int final_line_pos = element_pitch / 2 * black_thread + base / 2
					+ (tid - halfblock * black_thread);
			if (tid - halfblock * black_thread < (N - base) / 2)
				reorder_bufferD[blockIdx.x * element_pitch + final_line_pos] =
						buffer[tid];
			base += blockDim.x;
			__syncthreads();
		}
		base = 0;
		while (base + tid < N / 2) {
			dataD[line * element_pitch + base + tid] =
					reorder_bufferD[blockIdx.x * element_pitch + base + tid];
			dataD[line * element_pitch + element_pitch / 2 + base + tid] =
					reorder_bufferD[blockIdx.x * element_pitch
							+ element_pitch / 2 + base + tid];
			base += blockDim.x;
		}
		line += gridDim.x;
	}
}

__global__ void kreorderInvDP(double *dataD, int element_pitch, int N,
		double *reorder_bufferD) {
	double *buffer = d_shmem_base_ptr;
	const int tid = threadIdx.x;
	const int halfblock = blockDim.x / 2;
	int line = blockIdx.x;
	int black_thread = (tid >= halfblock);
	while (line < (N - NCPU)) {
		int am_i_black = (line + tid) % 2;
		int base = 0;
		while (base + tid < N / 2) {
			reorder_bufferD[blockIdx.x * element_pitch + base + tid] =
					dataD[line * element_pitch + base + tid];
			reorder_bufferD[blockIdx.x * element_pitch + element_pitch / 2
					+ base + tid] = dataD[line * element_pitch
					+ element_pitch / 2 + base + tid];
			base += blockDim.x;
		}
		base = 0;
		while (base < N) {
			__syncthreads();
			int final_line_pos = element_pitch / 2 * black_thread + base / 2
					+ (tid - halfblock * black_thread);
			if (tid - halfblock * black_thread < (N - base) / 2)
				buffer[tid] = reorder_bufferD[blockIdx.x * element_pitch
						+ final_line_pos];
			__syncthreads();
			if (base + tid < N) {
				int newpos = tid >> 1;
				if (am_i_black)
					newpos += halfblock;
				dataD[element_pitch * line + base + tid] = buffer[newpos];
			}
			base += blockDim.x;
		}
		line += gridDim.x;
	}
}

#if RECALC_LEVEL==2
template<int granularity>
void lmsorRedDP(const dim3 &dimBl, const dim3 &dimGr, double *dSrcU, double *dDstU, double *dFFh, double *dGGh, double *sqrerrorD, int fWidth, unsigned int N, cudaStream_t stream0) {
	const int shm_size = sizeof(double) * dimBl.x*dimBl.y
#if KERNEL_TYPE==3
	+ sizeof(double) * (dimBl.x+2)*(dimBl.y*granularity+2)
#endif
	;
	klmsorDP<granularity, 1><<<dimGr, dimBl, shm_size, stream0>>>(N, fWidth, dSrcU, dDstU, dFFh, dGGh, sqrerrorD);
}

template<int granularity>
void lmsorBlackDP(const dim3 &dimBl, const dim3 &dimGr, double *dSrcU, double *dDstU, double *dFFh, double *dGGh, double *sqrerrorD, int fWidth, unsigned int N, cudaStream_t stream1) {
	const int shm_size = sizeof(double) * dimBl.x*dimBl.y
#if KERNEL_TYPE==3
	+ sizeof(double) * (dimBl.x+2)*(dimBl.y*granularity+2)
#endif
	;
	klmsorDP<granularity, 0><<<dimGr, dimBl, shm_size, stream1>>>(N, fWidth, dSrcU, dDstU, dFFh, dGGh, sqrerrorD);
}

#elif RECALC_LEVEL==1

template<int granularity>
void lmsorRedDP(const dim3 &dimBl, const dim3 &dimGr, double *dSrcU,
		double *dDstU, double *dFFh, double *dGGh, double *dW,
		double *sqrerrorD, int fWidth, unsigned int N, cudaStream_t stream0) {
	const int shm_size = sizeof(double) * dimBl.x * dimBl.y
#if KERNEL_TYPE==3
	+ sizeof(double) * (dimBl.x+2)*(dimBl.y*granularity+2)
#endif
	;
	klmsorDP<granularity, 1> <<<dimGr, dimBl, shm_size, stream0>>>(N, fWidth, dSrcU,
			dDstU, dFFh, dGGh, dW, sqrerrorD);
}

template<int granularity>
void lmsorBlackDP(const dim3 &dimBl, const dim3 &dimGr, double *dSrcU,
		double *dDstU, double *dFFh, double *dGGh, double *dW,
		double *sqrerrorD, int fWidth, unsigned int N, cudaStream_t stream1) {
	const int shm_size = sizeof(double) * dimBl.x * dimBl.y
#if KERNEL_TYPE==3
	+ sizeof(double) * (dimBl.x+2)*(dimBl.y*granularity+2)
#endif
	;
	klmsorDP<granularity, 0> <<<dimGr, dimBl, shm_size, stream1>>>(N, fWidth, dSrcU,
			dDstU, dFFh, dGGh, dW, sqrerrorD);
}

#else
template<int granularity>
void lmsorRedDP(const dim3 &dimBl, const dim3 &dimGr, double *dSrcU,
		double *dDstU, double *dW, double *dL, double *dR, double *dB,
		double *dT, double *sqrerrorD, int fWidth, unsigned int N,
		cudaStream_t stream0) {
	const int shm_size = sizeof(double) * dimBl.x * dimBl.y
#if KERNEL_TYPE==3
			+ sizeof(double) * (dimBl.x+2)*(dimBl.y*granularity+2)
#endif
			;
	klmsorDP<granularity, 1> <<<dimGr, dimBl, shm_size, stream0>>>(N, fWidth,
			dSrcU, dDstU, dL, dR, dB, dT, dW, sqrerrorD);
}

template<int granularity>
void lmsorBlackDP(const dim3 &dimBl, const dim3 &dimGr, double *dSrcU,
		double *dDstU, double *dW, double *dL, double *dR, double *dB,
		double *dT, double *sqrerrorD, int fWidth, unsigned int N,
		cudaStream_t stream1) {
	const int shm_size = sizeof(double) * dimBl.x * dimBl.y
#if KERNEL_TYPE==3
			+ sizeof(double) * (dimBl.x+2)*(dimBl.y*granularity+2)
#endif
			;
	klmsorDP<granularity, 0> <<<dimGr, dimBl, shm_size, stream1>>>(N, fWidth,
			dSrcU, dDstU, dL, dR, dB, dT, dW, sqrerrorD);
}
#endif

template<int granularity>
void setThreadSetCacheConfigDP(bool sharedMem) {
	enum cudaFuncCache fc;
	fc = sharedMem ? cudaFuncCachePreferShared : cudaFuncCachePreferL1;

	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(klmsorDP<granularity, 0>, fc));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(klmsorDP<granularity, 1>, fc));

	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(kreorderDP, fc));
	CUDA_SAFE_CALL(cudaFuncSetCacheConfig(kreorderInvDP, fc));
}

float exchangeRows(double *A_RedD, double *A_BlackD, size_t pitch) {
	int i, j;

	double buf[R_EXCH][NMAX / 2];

	timestamp ts_start_exch = getTimestamp();

		for (i = 0; i < R_EXCH; i++) {
			for (j = 0; j < NMAX; j++) {
				buf[i][j / 2] = ro_u_host[0][NCPU - R_EXCH + i][j / 2];
			}
		}
		CUDA_SAFE_CALL(
				cudaMemcpy2D(ro_u_host[0][NCPU - R_EXCH], NMAX / 2 * sizeof(double), A_RedD, pitch,
						NMAX / 2 * sizeof(double), R_EXCH,
						cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(
				cudaMemcpy2D(A_RedD, pitch, buf, NMAX / 2 * sizeof(double),
						NMAX / 2 * sizeof(double), R_EXCH,
						cudaMemcpyHostToDevice));

		for (i = 0; i < R_EXCH; i++) {
			for (j = 0; j < NMAX; j++) {
				buf[i][j / 2] = ro_u_host[1][NCPU - R_EXCH + i][j / 2];
			}
		}
		CUDA_SAFE_CALL(
				cudaMemcpy2D(ro_u_host[1][NCPU - R_EXCH], NMAX / 2 * sizeof(double), A_BlackD,
						pitch, NMAX / 2 * sizeof(double), R_EXCH,
						cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(
				cudaMemcpy2D(A_BlackD, pitch, buf,
						NMAX / 2 * sizeof(double), NMAX / 2 * sizeof(double),
						R_EXCH, cudaMemcpyHostToDevice));

	return getElapsedtime(ts_start_exch) / 1000.0;
}

// Matrix reordering and copy (Host to device)
size_t Copy2DPMatrixDataToDeviceDP(int N, double **AdestRedD,
		double **AdestBlackD, double *Asrc, float *Atime) {
	size_t halfpitch, pitch, pitchRB;

	double *dummy;
	CUDA_SAFE_CALL(
			cudaMallocPitch(&dummy, &halfpitch, sizeof(double) * N / 2, 1));
	CUDA_SAFE_CALL(cudaFree(dummy));

	double *dataD;
	CUDA_SAFE_CALL(cudaMallocPitch(&dataD, &pitch, halfpitch * 2, N - NCPU));
	if (pitch / 2 != halfpitch) {
		printf("ERROR: pitch/2 != halfpitch (%d, %d)\n", pitch / 2, halfpitch);
		exit(1);
	}
	CUDA_SAFE_CALL(
			cudaMemcpy2D(dataD, pitch, Asrc, N * sizeof(double),
					N * sizeof(double), N - NCPU, cudaMemcpyHostToDevice));

	const int REORDER_BLOCKS = 60, BLOCK_SIZE = 256;
	double *reorder_bufferD;
	CUDA_SAFE_CALL(
			cudaMallocPitch(&reorder_bufferD, &pitchRB, pitch, REORDER_BLOCKS));
	if (pitchRB != pitch) {
		printf("ERROR: pitch != pitchRB (%d, %d)\n", pitchRB, pitch);
		exit(1);
	}

	cudaEvent_t t_start, t_stop;
	CUDA_SAFE_CALL(cudaEventCreate(&t_start));
	CUDA_SAFE_CALL(cudaEventCreate(&t_stop));
	CUDA_SAFE_CALL(cudaEventRecord(t_start));

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(REORDER_BLOCKS);
	kreorderDP<<<dimGrid, dimBlock, sizeof(double) * BLOCK_SIZE>>>(dataD,
			pitch / sizeof(double), N, reorder_bufferD);

	float secs_reordering;
	CUDA_SAFE_CALL(cudaEventRecord(t_stop));
	CUDA_SAFE_CALL(cudaEventSynchronize(t_stop));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&secs_reordering, t_start, t_stop));
	CUDA_SAFE_CALL(cudaEventDestroy(t_start));
	CUDA_SAFE_CALL(cudaEventDestroy(t_stop));
	secs_reordering = secs_reordering / 1000.0f;

	CUDA_SAFE_CALL(cudaFree(reorder_bufferD));

	if (Atime)
		*Atime = secs_reordering;

	*AdestRedD = dataD;
	*AdestBlackD = dataD + halfpitch / sizeof(double);

	return pitch;
}

// Matrix copy and reordering (Device to host)
void Copy2DPMatrixDataFromDeviceAndFreeDP(int N, double *Adest,
		double *AsrcRedD, double *AsrcBlackD, size_t Apitch, float *Atime) {
	const int REORDER_BLOCKS = 60, BLOCK_SIZE = 256;
	double *dataD = AsrcRedD;

	double *reorder_bufferD;
	size_t pitchrb;
	CUDA_SAFE_CALL(
			cudaMallocPitch(&reorder_bufferD, &pitchrb, Apitch,
					REORDER_BLOCKS));

	cudaEvent_t t_start, t_stop;
	CUDA_SAFE_CALL(cudaEventCreate(&t_start));
	CUDA_SAFE_CALL(cudaEventCreate(&t_stop));
	CUDA_SAFE_CALL(cudaEventRecord(t_start));

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(REORDER_BLOCKS);
	kreorderInvDP<<<dimGrid, dimBlock, sizeof(double) * BLOCK_SIZE>>>(dataD,
			Apitch / sizeof(double), N, reorder_bufferD);

	float secs_reordering;
	CUDA_SAFE_CALL(cudaEventRecord(t_stop));
	CUDA_SAFE_CALL(cudaEventSynchronize(t_stop));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&secs_reordering, t_start, t_stop));
	CUDA_SAFE_CALL(cudaEventDestroy(t_start));
	CUDA_SAFE_CALL(cudaEventDestroy(t_stop));
	secs_reordering = secs_reordering / 1000.0f;

	CUDA_SAFE_CALL(cudaFree(reorder_bufferD));

	if (Atime)
		*Atime = secs_reordering;

	CUDA_SAFE_CALL(
			cudaMemcpy2D(Adest, N * sizeof(double), dataD, Apitch,
					N * sizeof(double), N - NCPU, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(dataD));
}

void FreeReorderedPointers(double *AsrcRedD, double*) {
	CUDA_SAFE_CALL(cudaFree(AsrcRedD));
}

size_t getCUDAFreeMem(void) {
	size_t freeCUDAMem;
	cudaMemGetInfo(&freeCUDAMem, NULL);
	return freeCUDAMem;
}

void printCUDAMem(void) {
	size_t freeCUDAMem, totalCUDAMem;
	cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
	printf("Total GPU memory %uMB, free %uMB\n", totalCUDAMem / (1024 * 1024),
			freeCUDAMem / (1024 * 1024));
}

inline double calcPoint(const int pitch, const int line_indicator,
		double *pu_point, const double *pu_neighb, const double *pffh,
		const double *pggh, const double *pw) {
	double r = (1. - *pffh) / 4.;
	double l = (1. + *pffh) / 4.;
	double t = (1. - *pggh) / 4.;
	double b = (1. + *pggh) / 4.;
	*pu_point = (1. - *pw) * *pu_point
			+ *pw
					* (l * *(pu_neighb - pitch) + r * *(pu_neighb + pitch)
							+ b * *(pu_neighb - line_indicator)
							+ t * *(pu_neighb + 1 - line_indicator));
	return *pu_point;
}

inline double calcPointUpdErr(const int pitch, const int line_indicator,
		double *pu_point, const double *pu_neighb, const double *pffh,
		const double *pggh, const double *pw, double *sqrerror) {
	double old_val = *pu_point;
	double new_val = calcPoint(pitch, line_indicator, pu_point, pu_neighb, pffh,
			pggh, pw);
	double sqr_diff = pow2(old_val - new_val);
	*sqrerror += sqr_diff;
	return new_val;
}

template<int phase>
void calcSegment(const int bound_s_x, const int bound_e_x, const int bound_s_y,
		const int bound_e_y, double &sqrerror) {
#ifdef _INTRINSIC_SSE2_
	const int pitch = NMAX/2;
	const int SIMD_WIDTH = sizeof(__m128d) / sizeof(double);
	double * __restrict pu_des = (double*)ro_u_host[phase]; //ro_u._get_data(phase);
	const double * __restrict pu_oth = (double*)ro_u_host[1-phase];//ro_u._get_data(1-phase);
	const double * __restrict pw = (double*)ro_w1_host[phase];//ro_w._get_data(phase);
	const double * __restrict pffh = (double*)ro_ffh_host[phase];//ro_ffh._get_data(phase);
	const double * __restrict pggh = (double*)ro_ggh_host[phase];//ro_ggh._get_data(phase);
	const __m128d m_one = _mm_set1_pd(1.0), m_quarter = _mm_set1_pd(0.25);
	for(int i=bound_s_y; i<bound_e_y; i++) {
		int line_indicator = (1-phase+i)%2;

		int color_offset = (i+bound_s_x+phase) % 2;
		int line_idx_from = i*pitch + (bound_s_x + color_offset)/2;
		int line_idx_to = i*pitch + bound_e_x/2 + ((bound_s_x + color_offset)%2<(bound_e_x%2));

//printf("%d. start %p end %p\n", phase, u._get_data(phase)+line_idx_from, u._get_data(phase)+line_idx_to);
		double *tmp = (double*)ro_u_host[phase];
		if( ((size_t)(tmp+line_idx_from)) % sizeof(__m128d) ) {
			calcPointUpdErr(pitch, line_indicator, &pu_des[line_idx_from], &pu_oth[line_idx_from], &pffh[line_idx_from], &pggh[line_idx_from], &pw[line_idx_from], &sqrerror);
			line_idx_from++;
		}
		int do_last = ((size_t)(tmp+line_idx_to)) % sizeof(__m128d);
		if( do_last )
		line_idx_to--;

		for(int idx=line_idx_from; idx<line_idx_to; idx+=SIMD_WIDTH) {
			__m128d m_ffh = _mm_load_pd(&pffh[idx]);
			__m128d m_ggh = _mm_load_pd(&pggh[idx]);
			__m128d m_opr, m_op2, m_u_n, m_sum;
			// left
			m_u_n = _mm_load_pd(&pu_oth[idx-pitch]);
			m_opr = _mm_add_pd(m_one, m_ffh);
			m_opr = _mm_mul_pd(m_quarter, m_opr);
			m_sum = _mm_mul_pd(m_u_n, m_opr);
			// right
			m_u_n = _mm_load_pd(&pu_oth[idx+pitch]);
			m_opr = _mm_sub_pd(m_one, m_ffh);
			m_opr = _mm_mul_pd(m_quarter, m_opr);
			m_u_n = _mm_mul_pd(m_u_n, m_opr);
			m_sum = _mm_add_pd(m_u_n, m_sum);
			// bottom
			m_u_n = _mm_loadu_pd(&pu_oth[idx-line_indicator]);
			m_opr = _mm_add_pd(m_one, m_ggh);
			m_opr = _mm_mul_pd(m_quarter, m_opr);
			m_u_n = _mm_mul_pd(m_u_n, m_opr);
			m_sum = _mm_add_pd(m_u_n, m_sum);
			// top
			m_u_n = _mm_loadu_pd(&pu_oth[idx+1-line_indicator]);
			m_opr = _mm_sub_pd(m_one, m_ggh);
			m_opr = _mm_mul_pd(m_quarter, m_opr);
			m_u_n = _mm_mul_pd(m_u_n, m_opr);
			m_sum = _mm_add_pd(m_u_n, m_sum);
			// multiply with w_ij
			m_opr = _mm_load_pd(&pw[idx]);
			m_sum = _mm_mul_pd(m_opr, m_sum);
			// first term
			m_op2 = _mm_sub_pd(m_one, m_opr);
			m_u_n = _mm_load_pd(&pu_des[idx]);
			m_opr = _mm_mul_pd(m_op2, m_u_n);
			m_sum = _mm_add_pd(m_opr, m_sum);

			_mm_store_pd(&pu_des[idx], m_sum);
//			_mm_stream_pd(&pu_des[idx], m_sum);
//			_mm_store_sd(&v, m_sum);
			/*			_mm_storel_pd(&v, m_sum);
			 double max1 = fabs(v);
			 if( tmax<max1 ) tmax = max1;
			 _mm_storeh_pd(&v, m_sum);
			 max1 = fabs(v);
			 if( tmax<max1 ) tmax = max1;*/

			m_opr = _mm_sub_pd(m_u_n, m_sum);
			m_sum = _mm_mul_pd(m_opr, m_opr);

			__m128d m_perm = _mm_shuffle_pd (m_sum, m_sum, 1);
			m_perm = _mm_add_pd(m_sum, m_perm);
			double sqr_diff;
			_mm_storel_pd(&sqr_diff, m_perm);
//			double sqr_diff = fabs(v);
//			if( sqrerror<max1 ) sqrerror = max1;
			sqrerror += sqr_diff;
		}

		if( do_last ) {
			calcPointUpdErr(pitch, line_indicator, &pu_des[line_idx_to], &pu_oth[line_idx_to], &pffh[line_idx_to], &pggh[line_idx_to], &pw[line_idx_to], &sqrerror);
		}
	}
//	_mm_sfence();

#else
	for (int i = bound_s_y; i < bound_e_y; i++) {
		for (int j = bound_s_x; j < bound_e_x; j++) {
			if ((i + j) % 2 == phase) {
				double old_val;
				if ((i + j) % 2 == 0) {
					old_val = ro_u_host[0][i][j / 2];
				} else {
					old_val = ro_u_host[1][i][j / 2];
				}
#if PRECALC != 0
				double r;
				if ((i + j) % 2 == 0) {
					r = 1. - ro_ffh_host[0][i][j / 2];
				} else {
					r = 1. - ro_ffh_host[1][i][j / 2];
				}
				double l;
				if ((i + j) % 2 == 0) {
					l = 1. + ro_ffh_host[0][i][j / 2];
				} else {
					l = 1. + ro_ffh_host[1][i][j / 2];
				}
				double t;
				if ((i + j) % 2 == 0) {
					t = 1. - ro_ggh_host[0][i][j / 2];
				} else {
					t = 1. - ro_ggh_host[1][i][j / 2];
				}
				double b;
				if ((i + j) % 2 == 0) {
					b = 1. + ro_ggh_host[0][i][j / 2];
				} else {
					b = 1. + ro_ggh_host[1][i][j / 2];
				}

				const double D = 1. / 4.;
				r = r * D;
				l = l * D;
				t = t * D;
				b = b * D;

				double w_tmp;
				if ((i + j) % 2 == 0) {
					w_tmp = ro_w1_host[0][i][j / 2];
				} else {
					w_tmp = ro_w1_host[1][i][j / 2];
				}
				double u_tmp;
				if ((i + j) % 2 == 0) {
					u_tmp = ro_u_host[0][i][j / 2];
				} else {
					u_tmp = ro_u_host[1][i][j / 2];
				}
				double u_tmp_left;
				if (((i-1) + j) % 2 == 0) {
					u_tmp_left = ro_u_host[0][i - 1][j / 2];
				} else {
					u_tmp_left = ro_u_host[1][i - 1][j / 2];
				}
				double u_tmp_right;
				if (((i+1) + j) % 2 == 0) {
					u_tmp_right = ro_u_host[0][i + 1][j / 2];
				} else {
					u_tmp_right = ro_u_host[1][i + 1][j / 2];
				}
				double u_tmp_bottom;
				if ((i + (j-1)) % 2 == 0) {
					u_tmp_bottom = ro_u_host[0][i][(j - 1) / 2];
				} else {
					u_tmp_bottom = ro_u_host[1][i][(j - 1) / 2];
				}
				double u_tmp_top;
				if ((i + (j+1)) % 2 == 0) {
					u_tmp_top = ro_u_host[0][i][(j + 1) / 2];
				} else {
					u_tmp_top = ro_u_host[1][i][(j + 1) / 2];
				}
				if ((i + j) % 2 == 0) {
					ro_u_host[0][i][j / 2] = (1. - w_tmp) * u_tmp
					+ w_tmp
					* (l * u_tmp_left + r * u_tmp_right
							+ b * u_tmp_bottom + t * u_tmp_top);
				} else {
					ro_u_host[1][i][j / 2] = (1. - w_tmp) * u_tmp
					+ w_tmp
					* (l * u_tmp_left + r * u_tmp_right
							+ b * u_tmp_bottom + t * u_tmp_top);
				}
#else
				double w_tmp;
				if ((i + j) % 2 == 0) {
					w_tmp = ro_w1_host[0][i][j / 2];
				} else {
					w_tmp = ro_w1_host[1][i][j / 2];
				}
				double u_tmp;
				if ((i + j) % 2 == 0) {
					u_tmp = ro_u_host[0][i][j / 2];
				} else {
					u_tmp = ro_u_host[1][i][j / 2];
				}
				double l_tmp;
				if ((i + j) % 2 == 0) {
					l_tmp = ro_l_host[0][i][j / 2];
				} else {
					l_tmp = ro_l_host[1][i][j / 2];
				}
				double r_tmp;
				if ((i + j) % 2 == 0) {
					r_tmp = ro_r_host[0][i][j / 2];
				} else {
					r_tmp = ro_r_host[1][i][j / 2];
				}
				double b_tmp;
				if ((i + j) % 2 == 0) {
					b_tmp = ro_b_host[0][i][j / 2];
				} else {
					b_tmp = ro_b_host[1][i][j / 2];
				}
				double t_tmp;
				if ((i + j) % 2 == 0) {
					t_tmp = ro_t_host[0][i][j / 2];
				} else {
					t_tmp = ro_t_host[0][i][j / 2];
				}
				double u_tmp_left;
				if (((i - 1) + j) % 2 == 0) {
					u_tmp_left = ro_u_host[0][i - 1][j / 2];
				} else {
					u_tmp_left = ro_u_host[1][i - 1][j / 2];
				}
				double u_tmp_right;
				if (((i + 1) + j) % 2 == 0) {
					u_tmp_right = ro_u_host[0][i + 1][j / 2];
				} else {
					u_tmp_right = ro_u_host[1][i + 1][j / 2];
				}
				double u_tmp_bottom;
				if ((i + (j - 1)) % 2 == 0) {
					u_tmp_bottom = ro_u_host[0][i][(j - 1) / 2];
				} else {
					u_tmp_bottom = ro_u_host[1][i][(j - 1) / 2];
				}
				double u_tmp_top;
				if ((i + (j + 1)) % 2 == 0) {
					u_tmp_top = ro_u_host[0][i][(j + 1) / 2];
				} else {
					u_tmp_top = ro_u_host[1][i][(j + 1) / 2];
				}
				if ((i + j) % 2 == 0) {
					ro_u_host[0][i][j / 2] = (1. - w_tmp) * u_tmp
							+ w_tmp
									* (l_tmp * u_tmp_left + r_tmp * u_tmp_right
											+ b_tmp * u_tmp_bottom
											+ t_tmp * u_tmp_top);
				} else {
					ro_u_host[1][i][j / 2] = (1. - w_tmp) * u_tmp
							+ w_tmp
									* (l_tmp * u_tmp_left + r_tmp * u_tmp_right
											+ b_tmp * u_tmp_bottom
											+ t_tmp * u_tmp_top);
				}
#endif
				if ((i + j) % 2 == 0) {
					sqrerror += pow2(old_val - ro_u_host[0][i][j / 2]);
				} else {
					sqrerror += pow2(old_val - ro_u_host[1][i][j / 2]);
				}
			}
		}
	}
#endif
}

void fwReorder() {
	int i, j;

// reorder U
	for (i = 0; i < NCPU; i++) {
		for (j = 0; j < NMAX; j++) {
			if ((i + j) % 2 == 0) {
				ro_u_host[0][i][j / 2] = u_host[i][j];
			} else {
				ro_u_host[1][i][j / 2] = u_host[i][j];
			}
			// reorder W
			if ((i + j) % 2 == 0) {
				ro_w1_host[0][i][j / 2] = w1_host[i][j];
			} else {
				ro_w1_host[1][i][j / 2] = w1_host[i][j];
			}
#if PRECALC != 0
			// reorder FFH
			if ((i + j) % 2 == 0) {
				ro_ffh_host[0][i][j / 2] = ffh_host[i][j];
			} else {
				ro_ffh_host[1][i][j / 2] = ffh_host[i][j];
			}
			// reorder GGH
			if ((i + j) % 2 == 0) {
				ro_ggh_host[0][i][j / 2] = ggh_host[i][j];
			} else {
				ro_ggh_host[1][i][j / 2] = ggh_host[i][j];
			}
#else
			// reorder L
			if ((i + j) % 2 == 0) {
				ro_l_host[0][i][j / 2] = l_host[i][j];
			} else {
				ro_l_host[1][i][j / 2] = l_host[i][j];
			}
			// reorder R
			if ((i + j) % 2 == 0) {
				ro_r_host[0][i][j / 2] = r_host[i][j];
			} else {
				ro_r_host[1][i][j / 2] = r_host[i][j];
			}
			// reorder T
			if ((i + j) % 2 == 0) {
				ro_t_host[0][i][j / 2] = t_host[i][j];
			} else {
				ro_t_host[1][i][j / 2] = t_host[i][j];
			}
			// reorder B
			if ((i + j) % 2 == 0) {
				ro_b_host[0][i][j / 2] = b_host[i][j];
			} else {
				ro_b_host[1][i][j / 2] = b_host[i][j];
			}
#endif
		}
	}
}

void invReorder() {
	int i, j;

	// reorder U
	for (i = 0; i < NCPU; i++) {
		for (j = 0; j < NMAX; j++) {
			if ((i + j) % 2 == 0) {
				u_host[i][j] = ro_u_host[0][i][j / 2];
			} else {
				u_host[i][j] = ro_u_host[1][i][j / 2];
			}
		}
	}
}

long combinedLMSOR(int nmax, int ncpu, int maxiter, double l1, double l2,
		double* w, double* u, double* l, double* r, double* b, double* t,
		long* pk, int epilogi, double* ffh, double* ggh, double re,
		double* calctime, float* GPUreorderFwTime, float* GPUreorderInvTime) {
	if ((nmax % 2 != 0) || ((nmax - ncpu) % 2 != 0)) {
		printf("ERROR: NMAX and NGPU should be a multiple of 2\n");
		exit(1);
	}

	CUDA_SAFE_CALL(cudaSetDevice(0));

	switch (getFPDevSize()) {
	case 1:
		printf("ERROR: No double precision is supported\n");
		exit(1);
	case 0:
		printf("ERROR: CUDA is not supported\n");
		exit(1);
	}

	char *pEnvCacheConf, *pEnvBlockX, *pEnvBlockY;
	pEnvCacheConf = getenv("LMSOR_CACHECONF");
	pEnvBlockX = getenv("LMSOR_BLOCKX");
	pEnvBlockY = getenv("LMSOR_BLOCKY");

	printf("\nKernel configuration:\n");
	printf(" Redundant calculations: ");
#if RECALC_LEVEL==0
	printf("0-None\n");
#ifndef GRANULARITY
#define GRANULARITY 10
#endif
#elif RECALC_LEVEL==1
	printf("1-Factors r(i,j), l(i,j), t(i,j), b(i,j)\n");
#ifndef GRANULARITY
#define GRANULARITY 16
#endif
#else
	printf("2-All factors\n");
#ifndef GRANULARITY
#define GRANULARITY 10
#endif
#endif

	printf(" Kernel type           : ");
#if KERNEL_TYPE==1
	printf("1-Global memory only\n");
#elif KERNEL_TYPE==2
	printf("2-Texture memory (for neighbor elements)\n");
#elif KERNEL_TYPE==3
	printf("3-Shared memory (for neighbor elements)\n");
#else
#error "Unknown kernel!"
#endif

	const int granularity = (GRANULARITY);
	const int NGPU_inner = (nmax - ncpu) - 2;
	const bool confPrefSharedMem =
			(pEnvCacheConf != NULL) ?
					(atoi(pEnvCacheConf) != 0) : (PREF_SHARED_MEM);
	if (NGPU_inner % granularity != 0) {
		printf(
				"ERROR: GPU lines (%d) should be a multiple of granularity(%d)\n",
				NGPU_inner, granularity);
		exit(1);
	}
	dim3 dimBlock((pEnvBlockX != NULL) ? atoi(pEnvBlockX) : 64,
			(pEnvBlockY != NULL) ? atoi(pEnvBlockY) : 2);
	dim3 dimGrid((nmax / 2 + dimBlock.x - 1) / dimBlock.x,
			(nmax - ncpu + (dimBlock.y * granularity) - 1)
					/ (dimBlock.y * granularity));
	dim3 dimBlockReduction(256);
	dim3 dimGridReduction(32);

	printf(" Thread granularity    : %d\n", granularity);
	printf(" Grid configuration    : %dx%d\n Block configuration   : %dx%d\n",
			dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

	printf(" L1 cache configuration: %s\n",
			confPrefSharedMem ?
					"48KB shared mem/16KB L1 cache" :
					"16KB shared mem/48KB L1 cache");
	printf("\n");
	setThreadSetCacheConfigDP<granularity>(confPrefSharedMem);
	//	printCUDAMem();
	// copy data to device
	double *d_u_red, *d_u_black, *sqrerrorD;
	float secs_reorderingTmp, secs_reorderingFw = 0.0f, secs_reorderingInv =
			0.0f;
	double zero = 0.0;
	printf("Allocating device memory matrices...\n");
	size_t pitch = Copy2DPMatrixDataToDeviceDP(nmax, &d_u_red, &d_u_black, u,
			&secs_reorderingTmp);
	secs_reorderingFw += secs_reorderingTmp;
#if RECALC_LEVEL==0 || RECALC_LEVEL==1
	double *d_w_red, *d_w_black;
	Copy2DPMatrixDataToDeviceDP(nmax, &d_w_red, &d_w_black, w,
			&secs_reorderingTmp);
	secs_reorderingFw += secs_reorderingTmp;
#endif
#if RECALC_LEVEL==0
	double *d_l_red, *d_l_black, *d_r_red, *d_r_black, *d_b_red, *d_b_black,
			*d_t_red, *d_t_black;
	Copy2DPMatrixDataToDeviceDP(nmax, &d_l_red, &d_l_black, l,
			&secs_reorderingTmp);
	secs_reorderingFw += secs_reorderingTmp;
	Copy2DPMatrixDataToDeviceDP(nmax, &d_r_red, &d_r_black, r,
			&secs_reorderingTmp);
	secs_reorderingFw += secs_reorderingTmp;
	Copy2DPMatrixDataToDeviceDP(nmax, &d_b_red, &d_b_black, b,
			&secs_reorderingTmp);
	secs_reorderingFw += secs_reorderingTmp;
	Copy2DPMatrixDataToDeviceDP(nmax, &d_t_red, &d_t_black, t,
			&secs_reorderingTmp);
	secs_reorderingFw += secs_reorderingTmp;
#endif
#if RECALC_LEVEL==1 || RECALC_LEVEL==2
	double *d_ff_h_red, *d_ff_h_black, *d_gg_h_red, *d_gg_h_black;
	Copy2DPMatrixDataToDeviceDP(nmax, &d_ff_h_red, &d_ff_h_black, ffh,
			&secs_reorderingTmp);
	secs_reorderingFw += secs_reorderingTmp;
	Copy2DPMatrixDataToDeviceDP(nmax, &d_gg_h_red, &d_gg_h_black, ggh,
			&secs_reorderingTmp);
	secs_reorderingFw += secs_reorderingTmp;
#endif
	CUDA_SAFE_CALL(
			cudaMalloc(&sqrerrorD,
					sizeof(double) * (dimGrid.x * dimGrid.y + 1)));
	//printf(" Free memory: %uMB\n", getCUDAFreeMem()/(1024*1024));
	printCUDAMem();

#if KERNEL_TYPE==2
	CUDA_SAFE_CALL(bindTextureDP(texURed, d_u_red, nmax / 2, nmax - ncpu, pitch));
	CUDA_SAFE_CALL(bindTextureDP(texUBlack, d_u_black, nmax / 2, nmax - ncpu, pitch));
#endif

	const int total_error_vals = dimGrid.x * dimGrid.y;
	long metr = 0;
	int i, k;
	int nn = nmax - 1;
	int mm = ncpu - 1;
	int n = nn - 1;
	int m = mm - 1;
	double sqrerrorCPU, sqrerrorGPU;
	float exchTime = 0.0;
	timestamp ts_start_total;

	cudaStream_t stream0, stream1;
	CUDA_SAFE_CALL(cudaStreamCreate(&stream0));
	CUDA_SAFE_CALL(cudaStreamCreate(&stream1));
	CUDA_SAFE_CALL(cudaHostRegister(&zero, sizeof(double), 0));
	CUDA_SAFE_CALL(cudaHostRegister(&sqrerrorGPU, sizeof(double), 0));

	ts_start_total = getTimestamp();

#pragma omp parallel shared(sqrerrorCPU,metr,k) private(i)
	{
#pragma omp master
#ifdef _OPENMP
		printf("Using block partitioning (total threads %d)\n", omp_get_num_threads());
#endif

		int num_threads = omp_get_num_threads();
		int sy = num_threads, sx = 1, sx_limit = (int) sqrt((double) sy);
		for (i = sx_limit; i > 0; i--)
			if (num_threads % i == 0) {
				sx = i;
				sy = num_threads / i;
				break;
			}
		int bs_x = n / sx, bs_y = m / sy, lastoffset_x = n % sx, lastoffset_y =
				m % sy;
		int start_pos_x = 1 + bs_x * (omp_get_thread_num() % sx), start_pos_y =
				1 + bs_y * (omp_get_thread_num() / sx), end_pos_x = start_pos_x
				+ bs_x
				+ (omp_get_thread_num() % sx == sx - 1 ? lastoffset_x : 0),
				end_pos_y = start_pos_y + bs_y
						+ (omp_get_thread_num() % sy == sy - 1 ?
								lastoffset_y : 0);

		do {
#pragma omp master
			{
				CUDA_SAFE_CALL(
						cudaMemcpyAsync(&sqrerrorD[total_error_vals], &zero,
								sizeof(double), cudaMemcpyHostToDevice,
								stream0));
			}
			double tsqrerror = 0.0;
#pragma omp master
			{
				metr = metr + 1;
				sqrerrorCPU = 0.0;
#if RECALC_LEVEL==2
				lmsorRedDP<granularity>(dimBlock, dimGrid, d_u_black, d_u_red, d_ff_h_red, d_gg_h_red, sqrerrorD, pitch/sizeof(double), nmax, stream0);
				lmsorBlackDP<granularity>(dimBlock, dimGrid, d_u_red, d_u_black, d_ff_h_red, d_gg_h_red, sqrerrorD, pitch/sizeof(double), nmax, stream1)
#elif RECALC_LEVEL==1
				lmsorRedDP<granularity>(dimBlock, dimGrid, d_u_black, d_u_red, d_ff_h_red, d_gg_h_red, d_w_red, sqrerrorD, pitch / sizeof(double), nmax, stream0);
				lmsorBlackDP<granularity>(dimBlock, dimGrid, d_u_red, d_u_black, d_ff_h_black, d_gg_h_black, d_w_black, sqrerrorD, pitch / sizeof(double), nmax, stream1);
#else
				lmsorRedDP<granularity>(dimBlock, dimGrid, d_u_black, d_u_red,
						d_w_red, d_l_red, d_r_red, d_b_red, d_t_red, sqrerrorD,
						pitch / sizeof(double), nmax, stream0);
				lmsorBlackDP<granularity>(dimBlock, dimGrid, d_u_red, d_u_black,
						d_w_black, d_l_black, d_r_black, d_b_black, d_t_black,
						sqrerrorD, pitch / sizeof(double), nmax, stream1);
#endif
			}
			calcSegment<0>(start_pos_x, end_pos_x, start_pos_y, end_pos_y,
					tsqrerror);
			calcSegment<1>(start_pos_x, end_pos_x, start_pos_y, end_pos_y,
					tsqrerror);
#pragma omp master
			{
				kreduceDP<<<dimGridReduction, dimBlockReduction,
						dimBlockReduction.x * sizeof(double)>>>(sqrerrorD,
						&sqrerrorD[total_error_vals], total_error_vals);
			}
#pragma omp critical
			{
				sqrerrorCPU += tsqrerror;
			}
#pragma omp barrier
#pragma omp master
			{
				CUDA_SAFE_CALL(
						cudaMemcpy(&sqrerrorGPU, &sqrerrorD[total_error_vals],
								sizeof(double), cudaMemcpyDeviceToHost));
				if ((metr % R_EXCH) == 0) {
					// exchange border rows between host and device
					exchTime += exchangeRows(d_u_red, d_u_black, pitch);
				}
				if ((sqrerrorCPU + sqrerrorGPU) <= l1)
					k = 0;
				else {
					if (isnan(sqrerrorCPU + sqrerrorGPU)
							|| (sqrerrorCPU + sqrerrorGPU) >= l2)
						k = 2;
					else
						k = 1;
				}
				//printf("\n%f %f\n", sqrerrorCPU, sqrerrorGPU);
			}
#pragma omp barrier
		} while ((k == 1) && (metr <= maxiter));
	}

	float lcalc_time = getElapsedtime(ts_start_total) / 1000.0;

	CUDA_SAFE_CALL(cudaHostUnregister(&zero));
	CUDA_SAFE_CALL(cudaHostUnregister(&sqrerrorGPU));
	CUDA_SAFE_CALL(cudaStreamDestroy(stream0));
	CUDA_SAFE_CALL(cudaStreamDestroy(stream1));

#if KERNEL_TYPE==2
	CUDA_SAFE_CALL(unbindTextureDP(texURed));
	CUDA_SAFE_CALL(unbindTextureDP(texUBlack));
#endif

#if RECALC_LEVEL==0 && PRECALC==0
	const double ACCESSES_PER_ELEMENT = 8.;
#elif RECALC_LEVEL==1 && PRECALC==1
	const double ACCESSES_PER_ELEMENT = 6.;
#elif RECALC_LEVEL==2
	const double ACCESSES_PER_ELEMENT = 5.;	//3.;
#endif
	double bandwidth = (ACCESSES_PER_ELEMENT * nmax * nmax * sizeof(double)
			* metr / lcalc_time) / (1024. * 1024. * 1024.);
	printf(
			"\n %f seconds total calculation time (%f msecs/iteration, %.2fGB/sec)\n",
			lcalc_time, 1000.0f * lcalc_time / metr, bandwidth);

	if (calctime)
		*calctime = lcalc_time;

	// copy results back and release device memory
	Copy2DPMatrixDataFromDeviceAndFreeDP(nmax, u, d_u_red, d_u_black, pitch,
			&secs_reorderingInv);
	printf(
			"\n %f GPU reordering overhead (%f + %f initialization/finalization)\n",
			secs_reorderingFw + secs_reorderingInv, secs_reorderingFw,
			secs_reorderingInv);
	printf("\nTotal time taken for row exchanges: %f seconds\n", exchTime);
	*GPUreorderFwTime = secs_reorderingFw;
	*GPUreorderInvTime = secs_reorderingInv;
	printf("\nCSV_KERNEL;%d;%d;%d;%d;%d;\"(%dx%d)\";%ld;%f;%f\n\n", nmax,
			RECALC_LEVEL, KERNEL_TYPE, granularity, confPrefSharedMem,
			dimBlock.x, dimBlock.y, metr, lcalc_time, bandwidth);

#if RECALC_LEVEL==0 || RECALC_LEVEL==1
	FreeReorderedPointers(d_w_red, d_w_black);
#endif
#if RECALC_LEVEL==0
	FreeReorderedPointers(d_l_red, d_l_black);
	FreeReorderedPointers(d_r_red, d_r_black);
	FreeReorderedPointers(d_b_red, d_b_black);
	FreeReorderedPointers(d_t_red, d_t_black);
#endif
#if RECALC_LEVEL==1 || RECALC_LEVEL==2
	FreeReorderedPointers(d_ff_h_red, d_ff_h_black);
	FreeReorderedPointers(d_gg_h_red, d_gg_h_black);
#endif
	CUDA_SAFE_CALL(cudaFree(sqrerrorD));

	if (pk)
		*pk = k;

	CUDA_SAFE_CALL(cudaDeviceReset());

	return metr;
}

void allocMem() {
// initial arrays
	w1 = (lattice*) malloc(sizeof(lattice) * NMAX);
	w2 = (lattice*) malloc(sizeof(lattice) * NMAX);
	u = (lattice*) malloc(sizeof(lattice) * NMAX);
#if (PRECALC != 0) && (RECALC_LEVEL != 0)
	ffh = (lattice*) malloc(sizeof(lattice) * NMAX);
	ggh = (lattice*) malloc(sizeof(lattice) * NMAX);
#else
	l = (lattice*) malloc(sizeof(lattice) * NMAX);
	r = (lattice*) malloc(sizeof(lattice) * NMAX);
	b = (lattice*) malloc(sizeof(lattice) * NMAX);
	t = (lattice*) malloc(sizeof(lattice) * NMAX);
#endif

// rows on device
	w1_dev = (lattice*) malloc(sizeof(lattice) * (NMAX - NCPU));
	u_dev = (lattice*) malloc(sizeof(lattice) * (NMAX - NCPU));
#if RECALC_LEVEL == 0
	l_dev = (lattice*) malloc(sizeof(lattice) * (NMAX - NCPU));
	r_dev = (lattice*) malloc(sizeof(lattice) * (NMAX - NCPU));
	b_dev = (lattice*) malloc(sizeof(lattice) * (NMAX - NCPU));
	t_dev = (lattice*) malloc(sizeof(lattice) * (NMAX - NCPU));
#else
	ffh_dev = (lattice*) malloc(sizeof(lattice) * (NMAX - NCPU));
	ggh_dev = (lattice*) malloc(sizeof(lattice) * (NMAX - NCPU));
#endif

// rows on host
	w1_host = (lattice*) malloc(sizeof(lattice) * NCPU);
	u_host = (lattice*) malloc(sizeof(lattice) * NCPU);
#if PRECALC != 0
	ffh_host = (lattice*) malloc(sizeof(lattice) * NCPU);
	ggh_host = (lattice*) malloc(sizeof(lattice) * NCPU);
#else
	l_host = (lattice*) malloc(sizeof(lattice) * NCPU);
	r_host = (lattice*) malloc(sizeof(lattice) * NCPU);
	b_host = (lattice*) malloc(sizeof(lattice) * NCPU);
	t_host = (lattice*) malloc(sizeof(lattice) * NCPU);
#endif
}

void splitMatrices() {
	int i, j;
	for (i = 0; i < NCPU; i++) {
		for (j = 0; j < NMAX; j++) {
			w1_host[i][j] = w1[i][j];
			u_host[i][j] = u[i][j];
		}
	}
#if PRECALC != 0
	for (i = 0; i < NCPU; i++) {
		for (j = 0; j < NMAX; j++) {
			ffh_host[i][j] = ffh[i][j];
			ggh_host[i][j] = ggh[i][j];
		}
	}
#else
	for (i = 0; i < NCPU; i++) {
		for (j = 0; j < NMAX; j++) {
			l_host[i][j] = l[i][j];
			r_host[i][j] = r[i][j];
			b_host[i][j] = b[i][j];
			t_host[i][j] = t[i][j];
		}
	}
#endif

	for (i = 0; i < (NMAX - NCPU); i++) {
		for (j = 0; j < NMAX; j++) {
			w1_dev[i][j] = w1[i + NCPU][j];
			u_dev[i][j] = u[i + NCPU][j];
		}
	}
#if RECALC_LEVEL==0
	for (i = 0; i < (NMAX - NCPU); i++) {
		for (j = 0; j < NMAX; j++) {
			l_dev[i][j] = l[i + NCPU][j];
			r_dev[i][j] = r[i + NCPU][j];
			b_dev[i][j] = b[i + NCPU][j];
			t_dev[i][j] = t[i + NCPU][j];
		}
	}
#else
	for (i = 0; i < (NMAX - NCPU); i++) {
		for (j = 0; j < NMAX; j++) {
			ffh_dev[i][j] = ffh[i+NCPU][j];
			ggh_dev[i][j] = ggh[i+NCPU][j];
		}
	}
#endif
}

void mergeMatrices() {
	int i, j;
	for (i = 0; i < NCPU; i++) {
		for (j = 0; j < NMAX; j++) {
			u[i][j] = u_host[i][j];
		}
	}
	for (i = NCPU; i < NMAX; i++) {
		for (j = 0; j < NMAX; j++) {
			u[i][j] = u_dev[i - NCPU][j];
		}
	}
}

// Zeroes all values in involved matrices by using the same thread assignment used during computation
// so memory is local to assigned processors on NUMA architectures
void firstTouch(void) {
#pragma omp parallel
	{
#pragma omp master
		printf("touching data for first time\n");
		int num_threads = omp_get_num_threads();
		int sy = num_threads, sx = 1, sx_limit = (int) sqrt((double) sy);
		for (int i = sx_limit; i > 0; i--)
			if (num_threads % i == 0) {
				sx = i;
				sy = num_threads / i;
				break;
			}
		int n = (NMAX - 1);
		int m = (NCPU - 1);
		int bs_x = n / sx, bs_y = m / sy, lastoffset_x = n % sx, lastoffset_y =
				m % sy;
		int start_pos_x = 1 + bs_x * (omp_get_thread_num() % sx), start_pos_y =
				1 + bs_y * (omp_get_thread_num() / sx), end_pos_x = start_pos_x
				+ bs_x
				+ (omp_get_thread_num() % sx == sx - 1 ? lastoffset_x : 0),
				end_pos_y = start_pos_y + bs_y
						+ (omp_get_thread_num() / sx == sy - 1 ?
								lastoffset_y : 0);

		for (int i = start_pos_y; i < end_pos_y; i++)
			for (int i = start_pos_y; i < end_pos_y; i++)
				for (int j = start_pos_x; j < end_pos_x; j++) {
					u_host[i][j] = 0.0;
					if ((i + j) % 2 == 0) {
						ro_u_host[0][i][j / 2] = 0.0;
					} else {
						ro_u_host[1][i][j / 2] = 0.0;
					}
					w1_host[i][j] = 0.0;
					if ((i + j) % 2 == 0) {
						ro_w1_host[0][i][j / 2] = 0.0;
					} else {
						ro_w1_host[1][i][j / 2] = 0.0;
					}
#if PRECALC != 0
					ffh_host[i][j] = 0.0;
					if ((i + j) % 2 == 0) {
						ro_ffh_host[0][i][j / 2] = 0.0;
					} else {
						ro_ffh_host[1][i][j / 2] = 0.0;
					}
					ggh_host[i][j] = 0.0;
					if ((i + j) % 2 == 0) {
						ro_ggh_host[0][i][j / 2] = 0.0;
					} else {
						ro_ggh_host[1][i][j / 2] = 0.0;
					}
#else
					l_host[i][j] = 0.0;
					r_host[i][j] = 0.0;
					t_host[i][j] = 0.0;
					b_host[i][j] = 0.0;
					if ((i + j) % 2 == 0) {
						ro_l_host[0][i][j / 2] = 0.0;
					} else {
						ro_l_host[1][i][j / 2] = 0.0;
					}
					if ((i + j) % 2 == 0) {
						ro_r_host[0][i][j / 2] = 0.0;
					} else {
						ro_r_host[1][i][j / 2] = 0.0;
					}
					if ((i + j) % 2 == 0) {
						ro_t_host[0][i][j / 2] = 0.0;
					} else {
						ro_t_host[1][i][j / 2] = 0.0;
					}
					if ((i + j) % 2 == 0) {
						ro_b_host[0][i][j / 2] = 0.0;
					} else {
						ro_b_host[1][i][j / 2] = 0.0;
					}
#endif
				}
	}
}

int main(int argc, char* argv[]) {
	long i, j, k, metr;
	int epilogi, periptosi = 0, ektyp1, ektyp2;
	long maxiter;
	double D;
	int nn = NMAX - 1;
	int n = nn - 1;
	int gperiptosi = 0;
	double min_w1, max_w1, min_w2, max_w2, min_m_low = DBL_MAX, max_m_low =
			DBL_MAX, min_m_up = DBL_MIN, max_m_up = DBL_MIN;
	double x, y;
	double h = 1. / (n + 1.0);
	double C_E, C_W, C_N, C_S, g1, g2;
	double e1 = 1.0e-6, re;
	char filename[20];
	FILE *arxeio;
	double pi;
	long cImags = 0, cReals = 0;
	float reorderFwTime = 0, reorderInvTime = 0, GPUreorderFwTime = 0,
			GPUreorderInvTime = 0;

	if (argc != 7) {
		printf(
				"\nSyntax:\nlmsor_gpu FILENAME MAX_ITERATIONS SELECTION RE PRINT_RANGE PRINT_SOL\n");
	}

	if (argc >= 2)
		strcpy(filename, argv[1]);
	else {
		printf("\n dose to onoma tou arxeiou ektyposis apotelesmatwn :  ");
		if (scanf("%s", filename) == 0)
			exit(0);
		/* gets(filename);  */
	}
	printf("Output: %s\n", filename);
	if ((arxeio = fopen(filename, "w")) == NULL) {
		printf("i fopen apetixe \n");
		exit(0);
	}

	pi = 4.0 * atan(1.);
	if (argc >= 3)
		maxiter = atoi(argv[2]);
	else {
		printf("megisto epitrepto plithos epanalipsewn ( maxiter ) : ");
		if (scanf("%ld", &maxiter) == 0)
			exit(0);
	}
	printf("\n maxiter = %ld ", maxiter);
	if (argc >= 4)
		epilogi = atoi(argv[3]);
	else {
		printf(
				"\n epilogi twn syntelestwn ths merikhs diaforikhs exiswsis(PDE) : ");
		if (scanf("%d", &epilogi) == 0)
			exit(0);
	}
	printf("\n epilogi = %d\n", epilogi);
	if (argc >= 5)
		re = atof(argv[4]);
	else {
		printf("\n re = ");
		if (scanf("%lf", &re) == 0)
			exit(0);
	}
	printf("\n re = %lf\n", re);

	if (argc >= 7) {
		ektyp1 = atoi(argv[5]);
		ektyp2 = atoi(argv[6]);
	} else {
		printf("\n dose gia ektyposi range_w=1 , ektyposi lisis=1\n");
		if (scanf("%d %d", &ektyp1, &ektyp2) == 0)
			exit(0);
	}

	fprintf(arxeio,
			"-----------Local MSOR 5-points (LMSOR_RB)------------\n\n\n");
	fprintf(arxeio, " PDE = %2d           Re =  %.lf", epilogi, re);

	allocMem();

	firstTouch();

#ifdef __SIMD_SUPPORTED__
	printf("SIMD (SSE2) instructions are supported! :)\n");
#ifdef _INTRINSIC_SSE2_
	printf("Using SSE2 hand written intrinsic code...\n");
#endif
#endif

	fprintf(arxeio, "\n h = %lf \t  n = %6d\n\n", h, n);
	timestamp ts_start = getTimestamp();
	int tperiptosi = 0;
	for (i = 0; i <= n + 1; i++) {
		x = i * h;
		for (j = 0; j <= n + 1; j++) {
			y = j * h;

			double v_ffh = (1. / 2.) * h * FF(epilogi, re, x, y);
			double v_r = 1 - v_ffh; /* u[i+1][j]  */
			double v_l = 1 + v_ffh; /* u[i-1][j]  */
			double v_ggh = (1. / 2.) * h * GG(epilogi, re, x, y);
			double v_t = 1 - v_ggh; /* u[i][j+1]  */
			double v_b = 1 + v_ggh; /* u[i][j-1]  */
			D = v_r + v_l + v_t + v_b;
			v_r = v_r / D;
			v_l = v_l / D;
			v_t = v_t / D;
			v_b = v_b / D;

#if PRECALC == 0 &&  RECALC_LEVEL == 0
			r[i][j] = v_r;
			l[i][j] = v_l;
			t[i][j] = v_t;
			b[i][j] = v_b;
#elif PRECALC != 0 && RECALC_LEVEL != 0
			ffh[i][j] = v_ffh;
			ggh[i][j] = v_ggh;
#endif

			C_E = v_r;
			C_W = v_l;
			C_N = v_t;
			C_S = v_b;

			/* ektimhsh tou w_opt  */
			if (C_E * C_W * C_N * C_S >= 0) {
				if (C_E * C_W >= 0 && C_N * C_S >= 0)
					periptosi = 1;
				else if (C_E * C_W <= 0 && C_N * C_S <= 0)
					periptosi = 2;
			} else {
				if (C_E * C_W > 0 && C_E + C_W >= 0)
					periptosi = 3;
				else if (C_E * C_W < 0 && C_N + C_S >= 0)
					periptosi = 4;
				else
					periptosi = 5;
			}
			if (tperiptosi == 0)
				tperiptosi = periptosi;
			else if (tperiptosi != periptosi)
				tperiptosi = -1;

			double m_up = DBL_MIN, m_low = DBL_MAX;

			switch (periptosi) {
			case 1:
				/* case 1 (Im(m)=0) real */
				m_up = 2. * (sqrt(C_E * C_W) + sqrt(C_N * C_S)) * cos(pi * h);
				m_low = 2. * (sqrt(C_E * C_W) + sqrt(C_N * C_S))
						* cos(pi * (1. - h) / 2.);
#ifdef MODIFIED_SOR
				w1[i][j] = 2./(1.-m_up*m_low+sqrt((1.-m_up)*(1.-m_low)));
				w2[i][j] = 2./(1.+m_up*m_low+sqrt((1.-m_up)*(1.-m_low)));
#else
				w1[i][j] = w2[i][j] = 2. / (1. + sqrt(1. - m_up * m_up));
#endif
				cReals++;
				break;

			case 2:
				/* case 2 (Re(m)=0) imaginary */
				m_up = 2. * (sqrt(-C_E * C_W) + sqrt(-C_N * C_S)) * cos(pi * h);
				m_low = 2. * (sqrt(-C_E * C_W) + sqrt(-C_N * C_S))
						* cos(pi * (1. - h) / 2.);

#ifdef MODIFIED_SOR
				w1[i][j] = 2./(1.-m_up*m_low+sqrt((1.+pow(m_up,2.0))*(1.+pow(m_low,2.0))));
				w2[i][j] = 2./(1.+m_up*m_low+sqrt((1.+pow(m_up,2.0))*(1.+pow(m_low,2.0))));
#else
				w1[i][j] = w2[i][j] = 2. / (1. + sqrt(1. + m_up * m_up));
#endif
				cImags++;
				break;

			case 3:
				/* case 3a  complex  */
				g1 = pow((1. - pow((C_E + C_W), 2. / 3.)), -1. / 2.);
				w1[i][j] = 2. / (1. + g1 * fabs(C_N - C_S));
				break;

			case 4:
				/* case 3b complex  */
				g2 = pow((1. - pow((C_N + C_S), 2. / 3.)), -1. / 2.);
				w1[i][j] = 2. / (1. + g2 * fabs(C_E - C_W));
				w2[i][j] = w1[i][j];
				break;

			case 5:
				/* alli periptosi */
				printf("\n####h methodos diakoptetai####\n");
				fprintf(arxeio, "\n###### h methodos diakoptetai #######\n");
				break;

			default:
				printf("\n lathos periptosi");
				break;
			}
			validate_w(w1[i][j]);
			validate_w(w2[i][j]);

			if (i == 1 && j == 1) {
				min_m_low = max_m_low = m_low;
				min_m_up = max_m_up = m_up;
			} else if (i > 0 && i <= n && j >= 1 && j <= n) {
				min_max(m_low, &min_m_low, &max_m_low);
				min_max(m_up, &min_m_up, &max_m_up);
			}
		} /* end for j */
		{
			if (tperiptosi != 0 && gperiptosi != -1
					&& tperiptosi != gperiptosi) {
				if (gperiptosi != 0)
					gperiptosi = -1;
				else
					gperiptosi = tperiptosi;
			}
		}
	} /*end for i */
	printf(" %f seconds\n", getElapsedtime(ts_start) / 1000.0);

	switch (gperiptosi) {
	case 1:
		printf("\n ---Pragmatikh periptosi---- \n");
		break;
	case 2:
		printf("\n ---Fantastikh periptosi---- \n");
		break;
	default:
		printf("\n ---Mikth periptosi---- \n");
		break;
	}
#ifdef MODIFIED_SOR
	printf("R/B LMSOR method\n");
#else
	printf("R/B LOCAL SOR method\n");
#endif

// Support omega values input from the environment
	char *envOmega1 = getenv("LMSOR_OMEGA1"), *envOmega2 = getenv(
			"LMSOR_OMEGA2");
	if (envOmega1 && envOmega2) {
		cImags = cReals = 0;
		min_w1 = max_w1 = min_w2 = max_w2 = min_m_low = max_m_low = min_m_up =
				max_m_up = 0.0;
		double omega1 = atof(envOmega1), omega2 = atof(envOmega2);
		printf("\nNote: Read omega values from the environment (%.3f, %.3f)\n",
				omega1, omega2);
		min_w1 = max_w1 = omega1;
		min_w2 = max_w2 = omega2;
		for (i = 0; i <= n + 1; i++) {
			for (j = 0; j <= n + 1; j++) {
				w1[i][j] = omega1;
				w2[i][j] = omega2;
			}
		}
	}

	min_max_MAT(w1, n, &min_w1, &max_w1);
	min_max_MAT(w2, n, &min_w2, &max_w2);

	printf("PDE :  %2d      h =  %lf     n = %6d \n", epilogi, h, n);
	printf(
			"\n periptosi:  %2d RANGE_m_low = %5.4lf - %5.4lf \t RANGE_m_up = %5.4lf - %5.4lf\n",
			periptosi, min_m_low, max_m_low, min_m_up, max_m_up);

	fprintf(arxeio, "\n  %3d  \t  %5.4lf - %5.4lf \t %5.4lf - %5.4lf\n",
			periptosi, min_m_low, max_m_low, min_m_up, max_m_up);

// create unified matrix w
	for (i = 0; i <= n + 1; i++) {
		for (j = 0; j <= n + 1; j++) {
			w1[i][j] = ((i + j) % 2 == 0) ? w1[i][j] : w2[i][j];
		}
	}
	free(w2);
	w2 = NULL;

	for (i = 0; i <= n + 1; i++) {
		x = i * h;
		for (j = 0; j <= n + 1; j++) {
			y = j * h;
			u[i][j] = initial_guess(x, y);
		}
	}

	printCUDAMem();

	/* if calculations will be done in both CPU and GPU, split arrays between device and host */
	splitMatrices();

	ts_start = getTimestamp();
// red-black reordering on host
	fwReorder();
	reorderFwTime = getElapsedtime(ts_start) / 1000.0f;

	double calc_time;
	double exectime;
//ts_start = getTimestamp();
	metr = combinedLMSOR(NMAX, NCPU, maxiter, e1, 5.0e+8, (double*) w1_dev,
			(double*) u_dev, (double*) l_dev, (double*) r_dev, (double*) b_dev,
			(double*) t_dev, &k, epilogi, (double*) ffh_dev, (double*) ggh_dev,
			re, &calc_time, &GPUreorderFwTime, &GPUreorderInvTime);
	exectime = getElapsedtime(ts_start) / 1000.0f;
//printf(" %f seconds total execution time\n", exectime);

	ts_start = getTimestamp();
	invReorder();
	reorderInvTime = getElapsedtime(ts_start) / 1000.0f;

	/* merge back host and device matrices into a single matrix */
	mergeMatrices();

	printf(
			" %f CPU reordering overhead (%f forward reordering + %f inverse reordering)\n",
			reorderFwTime + reorderInvTime, reorderFwTime, reorderInvTime);
	printf(" %f seconds total time\n",
			reorderFwTime + reorderInvTime + GPUreorderFwTime
					+ GPUreorderInvTime + calc_time);
	printf(
			" %f total reordering overhead (%f CPU forward reordering + %f CPU inverse reordering + %f GPU forward reordering + %f GPU inverse reordering)\n",
			reorderFwTime + reorderInvTime + GPUreorderFwTime
					+ GPUreorderInvTime, reorderFwTime, reorderInvTime,
			GPUreorderFwTime, GPUreorderInvTime);

#if RECALC_LEVEL==0
	const char cKernel[] = "#1";
#elif  RECALC_LEVEL==1
	const char cKernel[] = "#2";
#elif RECALC_LEVEL==2
	const char cKernel[] = "#3";
#endif
#ifdef MODIFIED_SOR
	const char cMethod[] = "R/B LMSOR (GPU)";
#else
	const char cMethod[] = "R/B LOCAL SOR (GPU)";
#endif
	printf(
			"___\nEXCEL friendly line follows:\n%d;%d;%d;%d;%.1f;%d;(%ld,%ld);%s;%s;%ld;%f;%f;%d\n___\n",
			NMAX, NCPU, (NMAX - NCPU), epilogi, re, gperiptosi, cReals, cImags,
			cMethod, cKernel, metr, calc_time, exectime, k != 2);

	if (k == 2) {
		printf("oxi sygklisi ****");
		printf("\n periptosi: %3d ", periptosi);
		fprintf(arxeio, " oxi sygklisi  ****");
		exit(0);
	} else {
		printf("NITER = %6ld   ", metr);
		printf("\n periptosi: %3d", periptosi);
	}

	if (ektyp1 == 1) {
		printf("RANGE_w1 = %5.4lf - %5.4lf \t RANGE_w2 = %5.4lf - %5.4lf\n",
				min_w1, max_w1, min_w2, max_w2);

		printf(
				"\n periptosi: %3d RANGE_m_low = %5.4lf - %5.4lf \t RANGE_m_up = %5.4lf - %5.4lf\n",
				periptosi, min_m_low, max_m_low, min_m_up, max_m_up);

		fprintf(arxeio, "\n  %5ld \t %5.4lf - %5.4lf \t %5.4lf - %5.4lf \n",
				metr, min_m_low, max_m_low, min_m_up, max_m_up);

		fprintf(arxeio, "\n  %3d \t %5.4lf - %5.4lf \t %5.4lf - %5.4lf \n",
				periptosi, min_m_low, max_m_low, min_m_up, max_m_up);

		fprintf(arxeio,
				"..............................................................\n");
	}

	if (ektyp2 == 1) {
		for (j = n + 1; j >= 0; j--) {
			for (i = 0; i <= n + 1; i++) {
//          fprintf(arxeio,"  %5.4e",u1[i][j]);
				fprintf(arxeio, "  %5.4e", u[i][j]);
			}
			fprintf(arxeio, "\n");
		}
		fprintf(arxeio,
				" NITER = %6ld    RANGE_w1 = %5.4lf - %5.4lf \t RANGE_w2 = %5.4lf - %5.4lf \n",
				metr, min_w1, max_w1, min_w2, max_w2);
		fprintf(arxeio,
				" periptosi:  %3d   RANGE_m_low =  %5.4lf - %5.4lf \t RANGE_m_up= %5.4lf - %5.4lf \n",
				periptosi, min_m_low, max_m_low, min_m_up, max_m_up);
		fprintf(arxeio, "\n...............................................\n");
	}
	fclose(arxeio);
	if (argc < 2)
		mypause();

	return 0;
}

double initial_guess(double xx, double yy) {
	double u_arx;
	u_arx = xx * yy * (1. - xx) * (1. - yy);
	return u_arx;
}

double FF(int epilog, double re, double xx, double yy) {
	double f = 0.0;
	switch (epilog) {
	case 1:
		f = re * pow((2. * xx - 10.), 3.0);
		break;
	case 2:
		f = re * (2. * xx - 10.);
		break;
	case 3:
		f = re * pow(10., 4.0);
		break;
	case 4:
		f = re * pow((2. * xx - 10.), 5.0);
		break;
	case 5:
		f = re * pow(xx, 2.0);
		break;
	case 6:
		f = re * (10. + pow(xx, 2.0));
		break;
	case 7:
		f = re * pow((10. + pow(xx, 2.0)), 2.0);
		break;
	case 8:
		f = re * pow((2 * xx - 1.), 3.0);
		break;
	case 9:
		f = re * pow(xx, 2.0);
		break;
	case 10:
		/*          f = (1./2.)*re*(1.+pow(xx,2));*/
		f = re * (1. + pow(xx, 2.0));
		break;
	case 11:
		f = re * (1. - 2. * xx);
		break;
	case 12:
		f = re * (2. * xx - 1.);
		break;
	case 13:
		f = re * (10. - 2 * xx);
		break;
	case 14:
		f = 0.;
		break;
	default:
		printf("sfalma ston syntelesth FF\n");
		break;
	}
	return (f);
}

double GG(int epilog, double re, double xx, double yy) {
	double g = 0.0;
	switch (epilog) {
	case 1:
		g = re * pow((2 * yy - 10.), 3.0);
		break;
	case 2:
		g = re * (2 * yy - 10.);
		break;
	case 3:
		g = re * pow(10., 4.0);
		break;
	case 4:
		g = re * pow((2. * yy - 10.), 5.0);
		break;
	case 5:
		g = re * pow(xx, 2.0);
		break;
	case 6:
		g = re * (10. + pow(yy, 2.0));
		break;
	case 7:
		g = re * pow((10. + pow(yy, 2.0)), 2.0);
		break;
	case 8:
		g = 0;
		break;
	case 9:
		g = 0;
		break;
	case 10:
		g = 100.;
		break;
	case 11:
		g = re * (1. - 2. * yy);
		break;
	case 12:
		g = re * (2. * yy - 1.);
		break;
	case 13:
		g = re * (10. - 2. * yy);
		break;
	case 14:
		g = 0.;
		break;
	default:
		printf("sfalma ston syntelesth GG\n");
		break;
	}
	return (g);
}

void min_max(double value, double *min_, double *max_) {
	if (*min_ > value) {
		*min_ = value;
	}
	if (*max_ < value) {
		*max_ = value;
	}
}

void min_max_MAT(double MAT[][NMAX], int n, double *min_MAT, double *max_MAT) {
	int i, j;

	*min_MAT = MAT[1][1];

	*max_MAT = MAT[1][1];

	for (i = 1; i <= n; i++) {
		for (j = 1; j <= n; j++) {
			if (*min_MAT > MAT[i][j]) {
				*min_MAT = MAT[i][j];
			}
			if (*max_MAT < MAT[i][j]) {
				*max_MAT = MAT[i][j];
			}
		}
	}
}
