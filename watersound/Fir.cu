#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__constant__ __device__ float coef_filter_GPU[258] = { 0, 0.0007f, 0.0007f, 0.0007f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0009f, 0.0009f,
0.0009f, 0.0009f, 0.0010f, 0.0010f, 0.0010f, 0.0011f,
0.0011f, 0.0011f, 0.0011f, 0.0011f, 0.0012f, 0.0012f, 0.0012f, 0.0012f, 0.0012f, 0.0012f, 0.0011f, 0.0011f, 0.0011f, 0.0010f, 0.0010f,
0.0009f, 0.0008f, 0.0007f, 0.0006f, 0.0005f, 0.0004f, 0.0002f, 0.0001f, -0.0001f, -0.0003f, -0.0005f, -0.0007f, -0.0009f, -0.0011f,
-0.0014f, -0.0016f, -0.0019f, -0.0022f, -0.0025f, -0.0028f, -0.0031f, -0.0034f, -0.0037f, -0.0040f, -0.0043f, -0.0047f, -0.0050f,
-0.0053f, -0.0056f, -0.0060f, -0.0063f, -0.0066f, -0.0068f, -0.0071f, -0.0074f, -0.0076f, -0.0079f, -0.0081f, -0.0083f, -0.0084f,
-0.0086f, -0.0087f, -0.0088f, -0.0088f, -0.0088f, -0.0088f, -0.0088f, -0.0087f, -0.0086f, -0.0085f, -0.0083f, -0.0081f, -0.0079f,
-0.0076f, -0.0073f, -0.0069f, -0.0065f, -0.0061f, -0.0057f, -0.0052f, -0.0047f, -0.0041f, -0.0036f, -0.0030f, -0.0023f, -0.0017f,
-0.0010f, -0.0003f, 0.0004f, 0.0011f, 0.0018f, 0.0025f, 0.0033f, 0.0040f, 0.0047f, 0.0055f, 0.0062f, 0.0069f, 0.0076f, 0.0083f, 0.0090f,
0.0097f, 0.0103f, 0.0109f, 0.0115f, 0.0120f, 0.0125f, 0.0130f, 0.0135f, 0.0139f, 0.0142f, 0.0146f, 0.0148f, 0.0151f, 0.0153f, 0.0154f,
0.0155f, 0.0156f, 0.0156f, 0.0155f, 0.0154f, 0.0153f, 0.0151f, 0.0148f, 0.0146f, 0.0142f, 0.0139f, 0.0135f, 0.0130f, 0.0125f, 0.0120f,
0.0115f, 0.0109f, 0.0103f, 0.0097f, 0.0090f, 0.0083f, 0.0076f, 0.0069f, 0.0062f, 0.0055f, 0.0047f, 0.0040f, 0.0033f, 0.0025f, 0.0018f,
0.0011f, 0.0004f, -0.0003f, -0.0010f, -0.0017f, -0.0023f, -0.0030f, -0.0036f, -0.0041f, -0.0047f, -0.0052f, -0.0057f, -0.0061f, -0.0065f,
-0.0069f, -0.0073f, -0.0076f, -0.0079f, -0.0081f, -0.0083f, -0.0085f, -0.0086f, -0.0087f, -0.0088f, -0.0088f, -0.0088f, -0.0088f, -0.0088f,
-0.0087f, -0.0086f, -0.0084f, -0.0083f, -0.0081f, -0.0079f, -0.0076f, -0.0074f, -0.0071f, -0.0068f, -0.0066f, -0.0063f, -0.0060f, -0.0056f,
-0.0053f, -0.0050f, -0.0047f, -0.0043f, -0.0040f, -0.0037f, -0.0034f, -0.0031f, -0.0028f, -0.0025f, -0.0022f, -0.0019f, -0.0016f, -0.0014f,
-0.0011f, -0.0009f, -0.0007f, -0.0005f, -0.0003f, -0.0001f, 0.0001f, 0.0002f, 0.0004f, 0.0005f, 0.0006f, 0.0007f, 0.0008f, 0.0009f, 0.0010f,
0.0010f, 0.0011f, 0.0011f, 0.0011f, 0.0012f, 0.0012f, 0.0012f, 0.0012f, 0.0012f, 0.0012f, 0.0011f, 0.0011f, 0.0011f, 0.0011f, 0.0011f,
0.0010f, 0.0010f, 0.0010f, 0.0009f, 0.0009f, 0.0009f, 0.0009f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0007f, 0.0007f, 0.0007f, 0 };


/*
UP: upsample ratio
M: input processing sample per thread, UP*M should be less than 24, limited by register usage,
UP*M is output sample processing per thread
in x: input signal, buffer should be in begining of x
out y: output signal
in x_len: the input signal length, it satisfy x_len = seg_len * gridDim.x + h_len + padding
in seg_len: input sample is split into segments, seg_len <= M * blockDim.x, since limited by share memory size, seg_len +h_len < 12K / block per MP
in h_len: filter length, h_len *UP = actual filter len, since limited by code h_len < seg_len /M, h_len < blockDim.x
in y_len: y_len >= seg_len * gridDim.x * UP
thread is used for one channel M input pcm and one channel UP*M output
grid.y is for channel
grid.x is for segment
*/
template <int UP, int M>
__global__ void cuda_fir_filter(float * __restrict x, float * __restrict y,
	int x_len, int seg_len, int h_len, int y_len)
{
	int x_base = blockIdx.y * x_len + blockIdx.x * seg_len;
	extern __shared__ float x_smem[];
	float *h = coef_filter_GPU;
#pragma unroll

	for (int i = 0, x1 = threadIdx.x; i <= M; i++, x1 += blockDim.x)
		x_smem[x1] = x[x1 + x_base];

	__syncthreads();

	x_base = threadIdx.x * M;

	/*
	y0 = x0*h[u-1] + x1*h[2u-1] + xi*h[i*u+u-1]
	y1 = x0*h[u-2] + x1*h[2u-2] + xi*h[i*u+u-2]
	...
	y[u-1] = x0*h0 + x1*h[u] + xi*h[i*u]
	y[u] = x1*h[u-1] + x2*h[2u-1] + xi*h[i*u-1]
	y[u+1] = x1*h[u-2] + x2*h[2u-2] + xi*h[i*u-2]
	...

	*/
	int y_base = blockIdx.y * y_len + blockIdx.x * seg_len* UP + threadIdx.x * M *UP;
#if 1

	float y_reg[M][UP];
#pragma unroll 
	for (int j = 0; j < M; j++)
	{
#pragma unroll
		for (int k = 0; k < UP; k++)
			y_reg[j][k] = 0;
	}

	for (int i = 0, h_base = 0; i < h_len; i++, x_base++, h_base += UP)
	{
#pragma unroll
		for (int j = 0; j < M; j++)
		{
#pragma unroll
			for (int k = 0; k < UP; k++)
				y_reg[j][k] += x_smem[x_base + j] * h[h_base + UP - k - 1];
		}
	}

#pragma unroll 
	for (int j = 0; j < M; j++)
	{
#pragma unroll
		for (int k = 0; k < UP; k++)
			if (threadIdx.x * M *UP + j*UP + k < seg_len* UP)
				y[y_base + j*UP + k] = y_reg[j][k];
	}

#else
	if (M == 4 && UP == 3)
	{
		float y_reg_0_0 = 0, y_reg_0_1 = 0, y_reg_0_2 = 0;
		float y_reg_1_0 = 0, y_reg_1_1 = 0, y_reg_1_2 = 0;
		float y_reg_2_0 = 0, y_reg_2_1 = 0, y_reg_2_2 = 0;
		float y_reg_3_0 = 0, y_reg_3_1 = 0, y_reg_3_2 = 0;
		for (int i = 0, h_base = 0; i < h_len; i++, x_base++, h_base += UP)
		{
			y_reg_0_0 += x_smem[x_base + 0] * h[h_base + 2];
			y_reg_0_1 += x_smem[x_base + 0] * h[h_base + 1];
			y_reg_0_2 += x_smem[x_base + 0] * h[h_base + 0];
			y_reg_1_0 += x_smem[x_base + 1] * h[h_base + 2];
			y_reg_1_1 += x_smem[x_base + 1] * h[h_base + 1];
			y_reg_1_2 += x_smem[x_base + 1] * h[h_base + 0];
			y_reg_2_0 += x_smem[x_base + 2] * h[h_base + 2];
			y_reg_2_1 += x_smem[x_base + 2] * h[h_base + 1];
			y_reg_2_2 += x_smem[x_base + 2] * h[h_base + 0];
			y_reg_3_0 += x_smem[x_base + 3] * h[h_base + 2];
			y_reg_3_1 += x_smem[x_base + 3] * h[h_base + 1];
			y_reg_3_2 += x_smem[x_base + 3] * h[h_base + 0];
		}
		if (threadIdx.x * M *UP + 0 * UP + 0 < seg_len* UP)
			y[y_base + 0 * UP + 0] = y_reg_0_0;
		if (threadIdx.x * M *UP + 0 * UP + 1 < seg_len* UP)
			y[y_base + 0 * UP + 1] = y_reg_0_1;
		if (threadIdx.x * M *UP + 0 * UP + 2 < seg_len* UP)
			y[y_base + 0 * UP + 2] = y_reg_0_2;
		if (threadIdx.x * M *UP + 1 * UP + 0 < seg_len* UP)
			y[y_base + 1 * UP + 0] = y_reg_1_0;
		if (threadIdx.x * M *UP + 1 * UP + 1 < seg_len* UP)
			y[y_base + 1 * UP + 1] = y_reg_1_1;
		if (threadIdx.x * M *UP + 1 * UP + 2 < seg_len* UP)
			y[y_base + 1 * UP + 2] = y_reg_1_2;
		if (threadIdx.x * M *UP + 2 * UP + 0 < seg_len* UP)
			y[y_base + 2 * UP + 0] = y_reg_2_0;
		if (threadIdx.x * M *UP + 2 * UP + 1 < seg_len* UP)
			y[y_base + 2 * UP + 1] = y_reg_2_1;
		if (threadIdx.x * M *UP + 2 * UP + 2 < seg_len* UP)
			y[y_base + 2 * UP + 2] = y_reg_2_2;
		if (threadIdx.x * M *UP + 3 * UP + 0 < seg_len* UP)
			y[y_base + 3 * UP + 0] = y_reg_3_0;
		if (threadIdx.x * M *UP + 3 * UP + 1 < seg_len* UP)
			y[y_base + 3 * UP + 1] = y_reg_3_1;
		if (threadIdx.x * M *UP + 3 * UP + 2 < seg_len* UP)
			y[y_base + 3 * UP + 2] = y_reg_3_2;
	}

#endif	
}

void cuda_fir(int up, int m, dim3 grid, int thread_num, int smem_size, cudaStream_t * stream, float * x, float * y,
	int x_len, int seg_len, int h_len, int y_len)
{	
	if (up == 3 && m == 4)
		cuda_fir_filter<3, 4> << <grid, thread_num, smem_size, stream[0] >> >(
		x, y, x_len, seg_len, h_len, y_len);
}