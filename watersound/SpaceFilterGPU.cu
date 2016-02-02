
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SpaceFilter.h"
#include "math.h"
#include <complex>
#define __CUDA_ARCH__ 300
#if __CUDA_ARCH__ >= 300
#define APT 45
#else
#define APT 10
#endif
#define ANGLE_GROUP ((180-1)/APT+1)
#define UP 3
#define M 4


__constant__ __device__ short indew_CBF_time_gpu[40 * 180+30];
__constant__ __device__ short indew_time_load_offset_gpu[40 * ANGLE_GROUP];
__constant__ __device__ unsigned short indew_time_load_size_gpu[40 * ANGLE_GROUP];
__constant__ __device__ cufftComplex target_freq_gpu[FFT_LEN / 2 + 1];

#if 0
/*
vector substract
y=x1-x2, length is len
*/
__global__ void cuda_sub_vec(float * __restrict x1, float * __restrict x2, int len, float * __restrict y)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx<len)
		y[idx] = x1[idx] - x2[idx];
}

/*
	y(i,j) = x(i,j) - x(i+1,j+offset)
*/
__global__ void cuda_cancel_noise2(float * __restrict x, int x_len, 
	int offset, float * __restrict y, int y_len, int len)
{
	int i = blockIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < len)
		y[i*y_len + j] = x[i*x_len + j] - x[(i + 1)*x_len + j + offset];
}
#endif
/*
y(i,j) = x(i,j) - x(i+1,j+offset)
x_len: input one row length of x
y_len: input one row length of y
len: input one line substract length
offset: input, see above
*/
__global__ void cuda_cancel_noise(float * __restrict x, int x_len,
	int offset, float * __restrict y, int y_len, int len)
{
	int i = blockIdx.y;
	for (int j = threadIdx.x; j < len; j+=blockDim.x)
		y[i*y_len + j] = x[i*x_len + j] - x[(i + 1)*x_len + j + offset];
}

/*
y(k,j) = sum(x[i,offset[i,k]+j*up]), loop i=0..row-1. k=0..179,j is PCM index. offset is space filter delay
thread is used for one pcm point and APT angles
Grid.x is for pcm
Grid.y is for angle group
x_len: input one row length of x
y_len: input one row length of y
x: input seperate channel pcm
y: output space filter pcm, [180 * ]
len: output one line length
row: channel number
*/
__global__ void cuda_space_filter(float * __restrict x, int x_len, 
	float * __restrict y, int y_len, int up, int len, int row)
{
	int x_base = blockIdx.x * blockDim.x*up; //for pcm
	int a_base = blockIdx.y * APT; //for angle
	float sum[APT];
	extern __shared__ float x_smem[];

#pragma unroll
	for (int k = 0; k < APT; k++)
		sum[k] = 0;

	for (int i = 0; i < row; i++)
	{

		for (int j = threadIdx.x; j < indew_time_load_size_gpu[i*ANGLE_GROUP + blockIdx.y]; j += blockDim.x)
			x_smem[j] = x[x_base + j];
		x_base += x_len + indew_time_load_offset_gpu[i*ANGLE_GROUP + blockIdx.y];
		__syncthreads();

		int x_smem_base = threadIdx.x*up;
#pragma unroll
		for (int k = 0; k < APT; k++)
			sum[k] += x_smem[x_smem_base + indew_CBF_time_gpu[a_base + k]];
			
		a_base += 180;
		__syncthreads();
	}

	a_base = blockIdx.y * APT;
	int pcm_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ((pcm_idx &7) == 0) {
		pcm_idx = pcm_idx >>3;
#pragma unroll	
		for (int k = 0; k < APT; k++)
			if (a_base + k < 180 && pcm_idx< len)
				y[(a_base + k)*y_len + pcm_idx] = sum[k] / row;
	}
}

/* x[i, freq] -= x[i+cd, freq] * e^(delay)
*/
__global__ void cuda_freq_cancel_noise(cufftComplex * __restrict x, int x_len, int freq_start, 
	float angle, int row, float mic_d, int cd)
{
	int i;
	int freq_idx = threadIdx.x + freq_start;
	float t_offset = mic_d * cd * cos(angle *pi / 180) / 1500 * 2 * pi;
	cufftComplex t, a, b;
	t.x = cos(freq_idx * t_offset);
	t.y = sin(freq_idx * t_offset);
	for (i = 0; i < row - cd; i++) {
		a = x[(i+cd)*x_len + freq_idx];
		b.x = a.x * t.x - a.y * t.y;
		b.y = a.x * t.y + a.y * t.x;
		a = x[i*x_len + freq_idx];
		a.x -= b.x;
		a.y -= b.y;
		x[i*x_len + freq_idx] = a;
	}
		
}

/*
sum(k,j) = sum(x[k,i]* d[k,i,j])  loop i=0..row-1. k=0..freq_len-1, j=0..179
y(k,j) = abs(sum(k,j)^2
thread.x(j) is for angle
grid.x(k) is for freq
row: channel number, must <=180
*/
__global__ void cuda_freq_space_filter(cufftComplex * __restrict x, int x_len, int freq_start,
	float * __restrict y, int y_len, int row, float mic_d)
{

	int angle_idx = threadIdx.x;
	int freq_idx = blockIdx.x;
	cufftComplex sum;
	extern __shared__ cufftComplex xc_smem[];

	if (threadIdx.x < row)
		xc_smem[threadIdx.x] = x[threadIdx.x*x_len + freq_idx + freq_start];
	sum.x = sum.y = 0;
	__syncthreads();

	float t_array = mic_d * cos(pi * angle_idx / 180) / 1500 * 2 * pi;

	for (int i = 0; i < row; i++) {
		cufftComplex t, d;
		float delay = t_array * (freq_idx + freq_start)*i;
		d.x = cos(delay);
		d.y = sin(delay); //d = e^(j*delay)
		t.x = d.x * xc_smem[i].x - d.y * xc_smem[i].y;
		t.y = d.x * xc_smem[i].y + d.y * xc_smem[i].x; //t=d*xc_smem
		sum.x += t.x;
		sum.y += t.y; //sum += xc_smem *e^(j*delay)
	}

	y[freq_idx*y_len + angle_idx] = (sum.x /row) * (sum.x/row) + (sum.y/row) * (sum.y/row); //y = norm(sum)

}

/*
	x(i) = x(i) * target(i)
*/
__global__ void cuda_multiply(cufftComplex * __restrict x, int x_len, int len)
{
	int i = blockIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	cufftComplex s, a, b;
	if (j < len) {
		a = x[i*x_len + j];
		b = target_freq_gpu[j];
		s.x = (a.x * b.x - a.y * b.y);
		s.y = (a.x * b.y + a.y * b.x);
		x[i*x_len + j] = s;
	}
		
}

SpaceFilterGPU::SpaceFilterGPU(int channel, int sample) : SpaceFilter(channel, sample)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	if (sample == 16000) {
		x_len = 16384;
		seg_len = 4000;
		y_len = 16000 * UP * 2;		
		h_len = 258 / 3;
	}
	target_len = 1;
	cudaStatus = cudaMalloc((void**)&sig_array40_gpu, x_len * channel * sizeof(float) + 4096);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc device sig_array40_gpu failed!");

	cudaStatus = cudaMalloc((void**)&sig_3fs_array40_gpu, y_len * channel * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc device sig_3fs_array40_gpu failed!");
	cudaStatus = cudaMemset(sig_3fs_array40_gpu, 0, y_len * channel * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemset device sig_3fs_array40_gpu failed!");

	cudaStatus = cudaMalloc((void**)&sig_offset_array40_gpu, y_len * channel * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc device sig_offset_array40_gpu failed!");
	
	cudaStatus = cudaMalloc((void**)&sig_fs_2s_cbf_gpu, FFT_LEN * 180 * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc device sig_fs_2s_cbf_gpu failed!");
	cudaStatus = cudaMemset(sig_fs_2s_cbf_gpu, 0, FFT_LEN * 180 * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemset device sig_fs_2s_cbf_gpu failed!");

	cudaStatus = cudaMalloc((void**)&sig_freq_gpu, (FFT_LEN+2) * 180 * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc device sig_freq_gpu failed!");

	cudaStatus = cudaMalloc((void**)&sig_conv_gpu, FFT_LEN * 180 * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc device sig_conv_gpu failed!");
	
	cudaStatus = cudaMallocHost((void**)&sig_array40_cpu, x_len * channel * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc host x failed!");
#if CHENYU_DBG
	cudaStatus = cudaMallocHost((void**)&dbg_array_cpu, sample * 2 * 180 * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc host x failed!");
#endif

	stream = (cudaStream_t *)malloc(sizeof(cudaStream_t)*channel);
	for (int i = 0; i < channel; i++) {
		cudaStatus = cudaStreamCreate(&stream[i]);
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaStreamCreate failed");
	}

	if (cufftPlan1d(&fft_conv, FFT_LEN, CUFFT_R2C, 180) != CUFFT_SUCCESS)
		fprintf(stderr, "CUFFT error: Plan fft conv creation failed");
	cufftSetStream(fft_conv, stream[0]);
	if (cufftPlan1d(&fft_inv_conv, FFT_LEN, CUFFT_C2R, 180) != CUFFT_SUCCESS)
		fprintf(stderr, "CUFFT error: Plan fft inv conv creation failed");
	cufftSetStream(fft_inv_conv, stream[0]);
	
	memset(sig_array40_cpu, 0, x_len * channel * sizeof(float));	
	
	set_noise_para(true, 15);
#if CHENYU_DBG & 1
	printf("SpaceFilterGPU create success, channel=%d, sample=%d\nx_len=%d, seg_len=%d, smem_size=%d, y_len=%d, APT=%d\n", 
		channel, sample, x_len, seg_len, 1024 * (M + 1) * sizeof(float), y_len, APT);
#endif
#if CHENYU_DBG & 2
	f_dump_upfs = fopen("gpu_upsample.txt", "wt");
	if (f_dump_upfs == NULL)
		fprintf(stderr, "open gpu_upsample.txt failed\n");
#endif
#if CHENYU_DBG & 4
	f_dump_offset = fopen("gpu_cancel_noise.txt", "wt");
	if (f_dump_offset == NULL)
		fprintf(stderr, "open gpu_cancel_noise.txt failed\n");
#endif
#if CHENYU_DBG & 8
	f_dump_cbf = fopen("gpu_space_filter.txt", "wt");
	if (f_dump_cbf == NULL)
		fprintf(stderr, "open gpu_space_filter.txt failed\n");
#endif
#if CHENYU_DBG & 16
	f_dump_conv = fopen("gpu_conv.txt", "wt");
	if (f_dump_conv == NULL)
		fprintf(stderr, "open gpu_conv.txt failed\n");
#endif
}

SpaceFilterGPU::~SpaceFilterGPU()
{
	cudaFree(sig_array40_gpu);
	cudaFree(sig_3fs_array40_gpu);
	cudaFree(sig_offset_array40_gpu);
	cudaFree(sig_fs_2s_cbf_gpu);
	cudaFree(sig_freq_gpu);
	cudaFree(sig_conv_gpu);
	cudaFreeHost(sig_array40_cpu);
	cudaFreeHost(dbg_array_cpu);
	for (int i = 0; i < channel_num; i++)
		cudaStreamDestroy(stream[i]);
	cudaDeviceReset();
}

void SpaceFilterGPU::upsample(const vector<float *> pcm_in)
{
	dim3 grid(M, 1, 1);
	int thread_num = 1024;
	int smem_size = thread_num * (M + 1) * sizeof(float);
	cudaError_t cudaStatus;
	if (pcm_in.size() != channel_num) {
		fprintf(stderr, "pcm_in channel number wrong");
		return;
	}

	for (int i = 0; i < channel_num; i++) {
		cudaStatus = cudaMemcpyAsync(&sig_3fs_array40_gpu[y_len*i], &sig_3fs_array40_gpu[y_len*i + y_len / 2],
			y_len * sizeof(float) / 2, cudaMemcpyDeviceToDevice, stream[i]);
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaMemcpyAsync device x to device x failed!\n");
		memcpy(&sig_array40_cpu[x_len*i], &sig_array40_cpu[x_len*i + sample_num], (x_len - sample_num)*sizeof(float));
		memcpy(&sig_array40_cpu[x_len*i + x_len - sample_num], pcm_in[i], sample_num*sizeof(float));
		cudaStatus = cudaMemcpyAsync(&sig_array40_gpu[x_len*i], &sig_array40_cpu[x_len*i],
			x_len * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaMemcpyAsync host x to device x failed!\n");
		
		cuda_fir(UP, M, grid, thread_num, smem_size, &stream[i], 
			&sig_array40_gpu[x_len*i + x_len - sample_num - h_len + 1], 
			&sig_3fs_array40_gpu[y_len*i + y_len / 2],
			x_len, seg_len, h_len, y_len);
		
		if (cudaGetLastError()!=cudaSuccess)
			fprintf(stderr, "fir_filter launch failed\n");
	}
	
#if CHENYU_DBG & 2
	for (int i = 0; i < channel_num; i++)
		cudaMemcpyAsync(&dbg_array_cpu[y_len*i], &sig_3fs_array40_gpu[y_len*i + y_len / 2],
		y_len * sizeof(float) / 2, cudaMemcpyDeviceToHost, stream[i]);

	cudaDeviceSynchronize();
	for (int i = 0; i < channel_num; i++)
		for (int j = 0; j < sample_num*UP; j++)
			fprintf(f_dump_upfs, "%f\n", dbg_array_cpu[i*y_len + j]);
#endif
}

void SpaceFilterGPU::set_noise_para(bool para0, float angle)
{
	noise_exist = para0;
	noise_offset = (int)(cos(angle * pi / 180.0f)*2.0f / 1500.0f*16000.0f * 3.0f + 0.5f);
#if CHENYU_DBG & 1
	printf("SpaceFilterGPU noise offset=%d\n", noise_offset);
#endif
}

void SpaceFilterGPU::cancel_noise()
{
	int len = (int)(sample_num * UP * 1.2);
	int linshi1 = (int)(sample_num * UP * 0.7) + 1;
	dim3 grid(1, channel_num-1);
	cudaDeviceSynchronize();
	if (noise_exist) {
		cuda_cancel_noise << <grid, 1024, 0, stream[0] >> >(&sig_3fs_array40_gpu[linshi1], y_len, noise_offset, sig_offset_array40_gpu, y_len, len);
		if (cudaGetLastError() != cudaSuccess)
			fprintf(stderr, "cuda_cancel_noise launch failed\n");
	}		
	else {
		cudaMemcpy2DAsync(sig_offset_array40_gpu, y_len*sizeof(float),
			&sig_3fs_array40_gpu[linshi1], y_len*sizeof(float),
			len*sizeof(float), channel_num, cudaMemcpyDeviceToDevice, stream[0]);
	}
	

#if CHENYU_DBG & 4
	cudaDeviceSynchronize();
	int rows = noise_exist ? channel_num - 1 : channel_num;
	cudaMemcpy(dbg_array_cpu, sig_offset_array40_gpu,
		y_len * rows * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < len; j++)
			fprintf(f_dump_offset, "%f\n", dbg_array_cpu[i*y_len + j]);
#endif
}

void SpaceFilterGPU::set_sf_para(short *delay)
{
	short indew_CBF_time[40 * 180];
	short indew_time_load_offset[40 * ANGLE_GROUP];
	unsigned short indew_time_load_size[40 * ANGLE_GROUP];

	for (int i = 0; i < channel_num; i++)
		for (int j = 0; j < 180; j++) 
		{
			indew_time_load_offset[i*ANGLE_GROUP + j / APT] = 
				delay[j / APT*APT * 40 + i + 1] - delay[j / APT*APT * 40 + i];
			indew_CBF_time[i * 180 + j] = delay[j * 40 + i] - delay[j / APT*APT * 40 + i];
			int j_1 = (j / APT *APT + APT - 1 >= 180) ? (180 - 1) : j / APT *APT + APT - 1;
			indew_time_load_size[i*ANGLE_GROUP + j / APT] = delay[j_1 * 40 + i] - delay[j / APT*APT * 40 + i] + 1024 * UP;
		}		

#if CHENYU_DBG & 128
	printf("printing load size\n");
	for (int i = 0; i < channel_num; i++) {
		for (int j = 0; j < ANGLE_GROUP; j++)
			printf("%5d,", indew_time_load_size[i*ANGLE_GROUP + j] - 1024 * UP);
		printf("  :%d\n", i);
	}		
	printf("printing load  offset\n");
	for (int i = 0; i < channel_num; i++) {
		for (int j = 0; j < ANGLE_GROUP; j++)
			printf("%5d,", indew_time_load_offset[i*ANGLE_GROUP + j]);
		printf("  :%d\n", i);
	}
	printf("printing load delay\n");
	for (int i = 0; i < channel_num; i++) {
		for (int j = 0; j < ANGLE_GROUP; j++)
			printf("%5d,", indew_CBF_time[i * 180 + j*APT + 1]);
		printf("  :%d\n", i);
	}
#endif

	cudaMemcpyToSymbol(indew_CBF_time_gpu, indew_CBF_time, sizeof(indew_CBF_time));
	cudaMemcpyToSymbol(indew_time_load_offset_gpu, indew_time_load_offset, sizeof(indew_time_load_offset));
	cudaMemcpyToSymbol(indew_time_load_size_gpu, indew_time_load_size, sizeof(indew_time_load_size));
}

void SpaceFilterGPU::space_filter()
{
	int linshi1 = (int)(16000 * 3 * 0.1f);
	dim3 grid(sample_num / 1024 + 1, ANGLE_GROUP);
	int smem_size = (APT <= 26) ? 24576 : 49152;
	if (noise_exist)
		cuda_space_filter << <grid, 1024, smem_size, stream[0] >> > (&sig_offset_array40_gpu[linshi1], y_len,
		sig_fs_2s_cbf_gpu+target_len-1, FFT_LEN, UP, 2000, channel_num-1);
	else
		cuda_space_filter << <grid, 1024, smem_size, stream[0] >> > (&sig_offset_array40_gpu[linshi1], y_len,
		sig_fs_2s_cbf_gpu+target_len-1, FFT_LEN, UP, 2000, channel_num);

	if (cudaGetLastError() != cudaSuccess)
		fprintf(stderr, "cuda_space_filter launch failed\n");

#if CHENYU_DBG & 8
	cudaDeviceSynchronize();	
	cudaMemcpy(dbg_array_cpu, sig_fs_2s_cbf_gpu,
		FFT_LEN * 180 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 180; i++)
		for (int j = 0; j < 2000; j++)
			fprintf(f_dump_cbf, "%f\n", dbg_array_cpu[i*FFT_LEN + target_len - 1 + j]);
#endif
}

void SpaceFilterGPU::set_target(const float * target, int len)
{
	float target_freq_cpu[FFT_LEN+2];
	target_len = len;
	if (len > FFT_LEN) {
		fprintf(stderr, "set_target len %d is bigger than FFT_LEN\n", len);
		return;
	}
	memset(target_freq_cpu, 0, sizeof(target_freq_cpu));
	memcpy(target_freq_cpu, target, len*sizeof(float));
	fft_cpu.RealFFT(target_freq_cpu, target_freq_cpu, FFT_LEN);
	int cudaStatus = cudaMemcpyToSymbol(target_freq_gpu, target_freq_cpu, sizeof(target_freq_cpu));
	if (cudaStatus!=cudaSuccess)
		fprintf(stderr, "cudaMemcpy device target_freq_gpu failed!");
}

void SpaceFilterGPU::convol()
{
	dim3 grid((FFT_LEN/2+1)/256, 180);
	int cudaStatus;

	cudaStatus = cufftExecR2C(fft_conv, sig_fs_2s_cbf_gpu, sig_freq_gpu);
	if (cudaStatus != CUFFT_SUCCESS)
		fprintf(stderr, "cufftExecR2C sig_fs_2s_cbf_gpu failed!");

	cuda_multiply << < grid, 256, 0, stream[0] >> > (sig_freq_gpu, FFT_LEN / 2 + 1, FFT_LEN / 2 + 1);

	cudaStatus = cufftExecC2R(fft_inv_conv, sig_freq_gpu, sig_conv_gpu);

	if (cudaStatus != CUFFT_SUCCESS)
		fprintf(stderr, "cufftExecC2R sig_fs_2s_cbf_gpu failed!");
	cudaStatus = cudaMemcpy2DAsync(sig_fs_2s_cbf_gpu, FFT_LEN*sizeof(float),
		sig_fs_2s_cbf_gpu + 2000, FFT_LEN*sizeof(float),
		(target_len - 1)*sizeof(float), 180, cudaMemcpyDeviceToDevice, stream[0]);
	if (cudaStatus != CUFFT_SUCCESS)
		fprintf(stderr, "cudaMemcpy2DAsync sig_fs_2s_cbf_gpu failed!");
#if CHENYU_DBG & 16
	cudaDeviceSynchronize();
	cudaMemcpy(dbg_array_cpu, sig_conv_gpu,
		FFT_LEN * 180 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 180; i++)
		for (int j = 0; j < 2000; j++)
			fprintf(f_dump_conv, "%d\n", (int) (dbg_array_cpu[i*FFT_LEN + target_len - 1 + j] /4096));
#endif
}

void SpaceFilterGPU::process(const vector<float *> pcm_in, vector<float *> pcm_out)
{
	upsample(pcm_in);
	cancel_noise();
	space_filter();
	convol();
#if 1	
	cudaMemcpy(dbg_array_cpu, sig_conv_gpu,
		FFT_LEN * 180 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 180; i++)
		for (int j = 0; j < 2000; j++)
			pcm_out[i][j] = dbg_array_cpu[i*FFT_LEN + target_len - 1 + j] / 4096;
#endif
}

SpaceFilterFreqGPU::SpaceFilterFreqGPU(int channel, int sample)
{
	channel_num = channel;
	sample_num = 16000;
	if (cufftPlan1d(&fft_analyze, sample_num, CUFFT_R2C, channel) != CUFFT_SUCCESS)
		fprintf(stderr, "CUFFT error: Plan fft conv creation failed");

	int cudaStatus = cudaMalloc((void**)&input_fft_gpu, (sample_num + 2) * channel_num * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc device input_fft_gpu failed!");

	cudaStatus = cudaMalloc((void**)&input_gpu, sample_num * channel_num * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc device input_gpu failed!");
	
	cudaStatus = cudaMalloc((void**)&output_cbf_gpu, 100 * 180 * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc device output_cbf_gpu failed!");

#if CHENYU_DBG
	cudaStatus = cudaMallocHost((void**)&dbg_array_cpu, (sample_num + 2) * channel_num * sizeof(float));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc host x failed!");
#endif

#if CHENYU_DBG & 32
	f_dump_fft = fopen("input_fft.txt", "wt");
	if (f_dump_fft == NULL)
		fprintf(stderr, "open gpu_upsample.txt failed\n");
#endif
#if CHENYU_DBG & 64
	f_dump_cbf = fopen("output_cbf.txt", "wt");
	if (f_dump_cbf == NULL)
		fprintf(stderr, "open gpu_upsample.txt failed\n");
#endif
}

SpaceFilterFreqGPU::~SpaceFilterFreqGPU()
{
	cudaFree(input_fft_gpu);
	cudaFree(input_gpu);
	cudaFree(output_cbf_gpu);
	cudaFreeHost(dbg_array_cpu);
}

void SpaceFilterFreqGPU::process(const float * pcm_in, float * pcm_out, int start_freq, int freq_num, float mic_d,
	bool cancel_noise_enable, float noise_angle, int cd)
{
	int cudaStatus;

	cudaMemcpy(input_gpu, pcm_in, sample_num*channel_num*sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cufftExecR2C(fft_analyze, input_gpu, input_fft_gpu);
	if (cudaStatus != CUFFT_SUCCESS)
		fprintf(stderr, "cufftExecR2C sig_fs_2s_cbf_gpu failed!");
		
#if CHENYU_DBG & 32
	cudaMemcpy(dbg_array_cpu, input_fft_gpu,
		(sample_num + 2) * channel_num * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < channel_num; i++)
		for (int j = start_freq*2; j < (start_freq+freq_num)*2; j++)
			fprintf(f_dump_fft, "%f\n", dbg_array_cpu[i*(sample_num + 2) + j]);
#endif

#if 0
	complex<float> a[100];
	float t_array;
	for (int i = 0; i < channel_num; i++) {
		a[i]._Val[0] = dbg_array_cpu[i*(sample_num + 2) + 100];
		a[i]._Val[1] = dbg_array_cpu[i*(sample_num + 2) + 101];
	}
	complex<float> energy[180];
	for (int angle = 0; angle < 180; angle++) {
		energy[angle] = 0;
		t_array = (4 * cos(pi* angle / 180) / 1500) * 2 * pi;
		for (int i = 0; i < channel_num; i++)
			energy[angle] += a[i] * exp(complex<float>(0,1)*t_array*(float)(50*i));
		printf("%f\n", norm(energy[angle]/(float)channel_num));
	}
#endif

	if (cancel_noise_enable) {	
		cuda_freq_cancel_noise << <1, freq_num >> > (input_fft_gpu, sample_num / 2 + 1, start_freq,
			noise_angle, channel_num, mic_d, cd);
		cuda_freq_space_filter << <freq_num, 180, channel_num*sizeof(cufftComplex), 0 >> > (
			input_fft_gpu, sample_num / 2 + 1, start_freq, output_cbf_gpu, 180, channel_num - cd, mic_d);
	} else
		cuda_freq_space_filter << <freq_num, 180, channel_num*sizeof(cufftComplex), 0 >> > (
		input_fft_gpu, sample_num / 2 + 1, start_freq, output_cbf_gpu, 180, channel_num, mic_d);
	if (cudaGetLastError() != cudaSuccess)
		fprintf(stderr, "cuda_freq_space_filter launch failed\n");

	cudaStatus = cudaMemcpy(pcm_out, output_cbf_gpu, freq_num * 180 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != CUFFT_SUCCESS)
		fprintf(stderr, "cufftExecR2C sig_fs_2s_cbf_gpu failed!");
#if CHENYU_DBG & 64
	for (int i = 0; i<freq_num; i++)
		for (int j = 0; j < 180; j++)
			fprintf(f_dump_cbf, "%f\n", pcm_out[i * 180 + j]);
#endif
}