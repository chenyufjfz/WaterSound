#ifndef _SPACE_FILTER_H_
#define _SPACE_FILTER_H_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include "cufft.h"
using namespace std;

//Now CHENYU_DBG shouldn't be 0, because I use dbg_array_cpu for copy
#define CHENYU_DBG 0xff61
#define CHANNEL_NUM 40
#define FFT_LEN 4096
#define pi 3.1415926535f
#define TWOPI   6.283185307179586f

class FFTCPU {

protected:
	unsigned long   FFTSwapInitialised;
	unsigned long   FFTLog2N;
	unsigned long * FFTButter;
	unsigned long * FFTBitSwap;
	float         * FFTPhi;

public:
	FFTCPU()
	{
		FFTSwapInitialised = 0;
	}

	~FFTCPU()
	{
		FFTFree();
	}

	void FFTInit(unsigned long N)
	{
		unsigned long   C, L, K;
		float           Theta;
		float         * PFFTPhi;

		if ((FFTSwapInitialised != N) && (FFTSwapInitialised != 0))
			FFTFree();

		if (FFTSwapInitialised == N)
		{
			return;
		}
		else
		{
			C = N;
			for (FFTLog2N = 0; C > 1; C >>= 1)
				FFTLog2N++;

			C = 1;
			C <<= FFTLog2N;
			if (N == C)
				FFTSwapInitialised = N;

			FFTButter = (unsigned long *)malloc(sizeof(unsigned long) * (N >> 1));
			FFTBitSwap = (unsigned long *)malloc(sizeof(unsigned long) * N);
			FFTPhi = (float *)malloc(2 * sizeof(float) * (N >> 1));

			PFFTPhi = FFTPhi;
			for (C = 0; C < (N >> 1); C++)
			{
				Theta = (TWOPI * C) / N;
				(*(PFFTPhi++)) = (float)cos(Theta);
				(*(PFFTPhi++)) = (float)sin(Theta);
			}

			FFTButter[0] = 0;
			L = 1;
			K = N >> 2;
			while (K >= 1)
			{
				for (C = 0; C < L; C++)
					FFTButter[C + L] = FFTButter[C] + K;
				L <<= 1;
				K >>= 1;
			}
		}
	}

	void FFTFree(void)
	{
		if (FFTSwapInitialised != 0)
		{
			free(FFTButter);
			free(FFTBitSwap);
			free(FFTPhi);
			FFTSwapInitialised = 0;
		}
	}

	void FFT(float * x, unsigned long N)
	{
		unsigned long   Cycle, C, S, NC;
		unsigned long   Step = N >> 1;
		unsigned long   K1, K2;
		register float  R1, I1, R2, I2;
		float           ReFFTPhi, ImFFTPhi;

		if (N > 1)
		{
			FFTInit(N);

			for (Cycle = 1; Cycle < N; Cycle <<= 1, Step >>= 1)
			{
				K1 = 0;
				K2 = Step << 1;

				for (C = 0; C < Cycle; C++)
				{
					NC = FFTButter[C] << 1;
					ReFFTPhi = FFTPhi[NC];
					ImFFTPhi = FFTPhi[NC + 1];
					for (S = 0; S < Step; S++)
					{
						R1 = x[K1];
						I1 = x[K1 + 1];
						R2 = x[K2];
						I2 = x[K2 + 1];

						x[K1++] = R1 + ReFFTPhi * R2 + ImFFTPhi * I2;
						x[K1++] = I1 - ImFFTPhi * R2 + ReFFTPhi * I2;
						x[K2++] = R1 - ReFFTPhi * R2 - ImFFTPhi * I2;
						x[K2++] = I1 + ImFFTPhi * R2 - ReFFTPhi * I2;
					}
					K1 = K2;
					K2 = K1 + (Step << 1);
				}
			}

			NC = N >> 1;
			for (C = 0; C < NC; C++)
			{
				FFTBitSwap[C] = FFTButter[C] << 1;
				FFTBitSwap[C + NC] = 1 + FFTBitSwap[C];
			}
			for (C = 0; C < N; C++)
				if ((S = FFTBitSwap[C]) != C)
				{
					FFTBitSwap[S] = S;
					K1 = C << 1;
					K2 = S << 1;
					R1 = x[K1];
					x[K1++] = x[K2];
					x[K2++] = R1;
					R1 = x[K1];
					x[K1] = x[K2];
					x[K2] = R1;
				}
		}
	}

	void IFFT(float * x, unsigned long N)
	{
		unsigned long   Cycle, C, S, NC;
		unsigned long   Step = N >> 1;
		unsigned long   K1, K2;
		register float  R1, I1, R2, I2;
		float           ReFFTPhi, ImFFTPhi;

		if (N > 1)
		{
			FFTInit(N);

			for (Cycle = 1; Cycle < N; Cycle <<= 1, Step >>= 1)
			{
				K1 = 0;
				K2 = Step << 1;

				for (C = 0; C < Cycle; C++)
				{
					NC = FFTButter[C] << 1;
					ReFFTPhi = FFTPhi[NC];
					ImFFTPhi = FFTPhi[NC + 1];
					for (S = 0; S < Step; S++)
					{
						R1 = x[K1];
						I1 = x[K1 + 1];
						R2 = x[K2];
						I2 = x[K2 + 1];

						x[K1++] = R1 + ReFFTPhi * R2 - ImFFTPhi * I2;
						x[K1++] = I1 + ImFFTPhi * R2 + ReFFTPhi * I2;
						x[K2++] = R1 - ReFFTPhi * R2 + ImFFTPhi * I2;
						x[K2++] = I1 - ImFFTPhi * R2 - ReFFTPhi * I2;
					}
					K1 = K2;
					K2 = K1 + (Step << 1);
				}
			}

			NC = N >> 1;
			for (C = 0; C < NC; C++)
			{
				FFTBitSwap[C] = FFTButter[C] << 1;
				FFTBitSwap[C + NC] = 1 + FFTBitSwap[C];
			}
			for (C = 0; C < N; C++)
				if ((S = FFTBitSwap[C]) != C)
				{
					FFTBitSwap[S] = S;
					K1 = C << 1;
					K2 = S << 1;
					R1 = x[K1];
					x[K1++] = x[K2];
					x[K2++] = R1;
					R1 = x[K1];
					x[K1] = x[K2];
					x[K2] = R1;
				}

			NC = N << 1;
			for (C = 0; C < NC;)
				x[C++] /= N;
		}
	}

	/*
	input: x from 0..N-1,
	input: N N=2^M
	output: f from 0..N+1, f[0]
	x and f can be same.
	*/
	void RealFFT(const float *x, float * f, unsigned long N)
	{
		float            *y;
		unsigned long    i;

		y = (float *)malloc(2 * N * sizeof(float));

		for (i = 0; i < N; i++) {
			y[2 * i] = x[i];
			y[2 * i + 1] = 0.0f;
		}

		FFT(y, N);

		for (i = 0; i <= N / 2; i++) {
			f[2 * i] = y[2 * i];
			f[2 * i + 1] = y[2 * i + 1];
		}

		free(y);
	}

	/*
	input: f from 0..N+1,
	input: N N=2^M
	output: x from 0..N-1, f[0]
	x and f can be same.
	*/
	void RealIFFT(const float *f, float *x, unsigned long N)
	{

		float            *y;
		unsigned long    i;

		y = (float *)malloc(2 * N * sizeof(float));

		for (i = 0; i <= N / 2; i++) {
			y[2 * i] = f[2 * i];
			y[2 * i + 1] = f[2 * i + 1];
		}
		for (i = N / 2 + 1; i < N; i++) {
			int j = N - i;
			y[2 * i] = f[2 * j];
			y[2 * i + 1] = -f[2 * j + 1];
		}

		IFFT(y, N);

		for (i = 0; i < N; i++) {
			x[i] = y[2 * i];
		}

		free(y);
	}
};

class SpaceFilter {
protected:
	int channel_num, sample_num;	
public:
	SpaceFilter(int channel, int sample) { 
		channel_num = channel; 
		sample_num = 16000;
	}
	virtual ~SpaceFilter() {}
	virtual void set_noise_para(bool para0, float angle) = 0;
	virtual void set_target(const float * target, int len) = 0;
	virtual void set_sf_para(short *delay) = 0;
	virtual void process(const vector<float *> pcm_in, vector<float *> pcm_out) = 0;
};

class SpaceFilterGPU : public SpaceFilter
{
protected:
	int x_len, seg_len, y_len, h_len, target_len;
	float * sig_array40_gpu;
	float * sig_3fs_array40_gpu;
	float * sig_offset_array40_gpu;
	float * sig_fs_2s_cbf_gpu;
	cufftComplex * sig_freq_gpu;
	cufftReal * sig_conv_gpu;
	float * sig_array40_cpu;
	float * dbg_array_cpu;
	cudaStream_t * stream;
	bool noise_exist;
	int noise_offset;
	FFTCPU fft_cpu;
	cufftHandle fft_conv, fft_inv_conv;
#if CHENYU_DBG & 2
	FILE *f_dump_upfs;
#endif
#if CHENYU_DBG & 4
	FILE *f_dump_offset;
#endif
#if CHENYU_DBG & 8
	FILE *f_dump_cbf;
#endif
#if CHENYU_DBG & 16
	FILE * f_dump_conv;
#endif
public:
	SpaceFilterGPU(int channel = 40, int sample = 16000);
	~SpaceFilterGPU();
	void upsample(const vector<float *> pcm_in);
	void set_noise_para(bool para0, float angle);	
	void set_target(const float * target, int len);
	void cancel_noise();
	void set_sf_para(short *delay);
	void space_filter();
	void convol();
	void process(const vector<float *> pcm_in, vector<float *> pcm_out);
};

class SpaceFilterFreqGPU {
protected:
	int channel_num, sample_num;
	float * input_gpu;
	cufftComplex * input_fft_gpu;
	float * output_cbf_gpu;
	float * dbg_array_cpu;
	cufftHandle fft_analyze;
#if CHENYU_DBG & 32
	FILE * f_dump_fft;
#endif
#if CHENYU_DBG & 64
	FILE * f_dump_cbf;
#endif
public:
	SpaceFilterFreqGPU(int channel, int sample=16000);
	~SpaceFilterFreqGPU();
	void process(const float * pcm_in, float * pcm_out, int start_freq, int freq_num, float mic_d);
};

class SpaceFilterCPU : public SpaceFilter
{
protected:	
	int rise_fs;	
	int target_len;
	float sig_matching_C_GPU[16000];
	float sig_3fs_array40_GPU[40][16000 * 3 + 255];   
	float sig_3fs_OK_array40_GPU[40][16000 * 3 * 2];  
	float sig_offset_array40_GPU[40][16000 * 3 * 2];  
	int indew_CBF_time[180][40];        

	float sig_3fs_2s_CBF_GPU[180][FFT_LEN +FFT_LEN/2]; 
	float sig_3fs_2s_CBF_matching_GPU[180][2000]; 
	float target_freq_cpu[FFT_LEN + 2];
	bool noise_exist; 
	float noise_angle;
	FFTCPU fft_cpu;
#if CHENYU_DBG & 2
	FILE *f_dump_upfs;
#endif
#if CHENYU_DBG & 4
	FILE *f_dump_offset;
#endif
#if CHENYU_DBG & 8
	FILE *f_dump_cbf;
#endif
#if CHENYU_DBG & 16
	FILE * f_dump_conv;
#endif
public:
	SpaceFilterCPU(int channel = 40, int sample = 16000);
	~SpaceFilterCPU();
	void upsample(const vector<float *> pcm_in);
	void set_noise_para(bool para0, float angle);
	void set_target(const float * target, int len);
	void cancel_noise();
	void set_sf_para(short *delay);
	void space_filter();
	void convol();
	void process(const vector<float *> pcm_in, vector<float *> pcm_out);
};

void cuda_fir(int up, int m, dim3 grid, int thread_num, int smem_size, cudaStream_t * stream, float * x, float * y,
	int x_len, int seg_len, int h_len, int y_len);
#endif