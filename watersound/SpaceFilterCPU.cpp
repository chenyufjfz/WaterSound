#include "SpaceFilter.h"
#include "math.h"
#include <string.h>
/*
  Copy from Bao xizhong
*/
float coef_filter_GPU[256] = { 0.0007f, 0.0007f, 0.0007f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0009f, 0.0009f, 0.0009f, 0.0009f, 0.0010f, 0.0010f, 0.0010f, 0.0011f,
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
0.0010f, 0.0010f, 0.0010f, 0.0009f, 0.0009f, 0.0009f, 0.0009f, 0.0008f, 0.0008f, 0.0008f, 0.0008f, 0.0007f, 0.0007f, 0.0007f };

SpaceFilterCPU::SpaceFilterCPU(int channel, int sample) : SpaceFilter(channel, sample)
{
	rise_fs = 3;
	target_len = 1;
	memset(sig_3fs_array40_GPU, 0, sizeof(sig_3fs_array40_GPU));
	memset(sig_3fs_OK_array40_GPU, 0, sizeof(sig_3fs_OK_array40_GPU));
	memset(sig_3fs_2s_CBF_GPU, 0, sizeof(sig_3fs_2s_CBF_GPU));
	set_noise_para(true, 15);

#if CHENYU_DBG & 2
	f_dump_upfs = fopen("cpu_upsample.txt", "wt");
	if (f_dump_upfs == NULL)
		fprintf(stderr, "open cpu_upsample.txt failed\n");
#endif
#if CHENYU_DBG & 4
	f_dump_offset = fopen("cpu_cancel_noise.txt", "wt");
	if (f_dump_offset == NULL)
		fprintf(stderr, "open cpu_cancel_noise.txt failed\n");
#endif
#if CHENYU_DBG & 8
	f_dump_cbf = fopen("cpu_space_filter.txt", "wt");
	if (f_dump_cbf == NULL)
		fprintf(stderr, "open cpu_space_filter.txt failed\n");
#endif
#if CHENYU_DBG & 16
	f_dump_conv = fopen("cpu_conv.txt", "wt");
	if (f_dump_conv == NULL)
		fprintf(stderr, "open cpu_conv.txt failed\n");
#endif
}

SpaceFilterCPU::~SpaceFilterCPU()
{

}

void SpaceFilterCPU::upsample(const vector<float *> pcm_in)
{
	for (int n_array = 0; n_array<channel_num; n_array++)
	{
		memcpy(sig_3fs_array40_GPU[n_array], sig_3fs_array40_GPU[n_array] + 16000 * 3, sizeof(float)*(255));
	}

	for (int n_array = 0; n_array<channel_num; n_array++)
	{
		for (int n_sig = 0; n_sig<16000; n_sig++)
		{
			sig_3fs_array40_GPU[n_array][255 + n_sig * 3] = pcm_in[n_array][n_sig];
		}
	}


	for (int n_array = 0; n_array<channel_num; n_array++)
	{
		memcpy(sig_3fs_OK_array40_GPU[n_array], sig_3fs_OK_array40_GPU[n_array] + 16000 * 3, sizeof(float)*(16000 * 3));
	}

	for (int n_array = 0; n_array<channel_num; n_array++)
	{
		for (int n_vector1 = 0; n_vector1<16000 * 3; n_vector1++)
		{
			float out_sig = 0.0f;
			for (int n_vector2 = 0; n_vector2<256; n_vector2++)
			{
				out_sig = out_sig + sig_3fs_array40_GPU[n_array][n_vector1 + n_vector2] * coef_filter_GPU[n_vector2];
			}
			sig_3fs_OK_array40_GPU[n_array][16000 * 3 + n_vector1] = out_sig;
		}
	}
#if CHENYU_DBG & 2
	for (int i = 0; i < channel_num; i++)
		for (int j = 0; j < sample_num*rise_fs; j++)
			fprintf(f_dump_upfs, "%f\n", sig_3fs_OK_array40_GPU[i][16000 * 3 + j]);
#endif
}

void SpaceFilterCPU::set_noise_para(bool para0, float angle)
{
	noise_exist = para0;
	noise_angle = angle;
	int num_offset;
	num_offset = (int)(cos(noise_angle * pi / 180.0f)*2.0f / 1500.0f*16000.0f * 3.0f + 0.5f);
#if CHENYU_DBG & 1
	printf("set SpaceFilterCPU noise offset=%d\n", num_offset);
#endif
}

void SpaceFilterCPU::cancel_noise()
{
	if (noise_exist)
	{
		int num_offset;
		num_offset = (int)(cos(noise_angle * pi / 180.0f)*2.0f / 1500.0f*16000.0f * 3.0f + 0.5f);
		for (int n_array = 0; n_array<channel_num-1; n_array++)
		{
			int linshi1 = (int) (16000 * 3 * 0.7);
			for (int n_vector1 = 0; n_vector1<(int)(16000 * 3 * 1.2); n_vector1++)
			{
				sig_offset_array40_GPU[n_array][n_vector1] = sig_3fs_OK_array40_GPU[n_array][n_vector1 + linshi1] - sig_3fs_OK_array40_GPU[n_array+1][n_vector1 + linshi1 + num_offset];
			}
		}

	}
	else
	{
		for (int n_array = 0; n_array<channel_num; n_array++)
		{
			memcpy(sig_offset_array40_GPU[n_array], sig_3fs_OK_array40_GPU[n_array] + (int)(16000 * 3 * 0.7), sizeof(float)*(int)(16000 * 3 * 1.2));
		}

	}
#if CHENYU_DBG & 4
	int rows = noise_exist ? channel_num - 1 : channel_num;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < (int)(16000 * 3 * 1.2); j++)
			fprintf(f_dump_offset, "%f\n", sig_offset_array40_GPU[i][j]);
#endif
}

void SpaceFilterCPU::set_sf_para(short *delay)
{
	for (int i = 0; i < 180; i++)
		for (int j = 0; j < channel_num; j++)
			indew_CBF_time[i][j] = delay[i*40 + j];
}

void SpaceFilterCPU::space_filter()
{
	int row = noise_exist ? (channel_num - 1) : channel_num;

	int linshi1 = (int) (16000 * 3 * 0.1f);
	int linshi2 = 0;

	for (int n_vector1 = 0; n_vector1 < 180; n_vector1++)
	{
		float linshi3[16000] = { 0 };
		for (int n_vector2 = 0; n_vector2 < row; n_vector2++)
		{
			linshi2 = linshi1 + indew_CBF_time[n_vector1][n_vector2];

			for (int n_vector3 = 0; n_vector3 < 16000; n_vector3++)
			{
				linshi3[n_vector3] = linshi3[n_vector3] + sig_offset_array40_GPU[n_vector2][linshi2 + n_vector3 * 3];
			}
		}

		{
			float linshi4 = 1.0f / row;
			for (int n_vector3 = 0; n_vector3 < 2000; n_vector3++)
			{
				linshi3[n_vector3] = linshi3[n_vector3*8] * linshi4;
			}

		}
		memcpy(sig_3fs_2s_CBF_GPU[n_vector1], sig_3fs_2s_CBF_GPU[n_vector1] + 2000, sizeof(float)*(2000));
		memcpy(sig_3fs_2s_CBF_GPU[n_vector1] + 2000, linshi3, sizeof(float)*(2000));
	}	

#if CHENYU_DBG & 8	
	for (int i = 0; i < 180; i++)
		for (int j = 0; j < 2000; j++)
			fprintf(f_dump_cbf, "%f\n", sig_3fs_2s_CBF_GPU[i][j + 2000]);
#endif
}

void SpaceFilterCPU::set_target(const float * target, int len)
{
	target_len = len;
	if (len > FFT_LEN) {
		fprintf(stderr, "set_target len %d is bigger than FFT_LEN\n", len);
		return;
	}
	memset(target_freq_cpu, 0, sizeof(target_freq_cpu));
	memcpy(target_freq_cpu, target, len*sizeof(float));
	fft_cpu.RealFFT(target_freq_cpu, target_freq_cpu, FFT_LEN);
}

void SpaceFilterCPU::convol()
{
	float freq[FFT_LEN + 2];
	for (int i = 0; i < 180; i++) {
		fft_cpu.RealFFT(sig_3fs_2s_CBF_GPU[i] + 2001 - target_len, freq, FFT_LEN);

		for (int j = 0; j <= FFT_LEN / 2; j++) {
			float real = freq[j * 2] * target_freq_cpu[j * 2] - freq[j * 2 + 1] * target_freq_cpu[j * 2 + 1];
			float imag = freq[j * 2] * target_freq_cpu[j * 2 + 1] + freq[j * 2 + 1] * target_freq_cpu[j * 2];
			freq[j * 2] = real;
			freq[j * 2 + 1] = imag;
		}

		fft_cpu.RealIFFT(freq, freq, FFT_LEN);

		memcpy(sig_3fs_2s_CBF_matching_GPU[i], freq + target_len - 1, 2000*sizeof(float));
	}

#if CHENYU_DBG & 16
	cudaDeviceSynchronize();	
	for (int i = 0; i < 180; i++)
		for (int j = 0; j < 2000; j++)
			fprintf(f_dump_conv, "%d\n", (int) sig_3fs_2s_CBF_matching_GPU[i][j]);
#endif
}

void SpaceFilterCPU::process(const vector<float *> pcm_in, vector<float *> pcm_out)
{
	upsample(pcm_in);
	cancel_noise();
	space_filter();
	convol();
#if 1
	for (int i = 0; i < 180; i++)
		for (int j = 0; j < 2000; j++)
			pcm_out[i][j] = sig_3fs_2s_CBF_matching_GPU[i][j];
#endif
}