#include "WaterSndProcess.h"
#include <stdio.h>
#include <time.h>

#define pi 3.1415926f
float input1[40 * 16000], input2[40*16000], output1[91*180], output2[3*180];

int main()
{
#define target_len 100
	void * space_filter_gpu = init_watersound_processContext(true);
	void * space_filter_cpu = init_watersound_processContext(false);
	float target[target_len];
	for (int i = 0; i < target_len; i++)
		target[i] = rand() % 16;
	watersound_set_target(space_filter_cpu, target, target_len);
	watersound_set_target(space_filter_gpu, target, target_len);
	std::vector <float> input;
	std::vector <float> output(180 * 2000);
	generate_pcm_in(40, 16000, input);
	clock_t t0 = clock();
	watersound_process(space_filter_gpu, &input[0], &output[0]);
	watersound_process(space_filter_gpu, &input[0], &output[0]);
	clock_t t1 = clock();
	watersound_process(space_filter_cpu, &input[0], &output[0]);
	watersound_process(space_filter_cpu, &input[0], &output[0]);
	clock_t t2 = clock();
	printf("finished, gpu time=%f, cpu time=%f\n", (double)(t1 - t0) / CLOCKS_PER_SEC, (double)(t2 - t1) / CLOCKS_PER_SEC);

	void * space_filter_freq_gpu = init_watersound_freq_processContext();
	watersound_freq_set_noise_para(space_filter_freq_gpu, true, 30, true, 30);
	float t1_yanci, t2_yanci;
	t1_yanci = -cos(30 * pi / 180) * 4 / 1500;
	t2_yanci = -cos(70 * pi / 180) * 4 / 1500;

	for (int i = 0; i < 40; i++)
		for (int j = 0; j < 16000; j++)
			input1[i * 16000 + j] = cos(2 * pi * 50 * (j + 1 + (i + 1)*t1_yanci * 16000) / 16000) * 10 +
			cos(2 * pi * 50 * (j + 1 + (i + 1)*t2_yanci * 16000) / 16000);

	t1_yanci = -cos(30 * pi / 180) * 2 / 1500;
	t2_yanci = -cos(70 * pi / 180) * 2 / 1500;
	for (int i = 0; i < 40; i++)
		for (int j = 0; j < 16000; j++)
			input2[i * 16000 + j] = cos(2 * pi * 400 * (j + 1 + (i + 1)*t1_yanci * 16000) / 16000) * 10 +
			cos(2 * pi * 400 * (j + 1 + (i + 1)*t2_yanci * 16000) / 16000);
	t0 = clock();
	watersound_freq_process(space_filter_freq_gpu, &input1[0], &input2[0], &output1[0], &output2[0],399,3);
	t1 = clock();
	printf("finished, gpu time=%f\n", (double)(t1 - t0) / CLOCKS_PER_SEC);
	getchar();
	return 0;
}