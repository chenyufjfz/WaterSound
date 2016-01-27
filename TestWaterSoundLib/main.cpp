#include "WaterSndProcess.h"
#include <stdio.h>
#include <time.h>

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
	getchar();
	return 0;
}