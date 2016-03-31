#ifndef _WATERSND_PROCESS_h
#define _WATERSND_PROCESS_h
#include <vector>
void generate_pcm_in(int channel, int sample, std::vector<float> &input);
void * init_watersound_processContext(bool gpu);
void watersound_set_noise_para(void * context, bool enable, float angle);
void watersound_process(void * context, float * pcm_input, float * pcm_output);
void watersound_set_target(void * context, float * target, int target_len);
void * init_watersound_freq_processContext();
void watersound_freq_set_noise_para(void *context, bool enable1, float angle1, bool enable2, float angle2);
void watersound_freq_process(void * context, float * input1, float * input2, float * output1, float * output2, int start_freq, int freq_num);
#endif