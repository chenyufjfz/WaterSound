#ifndef _WATERSND_PROCESS_h
#define _WATERSND_PROCESS_h
#include <vector>
void generate_pcm_in(int channel, int sample, std::vector<float> &input);
void * init_watersound_processContext(bool gpu);
void watersound_process(void * context, float * pcm_input, float * pcm_output);
void watersound_set_target(void * context, float * target, int target_len);
#endif