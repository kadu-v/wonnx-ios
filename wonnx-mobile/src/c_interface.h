#include <stdint.h>

typedef struct
{
    float *data;
    int len;
    float preprocess_time;
    float inference_time;
    float post_process_time;
} Array;

int load_model(
    char *model_path,
    uint32_t model_path_len,
    uint32_t input_batch_size,
    uint32_t input_channels,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t output_channels,
    uint32_t output_height,
    uint32_t output_width);

Array predict(float *data, uint32_t len);
