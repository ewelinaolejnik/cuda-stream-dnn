#ifndef CUDA_STREAM_DNN_INCLUDE_IMAGE_DATA_H_
#define CUDA_STREAM_DNN_INCLUDE_IMAGE_DATA_H_

#include <vector>

std::vector<float> generateImages(int image_count, int image_width,
                                  int image_height);

#endif  // CUDA_STREAM_DNN_INCLUDE_IMAGE_DATA_H_
