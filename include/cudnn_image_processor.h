#ifndef CUDA_STREAM_DNN_INCLUDE_CUDNN_IMAGE_PROCESSOR_H_
#define CUDA_STREAM_DNN_INCLUDE_CUDNN_IMAGE_PROCESSOR_H_

#include <string>
#include <vector>

#include "options.h"

struct ProcessingResult {
  // Host-side copy of every image produced by cudnnConvolutionForward.
  std::vector<float> convolution_output_images;
  float elapsed_milliseconds = 0.0f;
  std::string gpu_name;
  std::string cudnn_version;
};

ProcessingResult processImagesWithCudnn(const std::vector<float>& input_images,
                                        const Options& options);

#endif  // CUDA_STREAM_DNN_INCLUDE_CUDNN_IMAGE_PROCESSOR_H_
