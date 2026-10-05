#ifndef CUDA_STREAM_DNN_INCLUDE_OPTIONS_H_
#define CUDA_STREAM_DNN_INCLUDE_OPTIONS_H_

struct Options {
  int image_count = 256;
  int image_width = 512;
  int image_height = 512;
  int stream_count = 4;
};

Options parseOptions(int argument_count, char** arguments);

#endif  // CUDA_STREAM_DNN_INCLUDE_OPTIONS_H_
