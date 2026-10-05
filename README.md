# CUDA Streams cuDNN Image Processor

## Overview

This project performs GPU image processing with NVIDIA cuDNN and CUDA streams.
It generates **256 noisy grayscale images** of **512 x 512 pixels** by default:
67,108,864 pixels total. This satisfies the course requirement to process
hundreds of small inputs while remaining fully reproducible without a download.

The GPU operation is a single cuDNN forward-convolution call for each batch. A
fixed 3 x 3 Gaussian kernel smooths noise in each image. Four non-blocking CUDA
streams asynchronously queue this pipeline for separate batches:

`host-to-device copy -> cuDNN convolution -> device-to-host copy`

cuDNN executes the convolution on the NVIDIA GPU; no CPU-only fallback exists.

## Code organization

`bin/` receives the compiled executable. It is empty in source control.

`data/` holds run outputs. `data/results/` is deliberately tracked so its PGM
images can be committed as proof of GPU execution.

`include/` contains small public interfaces for options, image generation, and
cuDNN processing.

`lib/` is reserved for project-local libraries. Install cuDNN system-wide or
pass `CUDNN_ROOT` to CMake instead of copying it into the repository.

`src/` contains the application entry point, command-line options, deterministic
image generator, and cuDNN/CUDA-stream pipeline.

`INSTALL` explains cuDNN installation and CMake setup. `run.sh` runs the built
executable and forwards optional workload parameters.

## Build and run

cuDNN is required in addition to the CUDA Toolkit. See `INSTALL` for setup.
For Windows, use Git Bash or WSL for `run.sh`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDNN_ROOT="C:/path/to/cudnn"
cmake --build build --config Release
bash run.sh
```

Quick smoke test:

```bash
bash run.sh --images 16 --width 256 --height 256 --streams 2
```

## Results and submission evidence

Every successful run writes the following hardcoded outputs:

- `data/results/input_0000.pgm` - a generated noisy input image.
- `data/results/cudnn_convolution_result_0000.pgm` - the cuDNN convolution
  result for the corresponding input image.

`ProcessingResult::convolution_output_images` holds the results for all 256
images after `cudnnConvolutionForward` completes. To keep the repository small,
the program saves image 0 as the committed visual proof instead of writing every
image in the batch.

`data/results/` is not ignored by Git. Commit these images and a terminal log
captured with `bash run.sh | Tee-Object execution-log.txt` as proof of execution.

## Short submission description

CUDA Streams cuDNN Image Processor generates hundreds of noisy grayscale images
and denoises them on an NVIDIA GPU using cuDNN's convolution API with a 3 x 3
Gaussian filter. Four non-blocking CUDA streams pipeline transfers and cuDNN GPU
work across image batches. It is a GPU-library implementation, not a CPU-only
or CPU-multithreaded solution.
