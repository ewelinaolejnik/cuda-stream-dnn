#include "image_data.h"

#include <cstddef>
#include <random>

std::vector<float> generateImages(int image_count, int image_width,
                                  int image_height) {
  const std::size_t pixels_per_image =
      static_cast<std::size_t>(image_width) * image_height;
  std::vector<float> images(image_count * pixels_per_image);
  std::mt19937 generator(20261005);
  std::normal_distribution<float> noise(0.0f, 0.10f);

  for (int image_index = 0; image_index < image_count; ++image_index) {
    for (int y = 0; y < image_height; ++y) {
      for (int x = 0; x < image_width; ++x) {
        const float horizontal = static_cast<float>(x) / image_width;
        const float vertical = static_cast<float>(y) / image_height;
        const float checkerboard = ((x / 32 + y / 32 + image_index) % 2) ?
            0.16f : -0.16f;
        float value = 0.50f * horizontal + 0.35f * vertical + checkerboard +
                      noise(generator);
        value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
        images[static_cast<std::size_t>(image_index) * pixels_per_image +
               static_cast<std::size_t>(y) * image_width + x] = value;
      }
    }
  }
  return images;
}
