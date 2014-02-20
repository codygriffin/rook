//------------------------------------------------------------------------------
/*
*  
*  The MIT License (MIT)
* 
*  Copyright (C) 2014 Cody Griffin (cody.m.griffin@gmail.com)
* 
*  Permission is hereby granted, free of charge, to any person obtaining a copy
*  of this software and associated documentation files (the "Software"), to deal
*  in the Software without restriction, including without limitation the rights
*  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*  copies of the Software, and to permit persons to whom the Software is
*  furnished to do so, subject to the following conditions:
* 
*  The above copyright notice and this permission notice shall be included in all
*  copies or substantial portions of the Software.
* 
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*  SOFTWARE.
*/

#include "Autoencoder.h"

#ifdef GRAPHICS
#include <Magick++.h>
#endif

#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cstdint>
#include <tuple>
#include <vector>
#include <fstream>
#include <chrono>

//------------------------------------------------------------------------------
/*
 * This is not a test yet, per se.  Really just an example.  Perhaps we just put
 * a lower bound on the accepted test error on MNIST (say, 6%).  
 *
 * This is pretty much the "Hello World" of machine learning - MNIST digits using
 * sigmoid neurons and backprop.  
 *
 */

// Some byte swapping (MNIST data is big endian)
#define SWAP_UINT16(x) (((x) >> 8) | ((x) << 8))
#define SWAP_UINT32(x) (((x) >> 24) \
                     | (((x) & 0x00FF0000) >> 8) \
                     | (((x) & 0x0000FF00) << 8) \
                     |  ((x) << 24))

//------------------------------------------------------------------------------
// For some primitive performance analysis
template <typename Resolution = std::chrono::nanoseconds>
struct Stopwatch {
  typedef std::chrono::high_resolution_clock  Clock;
  static uint64_t clock(std::function<void (void)> func) {
    auto start = Clock::now();
    func();
    auto end  = Clock::now();
    return std::chrono::duration_cast<Resolution>(end - start).count();
  }
};

//------------------------------------------------------------------------------
// Helper class for loading data from the MNIST files
struct MnistData {
  typedef std::vector<uint8_t> Image;
  typedef uint8_t              Label;

  MnistData(const std::string& imageFile, const std::string& labelFile) {
    std::ifstream images, labels;
    images.open(imageFile, std::ios::binary);
    labels.open(labelFile, std::ios::binary);

    uint32_t magic, numLabels;
    images.read(reinterpret_cast<char*>(&magic),      sizeof(uint32_t));
    images.read(reinterpret_cast<char*>(&numImages_), sizeof(uint32_t));
    images.read(reinterpret_cast<char*>(&numRows_),   sizeof(uint32_t));
    images.read(reinterpret_cast<char*>(&numCols_),   sizeof(uint32_t));

    labels.read(reinterpret_cast<char*>(&magic),      sizeof(uint32_t));
    labels.read(reinterpret_cast<char*>(&numLabels),  sizeof(uint32_t));

    numImages_ = SWAP_UINT32(numImages_);
    numLabels  = SWAP_UINT32(numLabels);
    numRows_   = SWAP_UINT32(numRows_);
    numCols_   = SWAP_UINT32(numCols_);

    uint8_t             pixel, label;
    Image               image(numRows_ * numCols_);
    imageData_.resize(numImages_);
    for (int c = 0; c < numImages_; c++) { 
      for (int y = 0; y < numRows_; y++) { 
        for (int x = 0; x < numCols_; x++) { 
          images.read(reinterpret_cast<char*>(&pixel), sizeof(uint8_t));
          image[(y*numCols_ + x)] = pixel; 
        }
      }

      labels.read(reinterpret_cast<char*>(&label), sizeof(uint8_t));
      imageData_[c] = std::make_tuple(image,label);
    }
  }
  
  // Do something for each image and label
  void each(std::function<void (const Image&, const Label&)> f) { 
    for (auto image : imageData_) {
      f(std::get<0>(image), std::get<1>(image));
    }
  }
  
  uint32_t                                numRows_;
  uint32_t                                numCols_;
  uint32_t                                numImages_;
private:
  std::vector<std::tuple<Image, Label>>   imageData_;
};

//------------------------------------------------------------------------------

typedef rook::Autoencoder<784, 300> Encoder;

// Some helper functions for get MNIST into our net
// Note that dimensions must be the same - this is 
// big mismatch between compile-time and run-time 
// parameterization
Encoder::Input encodeImage(const MnistData::Image& image) {
  Encoder::Input input;
  for (int x = 0; x < image.size(); x++) { 
    input.at(x) = image[x]/255.0f; 
  }
  return input;
}

MnistData::Image decodeImage(const Encoder::Input& input) {
  MnistData::Image image;
  for (int x = 0; x < input.raw().size(); x++) { 
    image.push_back((uint8_t)floor(input.raw()[x] * 255.0f));
  }
  return image;
}

MnistData::Image decodeFilter(const Encoder::Input& input) {
  MnistData::Image image;
  for (int x = 0; x < input.raw().size(); x++) { 
    image.push_back((uint8_t)floor(((input.raw()[x] + 1.0f)/2.0f) * 255.0f));
  }
  return image;
}

//------------------------------------------------------------------------------

template <size_t N>
float mag(const rook::Matrix<N, 1>& vec) {
  float result = 0.0f;
  for (int i = 0; i < N; i++) {
    result += vec.at(i) * vec.at(i);
  }
  return sqrtf(result);
}

//------------------------------------------------------------------------------

int main (int argc, char ** argv) {
  Encoder encoder;

  // Load our MNIST data
  MnistData trainingData("data/train-images-idx3-ubyte",
                         "data/train-labels-idx1-ubyte");
  MnistData     testData("data/t10k-images-idx3-ubyte",
                         "data/t10k-labels-idx1-ubyte");

#ifdef GRAPHICS
  Magick::InitializeMagick(*argv);
#endif

  auto count = 0;
  trainingData.each([&](const MnistData::Image& image, const MnistData::Label& label) {
    const auto digit = encodeImage(image);
    const auto error = encoder.learn(digit, 0.01f);
    std::cout << "Error: " << mag(error) << std::endl;
  });

#ifdef GRAPHICS
  Magick::Montage montageSettings;
  std::vector<Magick::Image> digits;
  std::vector<Magick::Image> montage; 
#endif

  testData.each([&](const MnistData::Image& image, const MnistData::Label& label) {
    const auto digit           = encodeImage(image);
    const auto reconstruction  = encoder.reconstruct(digit);
    std::cout << "Error: " << mag(digit - reconstruction) << std::endl;
  
    #ifdef GRAPHICS
    // Sample our reconstructions
    if (count % 250 == 0) {
      auto blob  = Magick::Blob(); 
      auto blob2 = Magick::Blob(); 
      auto rawi  = decodeImage(digit);
      auto rawo  = decodeImage(reconstruction);
      auto img0  = Magick::Image();
      auto img1  = Magick::Image();

      blob.update(rawi.data(), rawi.size() * sizeof(float));
      blob2.update(rawo.data(), rawo.size() * sizeof(float));

      img0.magick("GRAY");
      img0.size("28x28");
      img0.depth(8);
      img0.read(blob);
      img1.magick("GRAY");
      img1.size("28x28");
      img1.depth(8);
      img1.read(blob2);

      digits.push_back(img0);
      digits.push_back(img1);
    }
    #endif

    count++;
  });

#ifdef GRAPHICS
  montageSettings.tile( "4x20" );
  montageSettings.geometry( "28x28+2+2" );
  Magick::montageImages(&montage, digits.begin(), digits.end(), montageSettings);
  Magick::writeImages(montage.begin(), montage.end(), "img/test.png" );

  auto weights = encoder.getWeightMatrix();
  std::vector<Magick::Image> filters;
  filters.resize(weights.cols);
  montage.clear();

  weights.eachCol([&](size_t j, const rook::ColVector<784>& col) {
    auto blob  = Magick::Blob(); 
    auto fmag  = mag(col);
    auto ncol  = col.apply([fmag](float a) {
      return a/fmag;
    });
    auto raw   = decodeFilter(col);

    blob.update(raw.data(), raw.size() * sizeof(float));
    filters[j].magick("GRAY");
    filters[j].size("28x28");
    filters[j].depth(8);
    filters[j].read(blob);
  });

  montageSettings.tile( "25x12" );
  Magick::montageImages(&montage, filters.begin(), filters.end(), montageSettings);
  Magick::writeImages(montage.begin(), montage.end(), "img/filters.png" );
#endif
}

//------------------------------------------------------------------------------
