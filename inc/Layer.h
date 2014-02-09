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

#ifndef INCLUDED_LAYER_H
#define INCLUDED_LAYER_H

#ifndef INCLUDED_MATRIX_H
#include "Matrix.h"
#endif

#ifndef INCLUDED_RANDOM
#include <random>
#define INCLUDED_RANDOM
#endif

//------------------------------------------------------------------------------

namespace rook { 

//------------------------------------------------------------------------------

template <size_t X, size_t Y>
struct Layer {
  constexpr static float learningRate     = 0.1;
  constexpr static float initialMean      = 0.0f;
  constexpr static float initialDeviation = 0.3f;

  typedef ColVector<X,      float> Input;
  typedef ColVector<Y,      float> Output;
  typedef    Matrix<Y, X+1, float> WeightMatrix;

  Layer() {
    static std::random_device device;
    static std::mt19937 generator(device());
    static std::normal_distribution<float> normal(initialMean, initialDeviation);

    // Randomly initialize weights with a normal distribution
    weightMatrix_ = weightMatrix_.apply([](size_t i, size_t j) {
      return normal(generator);
    });
  }

  // Sigmoid Activation
  float activation(float z) {
    return 1.0f/(1.0f + expf(-z));
  }

  // Sigmoid Derivative
  float derivative(float y) {
    return y*(1.0f - y);
  }

  // For inference, we take an input vector and 
  // calculate an output vector according to
  // our activation function 
  Output
  infer(Input const& input) const {
    // These temps should be optimized out
    const auto biased = aug1(input, 1.0f);
    const auto sum    = (weightMatrix_ * biased);

    // Apply our activate
    return sum.apply([&](float z) -> float {
      return activation(z); 
    });
  }

  // Function call semantics aren't a bad thing...
  Output
  operator()(Input const& input) const {
    return infer(input);
  }

  Input
  learn(Input  const& x, Output const& y, Output const& e) {
    for (auto i = 0; i < Y; i++) {
      for (auto j = 0; j < X; j++) {
        auto dE = e.at(i) * derivative(y.at(i)) * x.at(j);
        weightMatrix_.at(i,j) += dE * learningRate;
      }
    }

    return strip1(weightMatrix_.transpose() * e);
  }

  const WeightMatrix& 
  getWeights() const {
    return weightMatrix_;
  }

  void 
  print() const {
    std::cout << "Layer: |X| = " << X << ", |Y| = " << Y << std::endl;
    weightMatrix_.print("Weights");
  }

private:
  WeightMatrix   weightMatrix_;
};

//------------------------------------------------------------------------------

} // namespace rook

//------------------------------------------------------------------------------

#endif