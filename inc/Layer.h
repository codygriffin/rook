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

//------------------------------------------------------------------------------

namespace rook { 

//------------------------------------------------------------------------------
// Activations
struct Sigmoid  {
  static float activation(float z) {
    return 1.0f/(1.0f + expf(-z));
  }

  static float derivative(float y) {
    return y*(1.0f - y);
  }
};

struct Linear  {
  static float activation(float z) {
    return z;
  }

  static float derivative(float y) {
    return 1.0f;
  }
};

struct Sinc  {
  static float activation(float z) {
    return abs(z)<1.0e-10f ? 1.0f : sin(z)/z;
  }

  static float derivative(float y) {
    return abs(y)<1.0e-10f ? 0.0f : (cos(y)/y) - (sin(y)/(y*y));
  }
};

struct Hinge  {
  static float activation(float z) {
    return std::max(0.0f, z);
  }

  static float derivative(float y) {
    return y > 0.0f ? 1.0f : 0.0f;
  }
};

struct Softmax {
  static float activation(float z);
  static float derivative(float y);
};

//------------------------------------------------------------------------------
// Loss
struct Error {
  template <typename O>
  static O error(const O& y, const O& t) {
    return (t - y).apply([](float x) { return 0.5f*x*x; });
  };
  template <typename O>
  static O derivative(const O& y, const O& t) {
    return t - y;
  };
};

//------------------------------------------------------------------------------
// Regularization

//------------------------------------------------------------------------------

template <size_t X, size_t Y, typename Activation = Sigmoid, typename Loss = Error> 
struct Layer {
  constexpr static float initialMean      = 0.0f;
  constexpr static float initialDeviation = 0.3f;

  // typedefs for our input and output vectors
  typedef ColVector<X, float> Input;
  typedef ColVector<Y, float> Output;

  // typedef for our weight matrix
  typedef    Matrix<Y, X, float> WeightMatrix;
  typedef ColVector<   Y, float> Bias;

  Layer() 
  : weightMatrix_ (WeightMatrix(normal(initialMean, initialDeviation)))
  , bias_         (        Bias(normal(initialMean, initialDeviation)))
  {}

  // Set the weights explicitly
  Layer(const WeightMatrix& weightMatrix, const Bias& bias)
  : weightMatrix_ (weightMatrix) 
  , bias_ (bias) {}

  // Set the weights explicitly
  Layer(const WeightMatrix& weightMatrix)
  : weightMatrix_ (weightMatrix) 
  , bias_         (Bias(normal(initialMean, initialDeviation)))
  {}

  // For inference, we take an input vector and 
  // calculate an output vector according to
  // our activation function 
  Output
  infer(Input const& input) const {
    // These temps should be optimized out
    //const auto biased = aug(input, 1.0f); 
    const auto sum    = (weightMatrix_ * input) + bias_;

    // Apply our activation function to each
    // output
    return sum.apply([&](float z) -> float {
      return Activation::activation(z); 
    });
  }

  std::tuple<Input, Output>
  learn(Input  const& x, Output const& y, Output const& t, float learningRate = 0.1f) {
    // For each output 
    const auto dError = Loss::derivative(y, t);
    for (auto i = 0; i < Y; i++) {
      // Look at each input from the previous layer
      for (auto j = 0; j < X; j++) {
        // Calculate the partial deriviate of the error with
        // respect to the weight
        auto dWeight = dError.at(i) * Activation::derivative(y.at(i)) * x.at(j);

        // Adjust our weights according to this error
        // derivative and the learning rate
        weightMatrix_.at(i,j) += dWeight * learningRate;
      }

      // Don't forget about the bias
      auto dBias = dError.at(i) * Activation::derivative(y.at(i));
      bias_.at(i) += dBias * learningRate;
    }

    // Back propagate the error
    return std::make_tuple(weightMatrix_.transpose() * dError, Loss::error(y, t));
  }

  WeightMatrix& getWeightMatrix() {
    return weightMatrix_;
  }

  Bias& getBias() {
    return bias_;
  }

  std::tuple<Input, Output>
  correct(Input  const& input, 
          Output const& output, 
          Output const& error, 
          float         learningRate = 0.1f) {
    // For each output 
    const auto target = output + error;
    const auto dError = Loss::derivative(output, target);
    for (auto i = 0; i < Y; i++) {
      // Look at each input from the previous layer
      for (auto j = 0; j < X; j++) {
        // Calculate the partial deriviate of the error with
        // respect to the weight
        auto dWeight = dError.at(i) * Activation::derivative(output.at(i)) * input.at(j);

        // Adjust our weights according to this error
        // derivative and the learning rate
        weightMatrix_.at(i,j) += dWeight * learningRate;
      }

      // Don't forget about the bias
      auto dBias = dError.at(i) * Activation::derivative(output.at(i));
      bias_.at(i) += dBias * learningRate;
    }

    // Back propagate the error
    return std::make_tuple(weightMatrix_.transpose() * dError, Loss::error(output, target));
  }

private:
  WeightMatrix  weightMatrix_;
  Bias          bias_;
};

//------------------------------------------------------------------------------

} // namespace rook

//------------------------------------------------------------------------------

#endif
