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

#ifndef INCLUDED_MATRIX_H
#define INCLUDED_MATRIX_H

#include <iostream>
#include <random>
#include <iomanip>
#include <cstdint>
#include <array>
#include <functional>

//------------------------------------------------------------------------------

namespace rook { 

//------------------------------------------------------------------------------

// An MxN matrix (M rows, N columns) over a 
// field K (defaults to float)
template <size_t M, size_t N, typename K = float>
struct Matrix {
  // Static definitions
  typedef K Field;
  static const size_t rows = M;
  static const size_t cols = N;

  typedef Matrix<1, N, K> Row;
  typedef Matrix<M, 1, K> Col;

  // Constructors
  Matrix(const std::array<K, M*N>& m); 
  Matrix(std::function<K (size_t, size_t)> func) { generate(func); } 
  Matrix(std::function<K (size_t)>         func) { generate(func); } 
  Matrix(); 
  
  // Arithmetic
  Matrix operator+=(Matrix const& a);
  Matrix operator-=(Matrix const& a);

  // Indexing
  K  at(size_t i, size_t j) const;
  K& at(size_t i, size_t j);
  K  at(size_t i) const;
  K& at(size_t i);

  Col&  col(size_t i);
  Row&  row(size_t i);

  Col   col(size_t i) const;
  Row   row(size_t i) const;

  // TODO Slicing? (MatLab : syntax)
  
  void print(const std::string& name = "") const;

  Matrix<N, M, K> transpose() const;

  void             generate(std::function<K (size_t, size_t)> func);
  void             generate(std::function<K (size_t)>         func);

  Matrix<M, N, K>  apply   (std::function<K (K)> func)              const;

  Matrix<M, N, K>  each    (std::function<K (size_t, size_t)> func) const;
  Matrix<M, N, K>  each    (std::function<K (size_t)> func)         const;

  Matrix<M, N, K>  eachRow (std::function<void (size_t, const Row&)> func)   const;
  Matrix<M, N, K>  eachCol (std::function<void (size_t, const Col&)> func)   const;

  std::array<K, M*N>&       raw()       { return weightMatrix_; }
  const std::array<K, M*N>& raw() const { return weightMatrix_; }

private:
  std::array<K, M*N> weightMatrix_;
};

// A column vector of size N has N rows and 
// a single column
template <size_t N, typename K = float>
using ColVector = Matrix<N, 1, K>;

// A row vector of size N has a single row
// and N columns
template <size_t N, typename K = float>
using RowVector = Matrix<1, N, K>;

//------------------------------------------------------------------------------
// Some handy generators
template <typename K>
K zero(size_t i, size_t j) {
  return (K)0.0f;
}

template <typename K>
K zero(size_t i) {
  return (K)0.0f;
}

std::function<float (size_t, size_t)> normal(float mean, float stddev) {
  static std::random_device rd;
  static std::mt19937       gen(rd());
  static std::normal_distribution<> normal(mean, stddev);
  return [&](size_t i, size_t j) -> float {
    return normal(gen); 
  };
}

//------------------------------------------------------------------------------

} // namespace rook

//------------------------------------------------------------------------------

#include "Matrix.hpp"

#endif
