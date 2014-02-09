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

  // Constructors
  Matrix(const std::array<K, M*N>& m); 
  Matrix(); 
  
  // Arithmetic
  Matrix operator+=(Matrix const& a);
  Matrix operator-=(Matrix const& a);

  // Indexing
  K  at(size_t i, size_t j) const;
  K& at(size_t i, size_t j);
  K  at(size_t i) const;
  K& at(size_t i);

  Matrix<M, 1, K> col(size_t i) const;
  Matrix<1, N, K> row(size_t i) const;

  // TODO Slicing? (MatLab : syntax)
  
  void print(const std::string& name = "") const;

  Matrix<N, M, K> transpose() const;

  // TODO provide overloaded versions w/ std::function
  Matrix<M, N, K>  apply(std::function<K (K)> func)              const;
  Matrix<M, N, K>  apply(std::function<K (size_t, size_t)> func) const;
  Matrix<M, N, K> vapply(std::function<K (size_t)> func)         const;
  Matrix<M, N, K>  apply(std::function<K (void)> func)           const;

private:
  std::array<K, M*N> weightMatrix_;
};

// A column vector of size N has N rows and 
// a single column
template <size_t N, typename K>
using ColVector = Matrix<N, 1, K>;

// A row vector of size N has a single row
// and N columns
template <size_t N, typename K>
using RowVector = Matrix<1, N, K>;

//------------------------------------------------------------------------------

} // namespace rook

//------------------------------------------------------------------------------

#include "Matrix.hpp"

#endif
