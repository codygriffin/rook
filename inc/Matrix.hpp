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

#ifndef INCLUDED_MATRIX_HPP
#define INCLUDED_MATRIX_HPP

#include <cmath>

//------------------------------------------------------------------------------

namespace rook {

//------------------------------------------------------------------------------

template <size_t M, size_t N, typename K>
Matrix<M, N, K>::Matrix(const std::array<K, M*N>& m) 
: weightMatrix_(m) { 
}

template <size_t M, size_t N, typename K>
Matrix<M, N, K>::Matrix() 
: weightMatrix_({0}) { 
}

template <size_t M, size_t N, typename K>
Matrix<M, N, K> 
Matrix<M, N, K>::operator+=(Matrix const& a) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      at(i, j) += a.at(i, j);    
    }
  }
  return *this;
}

template <size_t M, size_t N, typename K>
Matrix<M, N, K> 
Matrix<M, N, K>::operator-=(Matrix const& a) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      at(i, j) -= a.at(i, j);    
    }
  }
  return *this;
}

template <size_t M, size_t N, typename K>
K 
Matrix<M, N, K>::at(size_t i, size_t j) const {
  return weightMatrix_[(N*i) + j];
}

template <size_t M, size_t N, typename K>
K& 
Matrix<M, N, K>::at(size_t i, size_t j) {
  return weightMatrix_[(N*i) + j];
}

template <size_t M, size_t N, typename K>
K 
Matrix<M, N, K>::at(size_t i) const {
  return M == 1?
    at(0, i):
    at(i, 0); 
}

template <size_t M, size_t N, typename K>
K&  
Matrix<M, N, K>::at(size_t i) {
  return M == 1?
    at(0, i):
    at(i, 0); 
}

template <size_t M, size_t N, typename K>
Matrix<M, 1, K>  
Matrix<M, N, K>::col(size_t i) const {
  throw std::runtime_error("unimplemented");
}

template <size_t M, size_t N, typename K>
Matrix<1, N, K>  
Matrix<M, N, K>::row(size_t i) const {
  throw std::runtime_error("unimplemented");
}

template <size_t M, size_t N, typename K>
void 
Matrix<M, N, K>::print(const std::string& name) const {
  std::cout << name << std::endl;
  for (size_t i = 0; i < M; i++) {
    std::cout << "| ";
    for (size_t j = 0; j < N; j++) {
      std::cout << std::setw(7) << std::setprecision(3) << std::fixed << (this->at(i, j)) << " ";
    }
    std::cout << " |" << std::endl;
  }
  std::cout << std::endl;
}

template <size_t M, size_t N, typename K>
Matrix<N, M, K>
Matrix<M, N, K>::transpose() const {
  Matrix<N, M, K> result;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      result.at(i, j) += at(j, i);  
    } 
  }
  return result;
}

//------------------------------------------------------------------------------

template <size_t M, size_t N, typename K>
Matrix<M, N, K> 
Matrix<M, N, K>::apply(std::function<K (K)> func) const {
  Matrix<M, N, K> result;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      result.at(i, j) = func(at(i, j));  
    } 
  }
  return result;
}

template <size_t M, size_t N, typename K>
Matrix<M, N, K> 
Matrix<M, N, K>::apply(std::function<K (size_t, size_t)> func) const {
  Matrix<M, N, K> result;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      result.at(i, j) = func(i, j);  
    } 
  }
  return result;
}

template <size_t M, size_t N, typename K>
Matrix<M, N, K> 
Matrix<M, N, K>::vapply(std::function<K (size_t)> func) const {
  Matrix<M, N, K> result;
  for (size_t i = 0; i < std::max(M, N); i++) {
    result.at(i) = func(i);  
  }
  return result;
}

template <size_t M, size_t N, typename K>
Matrix<M, N, K> 
Matrix<M, N, K>::apply(std::function<K (void)> func) const {
  Matrix<M, N, K> result;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      result.at(i, j) = func();  
    } 
  }
  return result;
}

//------------------------------------------------------------------------------

template <size_t M, size_t N, typename K>
Matrix<M, N, K> 
operator+(Matrix<M, N, K> a, Matrix<M, N, K> const& b) {
  a += b;
  return a;  
}

template <size_t M, size_t N, typename K>
Matrix<M, N, K> 
operator-(Matrix<M, N, K> a, Matrix<M, N, K> const& b) {
  a -= b;
  return a;  
}

//TODO VECTORIZE
template <size_t L, size_t M, size_t N, typename K>
Matrix<M, N, K> 
operator*(Matrix<M, L, K> const& a, Matrix<L, N, K> const& b) {
  Matrix<M, N, K> result;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < L; k++) {
        result.at(i, j) += a.at(i, k) * b.at(k, j);  
      }
    } 
  }
  return result;
}

template <size_t M, size_t N, typename K>
Matrix<M, N, K> 
operator%(Matrix<M, N, K> a, Matrix<M, N, K> const& b) {
  Matrix<M, N, K> result;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      result.at(i, j) += a.at(i, j) * b.at(i, j);  
    } 
  }
  return result;
}


template <size_t M, size_t N, typename K>
bool
operator==(Matrix<M, N, K> a, Matrix<M, N, K> const& b) {
  bool result = true;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      result = result && a.at(i,j) == b.at(i,j);
    } 
  }
  return result;
}

template <size_t M, size_t N, typename K>
bool
operator!=(Matrix<M, N, K> a, Matrix<M, N, K> const& b) {
  return !(a == b);
}

//------------------------------------------------------------------------------

template <size_t N, typename K>
ColVector<N+1, K>
aug1(const ColVector<N, K>& col, K a) {
  ColVector<N+1, K> result;
  result = result.vapply([&](size_t i) {
    return (i < N)?col.at(i):a;
  });
  return result;
}

template <size_t N, typename K>
RowVector<N+1, K>
aug1(const RowVector<N, K>& row, K a) {
  RowVector<N+1, K> result;
  result = result.vapply([&](size_t i) {
    return (i < N)?row.at(i):a;
  });
  return result;
}

template <size_t N, typename K>
ColVector<N-1, K>
strip1(const ColVector<N, K>& col) {
  ColVector<N-1, K> result;
  result = result.vapply([&](size_t i) {
    return col.at(i);
  });
  return result;
}

template <size_t N, typename K>
RowVector<N-1, K>
strip1(const RowVector<N, K>& row) {
  RowVector<N-1, K> result;
  result = result.vapply([&](size_t i) {
    return row.at(i);
  });
  return result;
}

template <size_t N, typename K>
K
mag(const RowVector<N, K>& row) {
  K result;
  for (int i = 0; i < N; i++) {
    result += row.at(i);
  }
  return sqrtf(result);
}

//------------------------------------------------------------------------------

} // namespace rook

//------------------------------------------------------------------------------

#endif
