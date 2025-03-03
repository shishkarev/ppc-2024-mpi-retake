#include "seq/dudchenko_o_shtrassen_algorithm/include/ops_seq.hpp"

#include <algorithm>

bool dudchenko_o_shtrassen_algorithm_seq::StrassenAlgoriphmSequential::PreProcessingImpl() {
  auto* inputsA = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* inputsB = reinterpret_cast<double*>(task_data->inputs[1]);

  size_ = static_cast<size_t>(std::sqrt(task_data->inputs_count[0]));
  matrixA_.assign(inputsA, inputsA + size_ * size_);
  matrixB_.assign(inputsB, inputsB + size_ * size_);
  result_.resize(size_ * size_);

  return true;
}

bool dudchenko_o_shtrassen_algorithm_seq::StrassenAlgoriphmSequential::ValidationImpl() {
  return !task_data->inputs.empty() && task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->inputs_count[0] == static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) *
                                           static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool dudchenko_o_shtrassen_algorithm_seq::StrassenAlgoriphmSequential::RunImpl() {
  result_ = dudchenko_o_shtrassen_algorithm_seq::strassen(matrixA_, matrixB_, size_);
  return true;
}

bool dudchenko_o_shtrassen_algorithm_seq::StrassenAlgoriphmSequential::PostProcessingImpl() {
  auto* outputs = reinterpret_cast<double*>(task_data->outputs[0]);

  std::copy(result_.begin(), result_.end(), outputs);

  return true;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_seq::add(const std::vector<double>& A, const std::vector<double>& B,
                                                             size_t n) {
  std::vector<double> result(n * n);
  std::transform(A.begin(), A.end(), B.begin(), result.begin(), std::plus<double>());
  return result;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_seq::subtract(const std::vector<double>& A,
                                                                  const std::vector<double>& B, size_t n) {
  std::vector<double> result(n * n);
  std::transform(A.begin(), A.end(), B.begin(), result.begin(), std::minus<double>());
  return result;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_seq::strassen(const std::vector<double>& A,
                                                                  const std::vector<double>& B, size_t n) {
  if (n == 1) {
    return {A[0] * B[0]};
  }

  size_t newSize = 1;
  while (newSize < n) newSize *= 2;

  std::vector<double> A_ext(newSize * newSize, 0.0), B_ext(newSize * newSize, 0.0);
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++) {
      A_ext[i * newSize + j] = A[i * n + j];
      B_ext[i * newSize + j] = B[i * n + j];
    }

  size_t half = newSize / 2;
  size_t half_squared = half * half;
  auto getSubmatrix = [&](const std::vector<double>& M, size_t row, size_t col) {
    std::vector<double> sub(half_squared);
    for (size_t i = 0; i < half; ++i)
      for (size_t j = 0; j < half; ++j) sub[i * half + j] = M[(i + row) * newSize + j + col];
    return sub;
  };

  auto A11 = getSubmatrix(A_ext, 0, 0);
  auto A12 = getSubmatrix(A_ext, 0, half);
  auto A21 = getSubmatrix(A_ext, half, 0);
  auto A22 = getSubmatrix(A_ext, half, half);
  auto B11 = getSubmatrix(B_ext, 0, 0);
  auto B12 = getSubmatrix(B_ext, 0, half);
  auto B21 = getSubmatrix(B_ext, half, 0);
  auto B22 = getSubmatrix(B_ext, half, half);

  auto M1 = strassen(add(A11, A22, half), add(B11, B22, half), half);
  auto M2 = strassen(add(A21, A22, half), B11, half);
  auto M3 = strassen(A11, subtract(B12, B22, half), half);
  auto M4 = strassen(A22, subtract(B21, B11, half), half);
  auto M5 = strassen(add(A11, A12, half), B22, half);
  auto M6 = strassen(subtract(A21, A11, half), add(B11, B12, half), half);
  auto M7 = strassen(subtract(A12, A22, half), add(B21, B22, half), half);

  std::vector<double> result_ext(newSize * newSize, 0.0);
  for (size_t i = 0; i < half; ++i) {
    for (size_t j = 0; j < half; ++j) {
      size_t idx = i * half + j;
      size_t extIdx = i * newSize + j;
      result_ext[extIdx] = M1[idx] + M4[idx] - M5[idx] + M7[idx];
      result_ext[extIdx + half] = M3[idx] + M5[idx];
      result_ext[(i + half) * newSize + j] = M2[idx] + M4[idx];
      result_ext[(i + half) * newSize + j + half] = M1[idx] + M3[idx] - M2[idx] + M6[idx];
    }
  }

  std::vector<double> result(n * n);
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++) result[i * n + j] = result_ext[i * newSize + j];

  return result;
}