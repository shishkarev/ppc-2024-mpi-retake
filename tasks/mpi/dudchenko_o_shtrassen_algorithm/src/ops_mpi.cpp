#include "mpi/dudchenko_o_shtrassen_algorithm/include/ops_mpi.hpp"

#include <algorithm>

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmSequential::PreProcessingImpl() {
  auto* inputsA = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* inputsB = reinterpret_cast<double*>(task_data->inputs[1]);
  size_ = static_cast<size_t>(std::sqrt(task_data->inputs_count[0]));
  matrixA_.assign(inputsA, inputsA + size_ * size_);
  matrixB_.assign(inputsB, inputsB + size_ * size_);
  result_.resize(size_ * size_);

  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmSequential::ValidationImpl() {
  return !task_data->inputs.empty() && task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->inputs_count[0] == static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) *
                                           static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmSequential::RunImpl() {
  result_ = strassen_seq(matrixA_, matrixB_, size_);
  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmSequential::PostProcessingImpl() {
  auto* outputs = reinterpret_cast<double*>(task_data->outputs[0]);
  std::copy(result_.begin(), result_.end(), outputs);
  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel::PreProcessingImpl() {
  if (world.rank() == 0) {
    auto* inputsA = reinterpret_cast<double*>(task_data->inputs[0]);
    auto* inputsB = reinterpret_cast<double*>(task_data->inputs[1]);
    size_ = static_cast<size_t>(std::sqrt(task_data->inputs_count[0]));
    matrixA_.assign(inputsA, inputsA + size_ * size_);
    matrixB_.assign(inputsB, inputsB + size_ * size_);
    result_.resize(size_ * size_);
  }
  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel::ValidationImpl() {
  if (world.rank() == 0) {
    return !task_data->inputs.empty() && task_data->inputs_count[0] == task_data->inputs_count[1] &&
           task_data->inputs_count[0] == static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) *
                                             static_cast<size_t>(std::sqrt(task_data->inputs_count[0])) &&
           task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel::RunImpl() {
  result_ = strassen_mpi(matrixA_, matrixB_, size_);
  return true;
}

bool dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel::PostProcessingImpl() {
  if (world.rank() == 0) {
    auto* outputs = reinterpret_cast<double*>(task_data->outputs[0]);
    std::copy(result_.begin(), result_.end(), outputs);
  }
  return true;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_mpi::add(const std::vector<double>& A, const std::vector<double>& B,
                                                             size_t n) {
  std::vector<double> result(n * n);
  std::transform(A.begin(), A.end(), B.begin(), result.begin(), std::plus<double>());
  return result;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_mpi::subtract(const std::vector<double>& A,
                                                                  const std::vector<double>& B, size_t n) {
  std::vector<double> result(n * n);
  std::transform(A.begin(), A.end(), B.begin(), result.begin(), std::minus<double>());
  return result;
}

std::vector<double> dudchenko_o_shtrassen_algorithm_mpi::strassen_seq(const std::vector<double>& A,
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

  auto M1 = strassen_seq(add(A11, A22, half), add(B11, B22, half), half);
  auto M2 = strassen_seq(add(A21, A22, half), B11, half);
  auto M3 = strassen_seq(A11, subtract(B12, B22, half), half);
  auto M4 = strassen_seq(A22, subtract(B21, B11, half), half);
  auto M5 = strassen_seq(add(A11, A12, half), B22, half);
  auto M6 = strassen_seq(subtract(A21, A11, half), add(B11, B12, half), half);
  auto M7 = strassen_seq(subtract(A12, A22, half), add(B21, B22, half), half);

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

std::vector<double> dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel::strassen_mpi(
    const std::vector<double>& A, const std::vector<double>& B, size_t n) {
  if (world.rank() > 6) {
    world.split(1);
    return {};
  }

  boost::mpi::communicator active_comm = world.split(0);
  int rank = active_comm.rank();
  int size = active_comm.size();

  boost::mpi::broadcast(active_comm, n, 0);

  size_t newSize = 1;
  while (newSize < n) newSize *= 2;

  std::vector<double> A_ext(newSize * newSize, 0.0), B_ext(newSize * newSize, 0.0);
  if (rank == 0) {
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j) {
        A_ext[i * newSize + j] = A[i * n + j];
        B_ext[i * newSize + j] = B[i * n + j];
      }
  }

  boost::mpi::broadcast(active_comm, A_ext, 0);
  boost::mpi::broadcast(active_comm, B_ext, 0);

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

  std::vector<std::vector<double>> M(7, std::vector<double>(half_squared, 0.0));
  for (int task = rank; task < 7; task += size) {
    std::array<std::function<std::vector<double>()>, 7> computations = {
        [&]() { return strassen_seq(add(A11, A22, half), add(B11, B22, half), half); },
        [&]() { return strassen_seq(add(A21, A22, half), B11, half); },
        [&]() { return strassen_seq(A11, subtract(B12, B22, half), half); },
        [&]() { return strassen_seq(A22, subtract(B21, B11, half), half); },
        [&]() { return strassen_seq(add(A11, A12, half), B22, half); },
        [&]() { return strassen_seq(subtract(A21, A11, half), add(B11, B12, half), half); },
        [&]() { return strassen_seq(subtract(A12, A22, half), add(B21, B22, half), half); }};

    for (int task = rank; task < 7; task += size) {
      M[task] = computations[task]();
    }
  }

  std::vector<double> M_global(7 * half_squared, 0.0);
  for (size_t i = 0; i < 7; ++i) {
    boost::mpi::reduce(active_comm, M[i].data(), half_squared, M_global.data() + i * half_squared, std::plus(), 0);
  }

  if (rank == 0) {
    std::vector<double> result_ext(newSize * newSize, 0.0);
    for (size_t i = 0; i < half; ++i)
      for (size_t j = 0; j < half; ++j) {
        size_t idx = i * half + j;
        result_ext[i * newSize + j] = M_global[idx] + M_global[3 * half_squared + idx] -
                                      M_global[4 * half_squared + idx] + M_global[6 * half_squared + idx];
        result_ext[i * newSize + j + half] = M_global[2 * half_squared + idx] + M_global[4 * half_squared + idx];
        result_ext[(i + half) * newSize + j] = M_global[1 * half_squared + idx] + M_global[3 * half_squared + idx];
        result_ext[(i + half) * newSize + j + half] = M_global[idx] - M_global[1 * half_squared + idx] +
                                                      M_global[2 * half_squared + idx] +
                                                      M_global[5 * half_squared + idx];
      }

    std::vector<double> final_result(n * n);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j) final_result[i * n + j] = result_ext[i * newSize + j];

    return final_result;
  }
  return {};
}
