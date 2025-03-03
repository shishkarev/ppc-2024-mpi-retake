#pragma once

#include <cmath>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_shtrassen_algorithm_seq {

class StrassenAlgoriphmSequential : public ppc::core::Task {
 public:
  explicit StrassenAlgoriphmSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> matrixA_;
  std::vector<double> matrixB_;
  std::vector<double> result_;
  size_t size_;
};

std::vector<double> add(const std::vector<double>& A, const std::vector<double>& B, size_t n);
std::vector<double> subtract(const std::vector<double>& A, const std::vector<double>& B, size_t n);
std::vector<double> strassen(const std::vector<double>& A, const std::vector<double>& B, size_t n);
}  // namespace dudchenko_o_shtrassen_algorithm_seq