#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_shtrassen_algorithm_mpi {

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

class StrassenAlgoriphmParallel : public ppc::core::Task {
 public:
  explicit StrassenAlgoriphmParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> strassen_mpi(const std::vector<double>& A, const std::vector<double>& B, size_t n);

 private:
  std::vector<double> matrixA_;
  std::vector<double> matrixB_;
  std::vector<double> result_;
  size_t size_;

  boost::mpi::communicator world;
};

std::vector<double> add(const std::vector<double>& A, const std::vector<double>& B, size_t n);
std::vector<double> subtract(const std::vector<double>& A, const std::vector<double>& B, size_t n);
std::vector<double> strassen_seq(const std::vector<double>& A, const std::vector<double>& B, size_t n);
}  // namespace dudchenko_o_shtrassen_algorithm_mpi