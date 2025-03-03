#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/dudchenko_o_shtrassen_algorithm/include/ops_mpi.hpp"

static std::vector<double> generate_random_square_matrix(int n, double minValue = -50.0, double maxValue = 50.0) {
  std::vector<double> matrix(n * n);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(minValue, maxValue);

  for (int i = 0; i < n * n; ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const size_t N = 256;

  std::vector<double> A = generate_random_square_matrix(N);
  std::vector<double> B = generate_random_square_matrix(N);
  std::vector<double> out(N * N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testTaskParallel =
      std::make_shared<dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel>(
          taskDataPar);
  ASSERT_TRUE(testTaskParallel->ValidationImpl());
  ASSERT_TRUE(testTaskParallel->PreProcessingImpl());
  ASSERT_TRUE(testTaskParallel->RunImpl());
  ASSERT_TRUE(testTaskParallel->PostProcessingImpl());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 2;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
  }
}

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_task_run) {
  boost::mpi::communicator world;
  const size_t N = 256;

  std::vector<double> A = generate_random_square_matrix(N);
  std::vector<double> B = generate_random_square_matrix(N);
  std::vector<double> out(N * N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testTaskParallel =
      std::make_shared<dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel>(
          taskDataPar);
  ASSERT_TRUE(testTaskParallel->ValidationImpl());
  ASSERT_TRUE(testTaskParallel->PreProcessingImpl());
  ASSERT_TRUE(testTaskParallel->RunImpl());
  ASSERT_TRUE(testTaskParallel->PostProcessingImpl());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 2;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskParallel);
  perfAnalyzer->TaskRun(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
  }
}