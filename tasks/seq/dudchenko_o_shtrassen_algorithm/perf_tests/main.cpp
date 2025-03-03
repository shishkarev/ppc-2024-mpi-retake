#include <gtest/gtest.h>

#include <random>

#include "core/perf/include/perf.hpp"
#include "seq/dudchenko_o_shtrassen_algorithm/include/ops_seq.hpp"

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

TEST(dudchenko_o_shtrassen_algorithm_seq, test_pipeline_run) {
  const size_t N = 128;

  std::vector<double> A = generate_random_square_matrix(N);
  std::vector<double> B = generate_random_square_matrix(N);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  std::vector<double> out(N * N, 0.0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<dudchenko_o_shtrassen_algorithm_seq::StrassenAlgoriphmSequential>(taskDataSeq);
  ASSERT_TRUE(testTaskSequential->ValidationImpl());
  ASSERT_TRUE(testTaskSequential->PreProcessingImpl());
  ASSERT_TRUE(testTaskSequential->RunImpl());
  ASSERT_TRUE(testTaskSequential->PostProcessingImpl());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 2;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
}

TEST(dudchenko_o_shtrassen_algorithm_seq, test_task_run) {
  const size_t N = 128;

  std::vector<double> A = generate_random_square_matrix(N);
  std::vector<double> B = generate_random_square_matrix(N);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());

  std::vector<double> out(N * N, 0.0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<dudchenko_o_shtrassen_algorithm_seq::StrassenAlgoriphmSequential>(taskDataSeq);

  ASSERT_TRUE(testTaskSequential->ValidationImpl());
  ASSERT_TRUE(testTaskSequential->PreProcessingImpl());
  ASSERT_TRUE(testTaskSequential->RunImpl());
  ASSERT_TRUE(testTaskSequential->PostProcessingImpl());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 2;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->TaskRun(perfAttr, perfResults);
  ppc::core::Perf::PrintPerfStatistic(perfResults);
}