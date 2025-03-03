#include <gtest/gtest.h>

#include <mpi/dudchenko_o_shtrassen_algorithm/include/ops_mpi.hpp>
#include <random>

static std::vector<double> generate_random_square_matrix(int n, double minValue = -200.0, double maxValue = 200.0) {
  std::vector<double> matrix(n * n);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(minValue, maxValue);

  for (int i = 0; i < n * n; ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

static void create_test(size_t N) {
  boost::mpi::communicator world;
  std::vector<double> A = generate_random_square_matrix(N);
  std::vector<double> B = generate_random_square_matrix(N);

  std::vector<double> out_seq(N * N, 0.0);
  std::vector<double> out_par(N * N, 0.0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataSeq->inputs_count.emplace_back(A.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(B.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmSequential strassenMatrixMultSeq(
        taskDataSeq);
    ASSERT_TRUE(strassenMatrixMultSeq.ValidationImpl());
    ASSERT_TRUE(strassenMatrixMultSeq.PreProcessingImpl());
    ASSERT_TRUE(strassenMatrixMultSeq.RunImpl());
    ASSERT_TRUE(strassenMatrixMultSeq.PostProcessingImpl());
  }

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }
  dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel strassenMatrixMultPar(
      taskDataPar);
  ASSERT_TRUE(strassenMatrixMultPar.ValidationImpl());
  ASSERT_TRUE(strassenMatrixMultPar.PreProcessingImpl());
  ASSERT_TRUE(strassenMatrixMultPar.RunImpl());
  ASSERT_TRUE(strassenMatrixMultPar.PostProcessingImpl());

  for (size_t i = 0; i < N * N; i++) {
    EXPECT_NEAR(out_seq[i], out_par[i], 1e-8);
  }
}

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_2x2_matrices) { create_test(2); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_3x3_matrices) { create_test(3); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_4x4_matrices) { create_test(4); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_5x5_matrices) { create_test(5); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_10x10_matrices) { create_test(10); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_12x12_matrices) { create_test(12); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_15x15_matrices) { create_test(15); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_23x23_matrices) { create_test(23); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_25x25_matrices) { create_test(25); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_50x50_matrices) { create_test(50); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_64x64_matrices) { create_test(64); }

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_different_size_matrices) {
  boost::mpi::communicator world;
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> B = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());

    dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel strassenMatrixMultPar(
        taskDataPar);
    ASSERT_FALSE(strassenMatrixMultPar.ValidationImpl());
  }
}

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_non_squared_matrices) {
  boost::mpi::communicator world;
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> B = {6.0, 7.0, 8.0, 9.0, 10.0};

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());

    dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel strassenMatrixMultPar(
        taskDataPar);
    ASSERT_FALSE(strassenMatrixMultPar.ValidationImpl());
  }
}

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_non_valid_input) {
  boost::mpi::communicator world;
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> B = {6.0, 7.0, 8.0, 9.0, 10.0};

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(B.size());

    dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel strassenMatrixMultPar(
        taskDataPar);
    ASSERT_FALSE(strassenMatrixMultPar.ValidationImpl());
  }
}

TEST(dudchenko_o_shtrassen_algorithm_mpi, test_non_valid_outputs_size) {
  boost::mpi::communicator world;

  const size_t N = 2;
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> B = {5.0, 6.0, 7.0, 8.0};
  std::vector<double> out_par(N, 0.0);  // we need N * N

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());

    dudchenko_o_shtrassen_algorithm_mpi::StrassenAlgoriphmParallel strassenMatrixMultPar(
        taskDataPar);
    ASSERT_FALSE(strassenMatrixMultPar.ValidationImpl());
  }
}