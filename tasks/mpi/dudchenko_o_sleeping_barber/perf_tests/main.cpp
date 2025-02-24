#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/dudchenko_o_sleeping_barber/include/ops_mpi.hpp"

TEST(dudchenko_o_sleeping_barber_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  if (world.size() < 3) return;

  const int max_waiting_chairs = 3;
  bool barber_busy_ = false;
  std::vector<int> global_res(1, 0);
  int num_clients = std::max(1, world.size() - 2);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count = {max_waiting_chairs, static_cast<unsigned int>(barber_busy_)};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->ValidationImpl());
  testMpiTaskParallel->PreProcessingImpl();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = num_clients;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  testMpiTaskParallel->RunImpl();
  testMpiTaskParallel->PostProcessingImpl();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->PipelineRun(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    ASSERT_EQ(1, global_res[0]);
  }
}

TEST(dudchenko_o_sleeping_barber_mpi, test_task_run) {
  boost::mpi::communicator world;
  if (world.size() < 3) return;

  const int max_waiting_chairs = 3;
  std::vector<int> global_res(1, 0);
  bool barber_busy_ = false;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_clients = std::max(1, world.size() - 2);

  if (world.rank() == 0) {
    taskDataPar->inputs_count = {max_waiting_chairs, static_cast<unsigned int>(barber_busy_)};
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<dudchenko_o_sleeping_barber_mpi::TestMPISleepingBarber>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->ValidationImpl());
  testMpiTaskParallel->PreProcessingImpl();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = num_clients;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  testMpiTaskParallel->RunImpl();
  testMpiTaskParallel->PostProcessingImpl();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->TaskRun(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perfResults);
    ASSERT_EQ(1, global_res[0]);
  }
}