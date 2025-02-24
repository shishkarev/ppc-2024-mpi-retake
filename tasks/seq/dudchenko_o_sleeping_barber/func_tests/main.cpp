#include <gtest/gtest.h>

#include <memory>

#include "seq/dudchenko_o_sleeping_barber/include/ops_seq.hpp"

TEST(dudchenko_o_sleeping_barber_sequential, validation_test_1) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int output_value = 0;
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_value));
  taskDataSeq->outputs_count.emplace_back(sizeof(int));

  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);
  taskDataSeq->inputs_count = {0};
  EXPECT_FALSE(testSleepingBarber.ValidationImpl());
}

TEST(dudchenko_o_sleeping_barber_sequential, validation_test_2) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int output_value = 0;
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_value));
  taskDataSeq->outputs_count.emplace_back(sizeof(int));

  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);
  taskDataSeq->inputs_count = {1};
  EXPECT_FALSE(testSleepingBarber.ValidationImpl());
}

TEST(dudchenko_o_sleeping_barber_sequential, validation_test_3) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int output_value = 0;
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output_value));
  taskDataSeq->outputs_count.emplace_back(sizeof(int));

  dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);
  taskDataSeq->inputs_count = {5};
  EXPECT_TRUE(testSleepingBarber.ValidationImpl());
}

TEST(dudchenko_o_sleeping_barber_seq, functional_test) {
  std::vector<int> test_cases = {3, 1024};

  for (int max_waiting_chairs : test_cases) {
    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    int global_res = -1;

    taskDataSeq->inputs_count.emplace_back(max_waiting_chairs);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_res));
    taskDataSeq->outputs_count.emplace_back(sizeof(global_res));

    dudchenko_o_sleeping_barber_seq::TestSleepingBarber testSleepingBarber(taskDataSeq);

    ASSERT_TRUE(testSleepingBarber.ValidationImpl());
    ASSERT_TRUE(testSleepingBarber.PreProcessingImpl());
    ASSERT_TRUE(testSleepingBarber.RunImpl());
    ASSERT_TRUE(testSleepingBarber.PostProcessingImpl());

    EXPECT_EQ(global_res, 0);
  }
}
