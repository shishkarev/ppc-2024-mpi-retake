#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace dudchenko_o_sleeping_barber_seq {

class TestSleepingBarber : public ppc::core::Task {
 public:
  explicit TestSleepingBarber(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int max_wait{};
  int result{};

  static void next_client(int client);
};
}  // namespace dudchenko_o_sleeping_barber_seq