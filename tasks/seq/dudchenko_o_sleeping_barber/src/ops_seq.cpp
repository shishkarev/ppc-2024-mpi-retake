#include "seq/dudchenko_o_sleeping_barber/include/ops_seq.hpp"

#include <chrono>
#include <deque>
#include <iostream>
#include <thread>

namespace dudchenko_o_sleeping_barber_seq {

bool TestSleepingBarber::PreProcessingImpl() {
  result = -1;

  if (task_data && !task_data->inputs_count.empty()) {
    max_wait = task_data->inputs_count[0];
    return max_wait > 0;
  }

  return false;
}

bool TestSleepingBarber::ValidationImpl() {
  if (!task_data || task_data->outputs.empty() || task_data->outputs_count[0] != sizeof(int)) {
    return false;
  }

  return task_data && !task_data->inputs_count.empty() && task_data->inputs_count[0] > 1;
}

bool TestSleepingBarber::RunImpl() {
  int total_clients = 10;  // Примерное количество клиентов
  std::deque<int> waiting_clients;
  bool barber_busy = false;

  for (int client = 0; client < total_clients; ++client) {
    if (static_cast<int>(waiting_clients.size()) < max_wait) {
      waiting_clients.push_back(client);
    }

    if (!barber_busy && !waiting_clients.empty()) {
      int next_client_id = waiting_clients.front();
      waiting_clients.pop_front();
      next_client(next_client_id);
      barber_busy = true;
    }

    if (barber_busy && waiting_clients.empty()) {
      barber_busy = false;
    }
  }

  while (!waiting_clients.empty()) {
    int next_client_id = waiting_clients.front();
    waiting_clients.pop_front();
    next_client(next_client_id);
  }

  result = 0;
  return true;
}

bool TestSleepingBarber::PostProcessingImpl() {
  *reinterpret_cast<int*>(task_data->outputs[0]) = result;
  return true;
}

void TestSleepingBarber::next_client(int client) {
  (void)client;
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
}
}  // namespace dudchenko_o_sleeping_barber_seq
