#include "mpi/dudchenko_o_sleeping_barber/include/ops_mpi.hpp"

#include <deque>
#include <functional>
#include <string>
#include <thread>

namespace dudchenko_o_sleeping_barber_mpi {

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

bool TestMPISleepingBarber::PreProcessingImpl() {
  result = -1;

  if (world.rank() == 0) {
    max_wait = task_data->inputs_count[0];
  }

  return true;
}

bool TestMPISleepingBarber::ValidationImpl() {
  if (world.rank() == 0) {
    if (task_data->inputs_count.empty() || task_data->inputs_count[0] <= 0) {
      return false;
    }
  }

  if (world.rank() < 0 || world.rank() >= world.size()) {
    throw std::runtime_error("Invalid rank.");
  }

  return true;
}

bool TestMPISleepingBarber::RunImpl() {
  if (world.size() < 3) return true;
  if (world.rank() == 0) {
    while (true) {
      int client = -1;

      world.recv(1, 0, client);

      if (client == -1) {
        result = 0;
        break;
      }

      next_client(client);
    }
  } else if (world.rank() == 1) {
    std::deque<int> waiting_clients;
    max_wait = task_data->inputs_count[0];
    int remaining_clients = world.size() - 2;
    bool barber_busy = false;

    while (true) {
      int client = -1;

      if (world.iprobe(boost::mpi::any_source, 0)) {
        world.recv(boost::mpi::any_source, 0, client);

        if (static_cast<int>(waiting_clients.size()) < max_wait) {
          waiting_clients.push_back(client);
          world.send(client, 1, true);
        } else {
          world.send(client, 1, false);
        }
      }

      if (!barber_busy && !waiting_clients.empty()) {
        int next_client = waiting_clients.front();
        waiting_clients.pop_front();
        world.send(0, 0, next_client);
        barber_busy = true;
      }

      if (world.iprobe(0, 4)) {
        int barber_signal;
        world.recv(0, 4, barber_signal);
        barber_busy = false;
      }

      if (waiting_clients.empty() && remaining_clients == 0 && !barber_busy) {
        world.send(0, 0, -1);
        break;
      }

      if (world.iprobe(boost::mpi::any_source, 3)) {
        int done_signal;
        world.recv(boost::mpi::any_source, 3, done_signal);
        remaining_clients--;
      }
    }
  } else {
    int client = world.rank();
    bool accepted = false;

    world.send(1, 0, client);

    world.recv(1, 1, accepted);

    if (accepted) {
      world.recv(0, 2, client);
      world.send(1, 3, client);
    } else {
      world.send(1, 3, client);
    }
  }

  return true;
}

bool TestMPISleepingBarber::PostProcessingImpl() {
  world.barrier();

  if (world.rank() == 0) {
    if (!task_data->outputs.empty() && task_data->outputs_count[0] == sizeof(int)) {
      *reinterpret_cast<int*>(task_data->outputs[0]) = result;
    } else {
      return false;
    }
  }

  return true;
}

void TestMPISleepingBarber::next_client(int client) {
  std::this_thread::sleep_for(std::chrono::milliseconds(std::chrono::milliseconds(20)));
  world.send(client, 2, client);
  world.send(1, 4, client);
}

}  // namespace dudchenko_o_sleeping_barber_mpi