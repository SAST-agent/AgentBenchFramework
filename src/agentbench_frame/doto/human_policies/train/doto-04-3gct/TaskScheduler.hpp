#pragma once
#include "Common.hpp"
enum class TaskType { Trans, Steal, NxtSteal, Guard, Bonus, Group };
struct Task final {
    std::vector<int> id;
    TaskType task;
    int extra;
};
std::vector<Task> calcTask();
