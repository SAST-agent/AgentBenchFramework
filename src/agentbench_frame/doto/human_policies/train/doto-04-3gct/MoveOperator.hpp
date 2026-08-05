#pragma once
#include "Common.hpp"
#include <map>
#include <vector>
std::map<int, Point> doSteal(const std::vector<int>& ids, int cid);
std::map<int, Point> doNxtSteal(const std::vector<int>& ids, int cid);
std::map<int, Point> doGuard(const std::vector<int>& ids);
Point doBonus(int id, int bid);
std::map<int, Point> doTrans(const std::vector<int>& ids, int mid);
std::map<int, Point> doGroup(const std::vector<int>& ids);
