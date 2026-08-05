#pragma once
#include "Common.hpp"
#include "Map.hpp"
#include "Math.hpp"
namespace SafeOp {
    void move(int num, const Point& p);
    void fire(int num, const Point& p);
    void flash(int num);
    void meteor(int num, const Point& p);
}
