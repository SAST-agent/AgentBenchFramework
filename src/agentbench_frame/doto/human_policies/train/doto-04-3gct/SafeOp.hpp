#pragma once
#include "Common.hpp"
#include "Map.hpp"
#include "Math.hpp"
namespace SafeOp {
    void move(int num, const Point& p, const SourceLocation& callPos);
    void fire(int num, const Point& base, const Point& p,
              const SourceLocation& callPos);
    void flash(int num, const SourceLocation& callPos);
    void meteor(int num, const Point& p, const SourceLocation& callPos);
    void begin();
    void end();
}
