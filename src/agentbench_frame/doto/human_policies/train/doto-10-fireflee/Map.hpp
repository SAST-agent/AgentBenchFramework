#pragma once
#include "Common.hpp"
#include "Math.hpp"
namespace MapHelper {
    bool isWall(int x, int y);
    bool isWall(const Point& v);
    bool noWall(const Point& s, const Point& t);
    bool isWalkWall(const Point& v);
    FT estimateDis(const Point& a, const Point& b);
    Point maxFire(const Point& base);
    Point clampMap(const Point& p);
}
using PosInfo = std::pair<int, Point>;
std::vector<PosInfo> teamMove(std::vector<Point>& cnxt,
                              const MoveOp& op);
