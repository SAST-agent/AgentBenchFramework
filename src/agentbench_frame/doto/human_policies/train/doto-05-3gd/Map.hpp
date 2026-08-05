#pragma once
#include "Common.hpp"
#include "Math.hpp"
#include "Sampler.hpp"

bool isWall(const Point& v);
bool noWall(const Point& s, const Point& t);
Point maxFire(const Point& base);
bool canFireAt(const Point& cp, const Point& obj);
bool canFireTo(const Point& cp, const Point& obj);
bool inMeteor(const Point& p);
bool inFire(const Point& p);
Point path(int id, const Point& dst, FT minRad, FT maxRad);
void doMove(int num, const Point& dst, const SourceLocation& callPos);
PointSampler makeMoveBaseSampler(int id);
FT evalSafeFacFlash(const Point& base);
std::vector<std::pair<Point, Point>> preSafeFac(const Point& base);
FT evalSafeFacMove(const Point& base, const Point& mdir,
                   const std::vector<std::pair<Point, Point>>& fbs);
bool hasWall(const Point& p);
Point scanCorner(const Point& base, const Point& dst);

PointPredFunc isWallFunc();
PointPredFunc hasWallFunc();
PointPredFunc inFireFunc();
PointPredFunc inMeteorFunc();
PointPredFunc canFireAtFunc(const Point& src);
PointPredFunc noWallFunc(const Point& src);
PointPredFunc canFireToFunc(const Point& src);
