#pragma once
#include "Common.hpp"
#include "geometry.h"
#include <random>
inline Point operator+(const Point& lhs,
                       const Point& rhs) {
    return Point(lhs.x + rhs.x, lhs.y + rhs.y);
}
inline Point operator*(const Point& v, FT k) {
    return Point(v.x * k, v.y * k);
}
inline Point operator-(const Point& lhs,
                       const Point& rhs) {
    return Point(lhs.x - rhs.x, lhs.y - rhs.y);
}
inline FT length(const Point& v) {
    return sqrt(v.x * v.x + v.y * v.y);
}
inline Point normalize(const Point& v) {
    FT lv = length(v);
    if(lv < EP)
        return v;
    return Point(v.x / lv, v.y / lv);
}

namespace Math {
    const FT bias = 1e-4;
    const FT pi = 3.14159265358979323846;
    const FT twoPi = 2.0 * pi;
    const FT pi2 = 1.57079632679489661923;
    const FT pi4 = 0.78539816339744830962;
    const FT invPi = 0.31830988618379067154;
    const FT twoInvPi = 0.63661977236758134308;
    const FT twoISqrtPi = 1.12837916709551257390;
    const FT sqrt2 = 1.41421356237309504880;
    const FT invSqrt2 = 0.70710678118654752440;
    using Rng = std::mt19937_64;
    Rng& getRng();
    FT clampRad(FT x, FT rad);
    FT sampleGauss(FT radius);
    Point sampleRing(const Point& base, FT radius);
    Point sampleCircle(const Point& base, FT radius);
    Point sampleCircleGauss(const Point& base,
                            FT radius);
    Point lerp(const Point& a, const Point& b, FT w);
    FT uniform(FT l, FT r);
    Point rlerp(const Point& a, const Point& b);
    bool accept(FT p);
    FT dot(const Point& a, const Point& b);
    FT cross(const Point& a, const Point& b);
}
