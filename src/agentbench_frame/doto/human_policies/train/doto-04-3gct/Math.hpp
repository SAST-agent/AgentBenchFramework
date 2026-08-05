#pragma once
#include "Common.hpp"
#include "SDK/geometry.h"
#include <random>
inline Point operator+(const Point& lhs, const Point& rhs) {
    return Point(lhs.x + rhs.x, lhs.y + rhs.y);
}
inline Point operator*(const Point& v, FT k) {
    return Point(v.x * k, v.y * k);
}
inline Point operator-(const Point& lhs, const Point& rhs) {
    return Point(lhs.x - rhs.x, lhs.y - rhs.y);
}
inline Point operator-(const Point& x) {
    return Point(-x.x, -x.y);
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
inline bool operator==(const Point& a, const Point& b) {
    return std::max(fabs(a.x - b.x), fabs(a.y - b.y)) < 1e-8;
}
inline bool operator!=(const Point& a, const Point& b) {
    return std::max(fabs(a.x - b.x), fabs(a.y - b.y)) >= 1e-8;
}

namespace Math {
    const FT bias = 1e-4;
    const FT hurtBias = 0.02;
    const FT inf = 1e20;
    const FT nan = 0.0 / 0.0;

    const FT pi = 3.14159265358979323846;
    const FT twoPi = 2.0 * pi;
    const FT halfPi = 1.57079632679489661923;
    const FT pi4 = 0.78539816339744830962;
    const FT invPi = 0.31830988618379067154;
    const FT twoInvPi = 0.63661977236758134308;
    const FT twoISqrtPi = 1.12837916709551257390;
    const FT sqrt2 = 1.41421356237309504880;
    const FT invSqrt2 = 0.70710678118654752440;
    const Point invalidPos = Point(-1.0, -1.0);

    using Rng = std::mt19937_64;
    Rng& getRng();
    FT clampRad(FT x, FT rad);
    Point sampleCircle(const Point& base, FT rad);
    Point lerp(const Point& a, const Point& b, FT w);
    FT uniform(FT l, FT r);
    Point rlerp(const Point& a, const Point& b);
    bool accept(FT p);
    FT dot(const Point& a, const Point& b);
    FT cross(const Point& a, const Point& b);
    Point scale(const Point& vec, FT len);
    Point rotate(const Point& vec, FT ang);
    FT sharedArea(const Point& oa, FT ra, const Point& ob, FT rb);
    FT angle(const Point& a, const Point& b);
}
