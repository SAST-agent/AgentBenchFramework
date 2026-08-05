#include "Math.hpp"
namespace Math {
    Rng& getRng() {
        static Rng eng(
            Clock::now().time_since_epoch().count());
        return eng;
    }
    FT clampRad(FT x, FT rad) {
        return std::min(std::max(x, -rad), rad);
    }
    FT sampleGauss(FT radius) {
        std::normal_distribution<FT> distrib(0.0, 1.0);
        return clampRad(
            distrib(getRng()) / 3.0 * radius, radius);
    }
    Point sampleRing(const Point& base, FT radius) {
        FT angle = uniform(0.0, 2.0 * pi);
        return Point(base.x + cos(angle) * radius,
                     base.y + sin(angle) * radius);
    }
    Point sampleCircle(const Point& base, FT radius) {
        return sampleRing(base, uniform(0.0, radius));
    }
    Point sampleCircleGauss(const Point& base,
                            FT radius) {
        return sampleRing(base,
                          fabs(sampleGauss(radius)));
    }
    Point lerp(const Point& a, const Point& b, FT w) {
        return a * (1.0 - w) + b * w;
    }
    FT uniform(FT l, FT r) {
        std::uniform_real_distribution<FT> urd(l, r);
        return urd(getRng());
    }
    Point rlerp(const Point& a, const Point& b) {
        return lerp(a, b, uniform(0.0, 1.0));
    }
    bool accept(FT p) {
        std::uniform_real_distribution<FT> urd(0.0,
                                               1.0);
        return urd(getRng()) < p;
    }
    FT dot(const Point& a, const Point& b) {
        return a.x * b.x + a.y * b.y;
    }
    FT cross(const Point& a, const Point& b) {
        return a.x * b.y - a.y * b.x;
    }
}
