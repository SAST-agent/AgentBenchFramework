#include "Attack.hpp"
#include "Map.hpp"
#include "Math.hpp"
DiscreteSampler makeAttackBaseSampler() {
    return DiscreteSampler(getLogic().humans.size())
        .castPos(allUnitPos())
        .request(isEnemy())
        .request(isAlive())
        .request([](int id) { return getUnit(id).inv_time <= 0; });
}
PointSampler makeFireBaseSampler(const Point& src, const Point& dst,
                                 FT radius) {
    Point off = normalize(dst - src);
    return PointSampler([=] {
               FT rad = Math::uniform(0.0, radius);
               FT ang = Math::uniform(0.0, Math::pi);
               return dst + Math::rotate(off, ang) * rad;
           })
        .request(canFireAtFunc(src));
}
PointSampler makeMeteorBaseSampler(const Point& src) {
    return circleSampler(src, CONST::meteor_distance);
}
PointSampler makeMeteorBaseSampler(const Point& src, const Point& dst,
                                   FT radius) {
    return circleSampler(dst, radius)
        .request(distance(src) < CONST::meteor_distance);
}
bool syncFire(const std::vector<int>& ids) {
    return std::all_of(ids.begin(), ids.end(),
                       [](int id) { return canFire(getMyUnit(id)); });
}
bool syncMeteor(const std::vector<int>& ids) {
    return std::all_of(ids.begin(), ids.end(),
                       [](int id) { return canMeteor(getMyUnit(id)); });
}
DiscreteSampler makeFireBaseSampler(const Point& src) {
    const int angCnt = 360;
    return DiscreteSampler(angCnt)
        .castPos([=](int id) {
            FT ang = Math::twoPi * id / angCnt;
            return src + Point(cos(ang), sin(ang)) * CONST::splash_radius;
        })
        .request(!isWallFunc());
}
