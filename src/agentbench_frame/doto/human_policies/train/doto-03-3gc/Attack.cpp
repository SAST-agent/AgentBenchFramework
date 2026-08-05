#include "Attack.hpp"
#include "Map.hpp"
#include "Math.hpp"
#include <map>
DiscreteSampler makeAttackBaseSampler() {
    return DiscreteSampler(getLogic().humans.size())
        .castPos(estiUnitPos())
        .request(isEnemy())
        .request(isAlive())
        .request([](int id) { return getUnit(id).inv_time <= 0; });
}
PointSampler makeFireBaseSampler(const Point& src, const Point& dst, int id,
                                 FT radius) {
    static std::map<int, bool> semi;
    semi[id] ^= 1;
    bool ns = semi[id];
    Point off = normalize(dst - src);
    return PointSampler([=](int, int) {
               static uint32_t index = 0;
               ++index;
               FT rad = Math::halton3(index) * radius;
               FT ang = Math::halton7(index) * Math::pi;
               return dst + Math::rotate(off, ns ? -ang : ang) * rad;
           })
        .hotRequest(distance(dst) < radius)
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
