#pragma once
#include "Sampler.hpp"
const FT fireRatio = CONST::human_velocity / CONST::fireball_velocity;
DiscreteSampler makeAttackBaseSampler();
PointSampler makeFireBaseSampler(const Point& src, const Point& dst,
                                 FT radius);
DiscreteSampler makeFireBaseSampler(const Point& src);
PointSampler makeMeteorBaseSampler(const Point& src);
PointSampler makeMeteorBaseSampler(const Point& src, const Point& dst,
                                   FT radius);
bool syncFire(const std::vector<int>& ids);
bool syncMeteor(const std::vector<int>& ids);
