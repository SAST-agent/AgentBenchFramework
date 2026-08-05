#include "MoveOperator.hpp"
#include "Analyzer.hpp"
#include "Attack.hpp"
#include "Map.hpp"
#include "SafeOp.hpp"
#include <map>

PointApplyor advanceTo(const Point&, const Point& dst) {
    return [=](PointSampler& sampler) {
        sampler.request([=](const Point&) { return dst != Math::invalidPos; })
            .estimate(minv(distance(dst)));
    };
}
PointApplyor keepDistance(int id) {
    return [id](PointSampler& sampler) {
        sampler.estimate([id](const Point& cp) {
            FT minv = keep;
            for(int i = 0; i < humanSiz(); ++i) {
                if(i == id || getMyUnit(i).hp <= 0)
                    continue;
                minv = std::min(minv, dist(cp, getMyUnitPos(i)));
            }
            return minv;
        });
    };
}
PointApplyor fleeFireballMove(const Point& mp) {
    auto pre = preSafeFac(mp);
    return [=](PointSampler& sampler) {
        sampler
            .estimate([=](const Point& p) {
                return evalSafeFacMove(mp, normalize(p - mp), pre);
            })
            .optional(!inFireFunc());
    };
}
PointApplyor fleeFireballFlash() {
    return [](PointSampler& sampler) {
        sampler.estimate([=](const Point& p) { return evalSafeFacFlash(p); })
            .optional(!inFireFunc());
    };
}
PointApplyor fleeFireball(int id) {
    const Human& hu = getMyUnit(id);
    return canFlash(hu) ? fleeFireballFlash() : fleeFireballMove(hu.position);
}
std::function<Point()> randomMove(int id) {
    return [id] {
        info() << "unit " << id << " random move" << std::endl;
        return circleSampler(getMyUnitPos(id), canFlash(getMyUnit(id)) ?
                                 CONST::flash_distance :
                                 CONST::human_velocity)
            .request(!isWallFunc())
            .optional(!inMeteorFunc())
            .optional(!inFireFunc())
            .sample(2048, [id] {
                ERROR << "unit " << id << " no move" << std::endl;
                return getMyUnitPos(id);
            });
    };
}
PointApplyor fleeEnemy(FT maxRad) {
    return [=](PointSampler& sampler) {
        sampler.estimate(prec<Point>(
            [=](const Point& pos) {
                FT mdis = maxRad;
                int hsiz = getLogic().humans.size();
                for(int i = 0; i < hsiz; ++i)
                    if(getFac(i) != getMyFac())
                        mdis = std::min(mdis, dist(getUnit(i).position, pos));
                return mdis;
            },
            minTrans));
    };
}
PointApplyor flashBack(int id) {
    if(!canFlash(getMyUnit(id)))
        return {};
    Point cp = getMyUnitPos(id);
    int obj = makeAttackBaseSampler()
                  .request(distance(cp) < CONST::flash_distance)
                  .estimate(minv(distance(cp)))
                  .sample();
    if(obj != -1) {
        Point op = getUnit(obj).position;
        return [=](PointSampler& sampler) {
            return sampler.optional((dot(op, cp) < 0.0) &
                                    (distance(op) > CONST::splash_radius));
        };
    }
    return {};
}
std::pair<Point, SourceLocation>
calcDefaultMoveTo(int id, const Point& obj, FT radius, FT minRad, bool force) {
    Point mp = getMyUnitPos(id);
    if(inMeteor(mp)) {
        info() << "unit " << id << " flee meteor" << std::endl;
        Point res = makeMoveBaseSampler(id)
                        .optional(!inMeteorFunc())
                        .estimate([=](const Point& p) {
                            FT mdis = CONST::explode_radius + Math::hurtBias;
                            for(const Meteor& me : getLogic().meteors) {
                                if(isAnti(me.from_number))
                                    mdis = std::min(mdis, dist(me.position, p));
                            }
                            return mdis;
                        })
                        .estimate(minv(distance(mp)))
                        .sample(1024, randomMove(id));
        return std::make_pair(res, SRCLOC);
    }
    bool flee = !force && getMyUnit(id).hp < CONST::human_hp * 0.2;
    Point dst = path(id, obj, minRad, radius);
    Point res =
        makeMoveBaseSampler(id)
            .request(!inMeteorFunc())
            .apply(fleeFireball(id))
            .apply(fleeEnemy(flee ? 10 * CONST::fireball_velocity : minRad))
            .apply(flashBack(id))
            .apply(Math::accept(0.1) ? keepDistance(id) : PointApplyor{})
            .apply(advanceTo(mp, dst))
            .sample(768, randomMove(id));
    return std::make_pair(res, SRCLOC);
}
Point defaultMoveTo(int id, const Point& obj, FT radius, FT minRad = 0.0,
                    bool force = false) {
    std::pair<Point, SourceLocation> mp =
        calcDefaultMoveTo(id, obj, radius, minRad, force);
    doMove(id, mp.first, mp.second);
    return mp.first;
}
bool close(const std::vector<int>& ids) {
    for(int a : ids)
        for(int b : ids)
            if(dist(getMyUnitPos(a), getMyUnitPos(b)) > closeDis)
                return false;
    return true;
}
Point offset(const Point& base, const Point& dir, FT maxDis) {
    Point last = base;
    for(FT rad = 0.1; rad < maxDis; rad += 0.1) {
        Point cur = base + dir * rad;
        if(isWall(cur))
            break;
        last = cur;
    }
    return last;
}
Point closestEnemy(const Point& p) {
    int obj = DiscreteSampler(getLogic().humans.size())
                  .castPos(ghostPos(p))
                  .request(isEnemy())
                  .request([](const Point& cp) {
                      for(Point bp : getMap().bonus_places)
                          if(dist(bp, cp) < 2.0 * CONST::bonus_radius)
                              return false;
                      return true;
                  })
                  .estimate(minv(prec(distance(p), linkDis)))
                  .optional(unitHP() < CONST::human_hp * 0.2)
                  .sample();
    if(obj == -1)
        return p;
    if(getUnit(obj).hp > 0)
        return getUnit(obj).position;
    return getMap().birth_places[getFac(obj)][getNum(obj)];
}
std::map<int, Point> doStealImpl(const std::vector<int>& ids, const Point& cp) {
    int hid = fromContainer(ids)
                  .castPos(myUnitPos())
                  .estimate(minv(prec(distance(cp), vel)))
                  .estimate([](int id) { return getMyUnit(id).hp; })
                  .sample();
    std::map<int, Point> mps;
    mps[hid] = defaultMoveTo(hid, cp, CONST::ball_radius, 0.0, true);
    Point dst = closestEnemy(mps[hid]);
    for(int id : ids)
        if(id != hid)
            mps[id] = defaultMoveTo(id, dst, closeDis, 20.0);
    return mps;
}
std::map<int, Point> doSteal(const std::vector<int>& ids, int cid) {
    Point cp = crystalPos()(cid);
    return doStealImpl(ids, cp);
}
std::map<int, Point> doNxtSteal(const std::vector<int>& ids, int cid) {
    Point cp = crystalInitPos()(cid);
    if(getCrystal(getMyFac()).belong != -1)
        cp = getCrystal(getMyFac()).position;
    return doStealImpl(ids, cp);
}
Point doBonusImpl(int id, int bid, int obj) {
    info() << "do bonus" << std::endl;
    Point cp = getMyUnitPos(id);
    Point bp = bonusPos()(bid);
    const Human& hu = getMyUnit(id);
    Point op = getUnit(obj).position;
    int returnFrame =
        1 + ceil((dist(op, cp) - CONST::splash_radius) / CONST::human_velocity);
    int fireFrame = ceil((3.6 - dist(op, cp) / CONST::human_velocity));
    bool danger = getMyUnit(id).fire_time >= fireFrame ||
        getUnit(obj).fire_time <= returnFrame;
    bool will = getLogic().frame - getLastBonus(bid) >
        CONST::bonus_time_low - dist(cp, bp) / CONST::human_velocity * 1.2;
    FT limit = dist(op, bp) - CONST::human_velocity;
    Point hot = dist(bp, op) < CONST::human_velocity ?
        bp :
        bp + Math::scale(op - bp, dist(bp, op) - CONST::human_velocity);
    Point mp = stableSampler(cp, canFlash(hu) ? CONST::flash_distance :
                                                CONST::human_velocity)
                   .interest({ hot, bp })
                   .request(!isWallFunc())
                   .request(!inMeteorFunc())
                   .optional(!inFireFunc())
                   //.apply(fleeFireball(id))
                   //.optional(distance(op) <
                   //          CONST::splash_radius - CONST::human_velocity)
                   .estimate(minv(distance(bp)))
                   .sample(1024, randomMove(id));
    doMove(id, mp, SRCLOC);
    return mp;
}
Point doBonus(int id, int bid) {
    Point cp = getMyUnitPos(id);
    Point bp = bonusPos()(bid);
    int obj = makeAttackBaseSampler()
                  .request(distance(bp) < CONST::bonus_radius)
                  .sample();
    if(obj != -1 &&
       std::max(dist(cp, bp), dist(bp, getUnit(obj).position)) <
           CONST::bonus_radius &&
       !inMeteor(cp) &&
       getLogic().scores[getMyFac()] > getLogic().scores[getMyFac() ^ 1])
        return doBonusImpl(id, bid, obj);
    else
        return defaultMoveTo(id, bp, CONST::bonus_radius, 0.0, true);
}
std::map<int, Point> doTrans(const std::vector<int>& ids, int mid) {
    std::map<int, Point> mps;
    Point tp = getMyTarget();
    if(!close(ids) && (minDist(getMyUnitPos(mid), false) < closeDis &&
                       (isLocked(mid, getMyTarget()) ||
                        getMyUnit(mid).hp < CONST::human_hp * 0.2))) {
        Point dst = path(mid, getMyTarget(), 0.0, CONST::target_radius);
        Point pos = makeMoveBaseSampler(mid)
                        .estimate(minv(distance(dst)))
                        .sample(1024, randomMove(mid));
        doMove(mid, pos, SRCLOC);
        mps[mid] = pos;
    } else
        mps[mid] = defaultMoveTo(mid, tp, CONST::target_radius, 0.0, true);
    Point dst = closestEnemy(mps[mid]);
    for(int id : ids)
        if(id != mid)
            mps[id] = defaultMoveTo(id, dst, closeDis, 20.0);
    return mps;
}
