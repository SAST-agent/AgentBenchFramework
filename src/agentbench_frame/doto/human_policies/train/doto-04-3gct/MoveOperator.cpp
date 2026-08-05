#include "MoveOperator.hpp"
#include "Analyzer.hpp"
#include "Attack.hpp"
#include "Map.hpp"
#include "SafeOp.hpp"
#include <map>

PointApplyor advanceTo(const Point& base, const Point& dst) {
    return [=](PointSampler& sampler) {
        sampler.estimate(minv(prec(angle(base, dst), Math::pi / 36.0)))
            .optional(distance(dst) < dist(base, dst));
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
PointApplyor antiFollow(int id) {
    Point pos = getMyUnitPos(id), nxt = pos + estimate(id);
    return [nxt](PointSampler& sampler) { sampler.estimate(distance(nxt)); };
}
std::pair<Point, SourceLocation> calcDefaultMoveTo(int id, const Point& obj,
                                                   FT radius) {
    Point mp = getMyUnitPos(id);
    if(inMeteor(mp)) {
        info() << "unit " << id << " flee meteor" << std::endl;
        Point res = makeMoveBaseSampler(id)
                        .optional(!inMeteorFunc())
                        .estimate([=](const Point& p) {
                            FT mdis = CONST::explode_radius * 1.2;
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
    Point dst = path(id, obj, radius);
    Point res = makeMoveBaseSampler(id)
                    .request(!inMeteorFunc())
                    .apply(fleeFireball(id))
                    //.apply(keepDistance(id))
                    .apply(advanceTo(mp, dst))
                    .sample(1024, randomMove(id));
    return std::make_pair(res, SRCLOC);
}
Point defaultMoveTo(int id, const Point& obj, FT radius) {
    std::pair<Point, SourceLocation> mp = calcDefaultMoveTo(id, obj, radius);
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
std::map<int, Point> doSteal(const std::vector<int>& ids, int cid) {
    Point cp = crystalPos()(cid);
    std::map<int, Point> mps;
    /*
     if(ids.size() == 3 && close(ids)) {
         std::vector<int> hids = ids;
         std::sort(hids.begin(), hids.end(), [](int a, int b) {
             return getMyUnit(a).hp < getMyUnit(b).hp;
         });
         bool line = getMyUnit(hids.front()).hp > CONST::human_hp * 0.4;
         Point o0 = getMyUnitPos(hids[0]), o1 = getMyUnitPos(hids[1]),
               o2 = getMyUnitPos(hids[2]);
         Point base = defaultMoveTo(
             hids[0], cp, CONST::ball_radius, line ?
                 fabs(dist(o0, o1) + dist(o0, o2) - dist(o1, o2)) >
                     2.0 * CONST::human_velocity :
                 dist(o0, cp) < std::min(dist(o1, cp), dist(o2, cp)));
         mps[hids[0]] = base;
         Point p1, p2, dir = normalize(cp - base);
         const FT rad = 2.0 * CONST::fireball_radius;
         if(line) {
             Point ldir(dir.y, -dir.x);
             p1 = offset(base, ldir, rad);
             p2 = offset(base, -ldir, rad);
         } else {
             p1 = offset(base, Math::rotate(dir, Math::pi / 6.0), rad);
             p2 = offset(base, Math::rotate(dir, -Math::pi / 6.0), rad);
         }
         if(dist(getMyUnitPos(hids[1]), p1) > dist(getMyUnitPos(hids[1]), p2))
             std::swap(p1, p2);
         mps[hids[1]] = defaultMoveTo(hids[1], p1, 0.0, false);
         mps[hids[2]] = defaultMoveTo(hids[2], p2, 0.0, false);
     }
     */
    for(int id : ids)
        mps[id] = defaultMoveTo(id, cp, CONST::ball_radius);
    return mps;
}
std::map<int, Point> doNxtSteal(const std::vector<int>& ids, int cid) {
    std::map<int, Point> mps;
    for(int id : ids)
        mps[id] = defaultMoveTo(id, crystalInitPos()(cid), linkDis);
    return mps;
}
std::map<int, Point> doGuard(const std::vector<int>& ids) {
    std::map<int, Point> mps;
    for(int id : ids)
        mps[id] = defaultMoveTo(id, getMyTarget(), linkDis);
    return mps;
}
Point doBonus(int id, int bid) {
    Point cp = getMyUnitPos(id);
    Point bp = bonusPos()(bid);
    if(std::max(dist(cp, bp), minDist(bp, false)) < CONST::bonus_radius &&
       !inMeteor(cp)) {
        const Human& hu = getMyUnit(id);
        int obj = makeAttackBaseSampler()
                      .request(distance(bp) < CONST::bonus_radius)
                      .sample();
        Point op = obj != -1 ? getUnit(obj).position : Math::invalidPos;
        int activeFrame = obj != -1 ?
            std::max(3.0, ceil((dist(op, cp) - CONST::splash_radius) /
                               CONST::human_velocity)) :
            100.0;
        bool danger = !canFire(hu) ||
            (obj != -1 && (getUnit(obj).fire_time <= activeFrame ||
                           getUnit(obj).flash_time <= 1));
        FT odis = dist(op, bp);
        Point prePos = cp + estimate(getID(getMyFac(), id));
        Point mp =
            stableSampler(cp, canFlash(hu) ? CONST::flash_distance :
                                             CONST::human_velocity)
                .request(!isWallFunc())
                .request(!inMeteorFunc())
                .apply(obj != -1 ? PointApplyor{ [=](PointSampler& sampler) {
                    sampler.estimate(minv(prec(distance(bp), odis / 5.0)))
                        .estimate(minv(
                            prec(distanceCloseTo(op, CONST::splash_radius +
                                                     (danger ? -1.0 : 1.0)),
                                 0.1)));
                } } :
                                   PointApplyor{})
                .optional(!inFireFunc())
                .estimate(distance(prePos))
                .sample(1024, randomMove(id));
        doMove(id, mp, SRCLOC);
        return mp;
    } else
        return defaultMoveTo(id, bp, CONST::bonus_radius);
}
std::map<int, Point> doTrans(const std::vector<int>& ids, int mid) {
    std::map<int, Point> mps;
    Point tp = getMyTarget();
    /*
    if(ids.size() == 3 && close(ids)) {
        bool tri = minDist(getMyTarget(), false) < minDist(getMyTarget(), true);
        std::vector<int> ord = ids;
        std::iter_swap(ord.begin(), std::find(ord.begin(), ord.end(), mid));
        Point o0 = getMyUnitPos(ord[0]), o1 = getMyUnitPos(ord[1]),
              o2 = getMyUnitPos(ord[2]);
        mps[mid] = defaultMoveTo(
            mid, tp, CONST::target_radius, tri ?
                dist(o0, tp) < std::min(dist(o1, tp), dist(o2, tp)) :
                fabs(dist(o0, o1) + dist(o0, o2) - dist(o1, o2)) >
                    2.0 * CONST::human_velocity);
        Point base = mps[mid];
        Point p1, p2;
        const FT rad = 2.0 * CONST::fireball_radius;
        Point dir = normalize(getMyTarget() - base);
        if(tri) {
            p1 = offset(base, Math::rotate(dir, Math::pi / 6.0), rad);
            p2 = offset(base, Math::rotate(dir, -Math::pi / 6.0), rad);
        } else {
            p1 = offset(base, dir, rad);
            p2 = offset(base, -dir, rad);
        }
        if(dist(getMyUnitPos(ord[1]), p1) > dist(getMyUnitPos(ord[1]), p2))
            std::swap(p1, p2);
        mps[ord[1]] = defaultMoveTo(ord[1], p1, 0.0, false);
        mps[ord[2]] = defaultMoveTo(ord[2], p2, 0.0, false);
    }
    */
    mps[mid] = defaultMoveTo(mid, tp, CONST::target_radius);
    for(int id : ids) {
        if(id != mid) {
            int obj =
                fromContainer(listEnemys(mid))
                    .castPos(allUnitPos())
                    .request([](int id) { return getUnit(id).inv_time <= 0; })
                    .estimate(minv(distance(mps[mid])))
                    .sample();
            if(obj != -1)
                mps[id] = defaultMoveTo(id, getUnit(obj).position, closeDis);
            else
                mps[id] =
                    defaultMoveTo(id, getMyTarget(), CONST::target_radius);
        }
    }
    return mps;
}
std::map<int, Point> doGroup(const std::vector<int>& ids) {
    std::map<int, Point> mps;
    for(int id : ids) {
        int fid = id;
        FT mdis = 0.0;
        for(int p : ids) {
            FT cdis = dist(getMyUnitPos(p), getMyUnitPos(id));
            if(cdis > mdis)
                mdis = cdis, fid = p;
        }
        mps[id] = defaultMoveTo(id, getMyUnitPos(fid), linkDis);
    }
    return mps;
}
