#include "Analyzer.hpp"
#include "Attack.hpp"
#include "Map.hpp"
#include "Operator.hpp"
#include "SafeOp.hpp"
#include <map>
#include <memory>

enum class Theater { Trans, Guard, Bonus, Target };
PointApplyor advanceTo(const Point& base, const Point& dst) {
    return [=](PointSampler& sampler) {
        sampler.estimate(minv(prec(angle(base, dst), Math::pi / 18)))
            .optional(dot(dst, base) > 0.0);
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
        sampler
            .estimate(
                [=](const Point& p) { return evalSafeFacFlash(p); })
            .optional(!inFireFunc());
    };
}
PointApplyor fleeFireball(int id) {
    const Human& hu = getMyUnit(id);
    return canFlash(hu) ? fleeFireballFlash() :
                          fleeFireballMove(hu.position);
}
std::function<Point()> randomMove(int id) {
    return [id] {
        info() << "unit " << id << " random move" << std::endl;
        return circleSampler(getMyUnitPos(id),
                             canFlash(getMyUnit(id)) ?
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
    return [nxt](PointSampler& sampler) {
        sampler.estimate(distance(nxt));
    };
}
std::pair<Point, SourceLocation>
calcDefaultMoveTo(int id, const Point& obj, FT radius) {
    Point mp = getMyUnitPos(id);
    if(inMeteor(mp)) {
        info() << "unit " << id << " flee meteor" << std::endl;
        Point res =
            makeMoveBaseSampler(id)
                .optional(!inMeteorFunc())
                .estimate([=](const Point& p) {
                    FT mdis = CONST::explode_radius * 1.2;
                    for(const Meteor& me : getLogic().meteors) {
                        if(isAnti(me.from_number))
                            mdis =
                                std::min(mdis, dist(me.position, p));
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
                    .apply(advanceTo(mp, dst))
                    .apply(keepDistance(id))
                    .sample(1024, randomMove(id));
    return std::make_pair(res, SRCLOC);
}
Point defaultMoveTo(int id, const Point& obj, FT radius) {
    std::pair<Point, SourceLocation> mp =
        calcDefaultMoveTo(id, obj, radius);
    doMove(id, mp.first, mp.second);
    return mp.first;
}
void defaultAttack(int id, const Point& mp) {
    const Human& hu = getMyUnit(id);
    int obj =
        makeAttackBaseSampler()
            .request(distance(mp) < 2.0 * CONST::fireball_velocity *
                         CONST::frames_per_second)
            .optional(canFireToFunc(mp))
            .optional([](int id) {
                return getCrystal(getMyFac()).belong == id;
            })
            .estimate(minv<int>([mp](int id) {
                const Human& em = getUnit(id);
                return em.hp;
            }))
            .sample();
    bool fire = canFire(hu);
    if(obj != -1) {
        Point dst = getUnit(obj).position;
        if(fire) {
            Point fdst =
                makeFireBaseSampler(mp, dst, CONST::splash_radius)
                    .request(canFireToFunc(mp))
                    .sample(1024);
            if(fdst != Math::invalidPos) {
                SafeOp::fire(id, fdst, SRCLOC);
                fire = false;
            }
        }
        if(canMeteor(hu)) {
            Point mdst =
                makeMeteorBaseSampler(mp, dst, CONST::explode_radius)
                    .sample(1024);
            if(mdst != Math::invalidPos)
                SafeOp::meteor(id, mdst, SRCLOC);
        }
    }
    if(fire) {
        obj = makeAttackBaseSampler()
                  .optional(canFireToFunc(mp))
                  .estimate(minv(distance(mp)))
                  .sample();
        if(obj == -1)
            return;
        Point dir = getUnit(obj).position - mp;
        const FT ang = Math::pi / 6.0;
        Point dst = mp + Math::rotate(dir, Math::uniform(-ang, ang));
        if(canFireAt(mp, dst))
            SafeOp::fire(id, dst, SRCLOC);
    }
}
void doSteal(const std::vector<int>& ids, int cid) {
    std::map<int, Point> mps;
    for(int id : ids)
        mps[id] =
            defaultMoveTo(id, crystalPos()(cid), CONST::ball_radius);
}
void doNxtSteal(const std::vector<int>& ids, int cid) {
    for(int id : ids) {
        Point mp = defaultMoveTo(id, crystalInitPos()(cid), linkDis);
        defaultAttack(id, mp);
    }
}
void doGuard(const std::vector<int>& ids) {
    for(int id : ids) {
        Point mp = defaultMoveTo(id, getMyTarget(), linkDis);
        defaultAttack(id, mp);
    }
}
void doTrans(int id) {
    Point mp = defaultMoveTo(id, getMyTarget(), CONST::target_radius);
    int obj = makeAttackBaseSampler()
                  .optional(canFireToFunc(mp))
                  .estimate(minv(distance(mp)))
                  .sample();
    if(obj != -1) {
        const Human& hu = getMyUnit(id);
        const Human& em = getUnit(obj);
        if(canFire(hu)) {
            Point fobj = makeFireBaseSampler(mp, em.position,
                                             CONST::splash_radius)
                             .sample(1024);
            if(fobj != Math::invalidPos)
                SafeOp::fire(id, fobj, SRCLOC);
        }
        if(canMeteor(hu)) {
            Point off = em.position - mp;
            Point dir = estimate(obj);
            if(Math::dot(off, dir) < 0.0) {
                const FT delay =
                    CONST::human_velocity * CONST::meteor_delay / 2.0;
                FT ndis = dist(mp, em.position) - delay;
                if(fabs(ndis) < CONST::meteor_distance) {
                    Point noff = Math::scale(off, ndis);
                    SafeOp::meteor(id, mp + noff, SRCLOC);
                }
            }
        }
    }
}
void doBonus(int id, int bid) {
    Point cp = getMyUnitPos(id);
    Point bp = bonusPos()(bid);
    if(std::max(dist(cp, bp), minDist(bp, false)) <
           CONST::bonus_radius &&
       !inMeteor(cp)) {
        const Human& hu = getMyUnit(id);
        int obj = makeAttackBaseSampler()
                      .request(distance(bp) < CONST::bonus_radius)
                      .sample();
        Point op =
            obj != -1 ? getUnit(obj).position : Math::invalidPos;
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
                .apply(obj != -1 ?
                           PointApplyor{ [=](PointSampler& sampler) {
                               sampler
                                   .estimate(minv(prec(distance(bp),
                                                       odis / 5.0)))
                                   .estimate(minv(prec(
                                       distanceCloseTo(
                                           op, CONST::splash_radius +
                                               (danger ? -1.0 : 1.0)),
                                       0.1)));
                           } } :
                           PointApplyor{})
                .optional(!inFireFunc())
                .estimate(distance(prePos))
                .sample(1024, randomMove(id));
        doMove(id, mp, SRCLOC);
        if(obj != -1) {
            if(canFire(hu)) {
                //!!!
                SafeOp::fire(id, op, SRCLOC);
            }
            if(canMeteor(hu) &&
               dist(mp, bp) < CONST::meteor_distance &&
               getLogic().frame - getLastBonus(bid) >
                   CONST::bonus_time_low)
                SafeOp::meteor(id, bp, SRCLOC);
        }
    } else {
        Point mp = defaultMoveTo(id, bp, CONST::bonus_radius);
        defaultAttack(id, mp);
    }
}
void doHelp(const std::vector<int>& ids, int mid) {
    std::vector<int> enemys = listEnemys(mid);
    std::map<int, Point> mps;
    for(int id : ids)
        mps[id] = defaultMoveTo(id, getMyUnitPos(mid), linkDis);
    int obj =
        fromContainer(enemys)
            .castPos(allUnitPos())
            .estimate(minv(prec(distance(getMyUnitPos(mid)), vel)))
            .estimate(
                minv<int>([](int id) { return getUnit(id).hp; }))
            .sample();
    if(obj != -1) {
        const Human& em = getUnit(obj);
        for(int id : ids) {
            const Human& hu = getMyUnit(id);
            if(canFire(hu)) {
                Point pos = makeFireBaseSampler(mps[id], em.position,
                                                CONST::splash_radius)
                                .optional(canFireToFunc(mps[id]))
                                .sample(1024);
                if(pos != Math::invalidPos)
                    SafeOp::fire(id, pos, SRCLOC);
            }
        }
        if(syncMeteor(ids)) {
            Point dir = em.position - getMyUnitPos(mid);
            const FT delay =
                CONST::human_velocity * CONST::meteor_delay / 2.0;
            FT ndis = length(dir) - delay;
            dir = Math::scale(dir, ndis);
            Point midP = getMyUnitPos(mid) + dir;
            std::vector<Meteor> meteors;
            for(const Meteor& me : getLogic().meteors)
                if(getFac(me.from_number) == getMyFac())
                    meteors.push_back(me);
            for(int id : ids) {
                Point pos =
                    makeMeteorBaseSampler(mps[id])
                        .request(distance(midP) <
                                 CONST::explode_radius * 1.5)
                        .estimate(minv<Point>([&](const Point& p) {
                            FT res = 0.0;
                            for(const Meteor& me : meteors)
                                res += Math::sharedArea(
                                           me.position,
                                           CONST::explode_radius, p,
                                           CONST::explode_radius) *
                                    me.last_time;
                            return res;
                        }))
                        .sample(1024);
                if(pos != Math::invalidPos) {
                    meteors.push_back(
                        Meteor(pos.x, pos.y, CONST::meteor_delay, 0));
                    SafeOp::meteor(id, pos, SRCLOC);
                }
            }
        }
    } else {
        for(int id : ids)
            defaultAttack(id, mps[id]);
    }
}
void doGroup(const std::vector<int>& ids) {
    for(int id : ids) {
        int fid = id;
        FT mdis = 0.0;
        for(int p : ids) {
            FT cdis = dist(getMyUnitPos(p), getMyUnitPos(id));
            if(cdis > mdis)
                mdis = cdis, fid = p;
        }
        Point mp = defaultMoveTo(id, getMyUnitPos(fid), linkDis);
        defaultAttack(id, mp);
    }
}
