#include "AttackOperator.hpp"
#include "Analyzer.hpp"
#include "Attack.hpp"
#include "SafeOp.hpp"
#include "Sampler.hpp"
#include <set>
Point midPoint(const std::set<int>& ids, const Int2PointFunc& cast) {
    Point mid(0.0, 0.0);
    for(int id : ids)
        mid = mid + cast(id);
    return mid * (1.0 / ids.size());
}
void remove(std::set<int>& old, const std::set<int>& del) {
    for(int id : del)
        old.erase(id);
}

// enum class Theater { Trans, Guard, Bonus, Crystal, Other };
DiscreteApplyor defaultChoose(const Point& mid) {
    return [=](DiscreteSampler& sampler) {
        sampler.optional(distance(mid) < CONST::fireball_velocity * 10)
            .estimate(minv(unitHP()))
            .estimate(minv(prec(distance(mid), CONST::fireball_velocity * 2.0)))
            .optional([](int id) { return !canFlash(getUnit(id)); });
    };
}
PointEvalFunc fireCnt(const Point& cp) {
    return [cp](const Point& dst) {
        int hsiz = getLogic().humans.size();
        Point dir = Math::scale(dst - cp, CONST::fireball_velocity);
        for(int i = 0; i < 5; ++i) {
            Point fp = cp + dir;
            bool hit = isWall(fp);
            if(!hit) {
                for(int j = 0; j < hsiz; ++j) {
                    if(getFac(j) == getMyFac() || getUnit(j).hp < 0)
                        continue;
                    if(dist(fp, getUnit(j).position) < CONST::fireball_radius) {
                        hit = true;
                        break;
                    }
                }
            }
            if(hit) {
                FT cnt = 0.0;
                for(int j = 0; j < hsiz; ++j) {
                    if(getFac(j) == getMyFac() || getUnit(j).hp < 0)
                        continue;
                    if(dist(fp, getUnit(j).position) < CONST::splash_radius)
                        ++cnt;
                }
                return cnt;
            }
        }
        return 0.0;
    };
}
PointEvalFunc fireDis(const Point& cp) {
    return [cp](const Point& dst) {
        int hsiz = getLogic().humans.size();
        FT sum = 0.0;
        for(int i = 0; i < hsiz; ++i) {
            if(getFac(i) == getMyFac() || getUnit(i).hp < 0)
                continue;
            FT dis = ptoldist(getUnit(i).position, Lineseg(cp, dst));
            if(dis < CONST::splash_radius)
                sum += 1e5 - dis;
        }
        return sum;
    };
}
const FT fireRad = CONST::fireball_radius;
void defaultFire(int id, const Point& cp) {
    DiscreteSampler sampler = makeAttackBaseSampler()
                                  .optional(canFireToFunc(cp))
                                  .apply(defaultChoose(cp));
    int obj = sampler.sample();
    if(obj != -1) {
        Point odst = sampler.castToPos(obj);
        Point fdst = makeFireBaseSampler(cp, odst, obj, fireRad)
                         .request(canFireToFunc(cp))
                         .estimate(fireDis(cp))
                         .sample(1024);
        if(fdst != Math::invalidPos) {
            SafeOp::fire(id, cp, fdst, SRCLOC);
            return;
        } else {
            Point dir = scanCorner(cp, normalize(odst - cp));
            if(canFireAt(cp, cp + dir)) {
                info() << "scan corner" << std::endl;
                SafeOp::fire(id, cp, cp + dir, SRCLOC);
                return;
            }
        }
    }
    Point dst = maxFire(cp);
    if(canFireAt(cp, dst)) {
        info() << "max fire" << std::endl;
        SafeOp::fire(id, cp, dst, SRCLOC);
    }
}
void doBonus(int bid, int cid, int eid, std::map<int, Point>& nxtp) {
    const Human& hu = getMyUnit(cid);
    Point mp = nxtp[cid];
    Point op = getUnit(eid).position;
    Point bp = getMap().bonus_places[bid];
    if(canFire(hu)) {
        FT cdis = dist(mp, op);
        FT sharedA = Math::sharedArea(
            Point(0.0, 2.0 * CONST::splash_radius), CONST::splash_radius,
            Point(0.0, cdis - CONST::human_velocity), CONST::human_velocity);
        FT sharedB = Math::sharedArea(
            Point(0.0, 2.0 * CONST::splash_radius), CONST::splash_radius,
            Point(0.0, cdis + CONST::human_velocity), CONST::human_velocity);
        const FT base =
            Math::pi * CONST::human_velocity * CONST::human_velocity;
        if(cdis > 2.0 * CONST::splash_radius ||
           Math::accept(std::min(sharedA, sharedB) / base))
            SafeOp::fire(cid, mp, op, SRCLOC);
    }
    int delta = getLogic().frame - getLastBonus(bid);
    if(canMeteor(hu) && dist(mp, bp) < CONST::meteor_distance &&
       delta > CONST::bonus_time_low)
        SafeOp::meteor(cid, bp, SRCLOC);
}
void doPlain(const std::set<int>& myu, const std::set<int>& emu,
             std::map<int, Point>& nxtp, std::vector<Meteor>& mes) {
    for(int id : myu) {
        const Human& hu = getMyUnit(id);
        if(canFire(hu))
            defaultFire(id, nxtp[id]);
    }
    for(int id : myu) {
        const Human& hu = getMyUnit(id);
        Point cp = nxtp[id];
        if(canMeteor(hu)) {
            for(int obj : emu) {
                const int predict = 20;
                Point np = getUnit(obj).position + estimate(obj) * predict;
                Point mp =
                    makeMeteorBaseSampler(cp, np, CONST::explode_radius)
                        .estimate(minv<Point>([&](const Point& pos) {
                            FT sum = 0.0;
                            for(const Meteor& me : mes) {
                                sum += Math::sharedArea(
                                           me.position, CONST::explode_radius,
                                           pos, CONST::explode_radius) *
                                    me.last_time;
                            }
                            return sum;
                        }))
                        .sample(1024);
                if(mp != Math::invalidPos) {
                    SafeOp::meteor(id, mp, SRCLOC);
                    mes.emplace_back(mp.x, mp.y, CONST::meteor_delay, 0);
                    break;
                }
            }
        }
    }
}

std::set<int> split(const Point& mid, FT radius, std::set<int>& ids,
                    const Int2PointFunc& cast) {
    std::set<int> res;
    for(int id : ids)
        if(dist(cast(id), mid) < radius)
            res.insert(id);
    return res;
}
std::vector<std::set<int>> group(const std::set<int>& ids,
                                 const Int2PointFunc& cast) {
    std::vector<std::set<int>> res;
    for(int id : ids) {
        Point cp = cast(id);
        bool newGroup = true;
        for(std::set<int>& g : res) {
            bool flag = true;
            for(int mate : g)
                if(dist(cp, cast(mate)) > closeDis) {
                    flag = false;
                    break;
                }
            if(flag) {
                g.insert(id);
                newGroup = false;
                break;
            }
        }
        if(newGroup)
            res.push_back({ id });
    }
    return res;
}
void doAttack(std::map<int, Point>& nxtp) {
    std::set<int> idleM, idleE;
    for(int i = 0; i < humanSiz(); ++i)
        if(getMyUnit(i).hp > 0)
            idleM.insert(i);
    int hsiz = getLogic().humans.size();
    for(int i = 0; i < hsiz; ++i)
        if(getFac(i) != getMyFac() && getUnit(i).hp > 0)
            idleE.insert(i);

    for(size_t i = 0; i < getLogic().bonus.size(); ++i) {
        Point bp = getMap().bonus_places[i];
        int cid = -1, eid = -1;
        for(int id : idleM)
            if(dist(bp, nxtp[id]) < CONST::bonus_radius) {
                cid = id;
                break;
            }
        for(int id : idleE)
            if(dist(bp, getUnit(id).position) < CONST::bonus_radius) {
                eid = id;
                break;
            }
        if(cid != -1 && eid != -1) {
            idleM.erase(cid);
            idleE.erase(eid);
            doBonus(i, cid, eid, nxtp);
        }
    }

    std::vector<Meteor> mes;
    for(const Meteor& me : getLogic().meteors)
        if(getFac(me.from_number) == getMyFac())
            mes.push_back(me);

    std::vector<std::set<int>> groupM = group(idleM, myUnitPos());
    std::vector<std::set<int>> groupE = group(idleE, allUnitPos());
    for(const std::set<int>& gm : groupM) {
        Point mm = midPoint(gm, myUnitPos());
        FT mdis = Math::inf;
        int mi = -1;
        for(size_t i = 0; i < groupE.size(); ++i) {
            Point me = midPoint(groupE[i], allUnitPos());
            FT cdis = dist(me, mm);
            if(cdis < mdis)
                mdis = cdis, mi = i;
        }
        if(mi != -1)
            doPlain(gm, groupE[mi], nxtp, mes);
    }
}
