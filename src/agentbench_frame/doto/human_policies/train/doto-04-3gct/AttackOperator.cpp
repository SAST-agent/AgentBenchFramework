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
        sampler
            .optional(
                [](int id) { return getCrystal(getMyFac()).belong == id; })
            .estimate(minv(prec(distance(mid), CONST::fireball_velocity * 2.0)))
            .estimate(minv<int>([](int id) { return getUnit(id).hp; }));
    };
}
void defaultFire(int id, const Point& cp) {
    int obj = makeAttackBaseSampler()
                  .request(canFireToFunc(cp))
                  .apply(defaultChoose(cp))
                  .sample();
    if(obj != -1) {
        Point odst = getUnit(obj).position;
        Point fdst = makeFireBaseSampler(cp, odst, CONST::splash_radius)
                         .request(canFireToFunc(cp))
                         .sample(1024);
        if(fdst != Math::invalidPos) {
            SafeOp::fire(id, cp, fdst, SRCLOC);
            return;
        }
    }
    Point dst = maxFire(cp);
    if(canFireAt(cp, dst))
        SafeOp::fire(id, cp, dst, SRCLOC);
}
void doGuard(const std::set<int>& myu, const std::set<int>& emu, bool isStatic,
             const Point& mid, std::map<int, Point>& nxtp) {
    int obj = fromContainer(emu)
                  .castPos(allUnitPos())
                  .apply(defaultChoose(mid))
                  .sample();
    if(obj == -1)
        obj = *emu.begin();
    Point odst = getUnit(obj).position;
    for(int id : myu) {
        Point cp = nxtp[id];
        const Human& hu = getMyUnit(id);
        if(canFire(hu)) {
            Point fdst = makeFireBaseSampler(cp, odst, CONST::splash_radius)
                             .request(canFireToFunc(cp))
                             .sample(1024);
            if(fdst != Math::invalidPos)
                SafeOp::fire(id, cp, fdst, SRCLOC);
            else
                defaultFire(id, cp);
        }
        if(canMeteor(hu)) {
            if(isStatic) {
                Point mdst =
                    makeMeteorBaseSampler(cp, mid, CONST::explode_radius)
                        .sample(1024);
                if(mdst != Math::invalidPos) {
                    SafeOp::meteor(id, mdst, SRCLOC);
                    continue;
                }
            }
            for(int obj : emu) {
                Point mdst = makeMeteorBaseSampler(cp, getUnit(obj).position,
                                                   CONST::explode_radius)
                                 .sample(1024);
                if(mdst != Math::invalidPos) {
                    SafeOp::meteor(id, mdst, SRCLOC);
                    break;
                }
            }
        }
    }
}
void doBonus(int bid, int cid, int eid, std::map<int, Point>& nxtp) {
    const Human& hu = getMyUnit(cid);
    Point mp = nxtp[cid];
    Point op = getUnit(eid).position;
    Point bp = getMap().bonus_places[bid];
    if(canFire(hu)) {
        if(dist(mp, op) > CONST::splash_radius)
            SafeOp::fire(cid, mp, op, SRCLOC);
    }
    if(canMeteor(hu) && dist(mp, bp) < CONST::meteor_distance &&
       getLogic().frame - getLastBonus(bid) > CONST::bonus_time_low)
        SafeOp::meteor(cid, bp, SRCLOC);
}
void doPlain(const std::set<int>& myu, const std::set<int>& emu,
             std::map<int, Point>& nxtp) {
    /*
    Point mm = midPoint(myu, myUnitPos());
    Point me = midPoint(emu, allUnitPos());
    std::set<int> rem = myu;
    if(dist(mm, me) > CONST::fireball_velocity * CONST::frames_per_second) {
        Point baseDir = me - mm;
        Point dir = baseDir;
        FT mav = Math::inf;
        for(int id : myu) {
            Point cdir = maxFire(nxtp[id]) - nxtp[id];
            if(Math::dot(cdir, baseDir) < 0.0)
                cdir = -cdir;
            FT cav = Math::angle(cdir, baseDir);
            if(cav < mav)
                mav = cav, dir = cdir;
        }
        dir = Math::scale(dir, CONST::splash_radius);
        std::vector<int> sync;
        for(int id : myu)
            if(!isWall(nxtp[id] + dir))
                sync.push_back(id), rem.erase(id);
        if(syncFire(sync)) {
            for(int id : sync)
                SafeOp::fire(id, nxtp[id], nxtp[id] + dir, SRCLOC);
        }
    }
    */
    for(int id : myu) {
        const Human& hu = getMyUnit(id);
        if(canFire(hu))
            defaultFire(id, nxtp[id]);
    }
    for(int id : myu) {
        const Human& hu = getMyUnit(id);
        if(canMeteor(hu)) {
            Point cp = nxtp[id];
            for(int obj : emu) {
                Point mdst = makeMeteorBaseSampler(cp, getUnit(obj).position,
                                                   CONST::explode_radius)
                                 .sample(1024);
                if(mdst != Math::invalidPos) {
                    SafeOp::meteor(id, mdst, SRCLOC);
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

    int mid = -1;
    for(int i = 0; i < facSiz(); ++i) {
        int bel = getCrystal(i).belong;
        if(bel != -1 && getFac(bel) == getMyFac()) {
            mid = getNum(bel);
            break;
        }
    }
    if(mid != -1) {
        Point mp = nxtp[mid];
        std::set<int> cim = split(mp, linkDis, idleM, myUnitPos()), cie;
        for(int id : idleE)
            if(canFireTo(getUnit(id).position, mp))
                cie.insert(id);
        if(cim.size() && cie.size()) {
            remove(idleM, cim);
            remove(idleE, cie);
            doGuard(cim, cie, false, mp, nxtp);
        }
    }

    Point cp = getCrystal(getMyFac()).position;
    {
        std::set<int> cim =
            split(cp, 20.0 * CONST::ball_radius, idleM, myUnitPos());
        std::set<int> cie =
            split(cp, 20.0 * CONST::ball_radius, idleE, allUnitPos());
        if(cim.size() && cie.size()) {
            remove(idleM, cim);
            remove(idleE, cie);
            doGuard(cim, cie, getCrystal(getMyFac()).belong == -1, cp, nxtp);
        }
    }
    Point cip = getMap().crystal_places[getMyFac()];
    if(cp != cip) {
        std::set<int> cim =
            split(cip, 20.0 * CONST::ball_radius, idleM, myUnitPos());
        std::set<int> cie =
            split(cip, 20.0 * CONST::ball_radius, idleE, allUnitPos());
        if(cim.size() && cie.size()) {
            remove(idleM, cim);
            remove(idleE, cie);
            doGuard(cim, cie, true, cip, nxtp);
        }
    }
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
            doPlain(gm, groupE[mi], nxtp);
    }
}
