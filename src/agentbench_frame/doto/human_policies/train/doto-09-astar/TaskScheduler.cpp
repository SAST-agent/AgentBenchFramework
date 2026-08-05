#include "TaskScheduler.hpp"
#include "Map.hpp"
#include <algorithm>
#include <set>
FT evalCrystal(const Crystal& ref) {
    FT mdisA = 1e20, mdisB = 1e20;
    for(int i = 0; i < facSiz(); ++i) {
        FT& val = (i == getMyFac() ? mdisA : mdisB);
        for(int j = 0; j < humanSiz(); ++j) {
            if(getUnit(i, j).hp > 0) {
                val = std::min(
                    val, MapHelper::estimateDis(
                             ref.position,
                             getUnit(i, j).position));
            }
        }
    }
    FT res = mdisB - mdisA;
    if(ref.faction == getMyFac())
        return res;
    if(ref.belong != -1)
        res -= panic *
            (getUnit(ref.belong).inv_time + 10);
    return res - MapHelper::estimateDis(ref.position,
                                        getMyTarget());
}
void moveSome(std::set<int>& idle, const Point& dst,
              FT radius, int sumHP, MoveOp& mop) {
    std::vector<int> team(idle.begin(), idle.end());
    std::sort(
        team.begin(), team.end(), [&](int a, int b) {
            auto esti = [&](int x) {
                return getMyUnit(x).hp * 2 -
                    MapHelper::estimateDis(
                           getMyUnit(x).position, dst);
            };
            return esti(a) > esti(b);
        });
    int chp = 0;
    for(int id : team) {
        if(chp >= sumHP)
            break;
        idle.erase(id);
        mop.id.push_back(id);
        chp += getMyUnit(id).hp;
    }
    mop.dst = dst;
    mop.radius = radius;
}
void trySteal(std::set<int>& idle, MoveOp& mop) {
    int shp = 0;
    for(int id : idle)
        shp += getMyUnit(id).hp;
    int stealHP = std::max(shp * 3 / 5, 200);
    if(shp < stealHP)
        return;
    FT mval = -1e20;
    int id = -1;
    for(int i = 0; i < facSiz(); ++i)
        if(getCrystal(i).faction != getMyFac()) {
            FT val = evalCrystal(getCrystal(i));
            if(val > mval) {
                mval = val;
                id = i;
            }
        }
    if(id != -1) {
        const Crystal& ref = getCrystal(id);
        moveSome(idle, ref.position,
                 CONST::ball_radius, stealHP, mop);
    }
}
int countUnits(const Point& v, FT radius, bool self) {
    int hsiz = getLogic().humans.size(), res = 0;
    for(int i = 0; i < hsiz; ++i)
        if(getUnit(i).hp > 0 &&
           (self || getFac(i) != getMyFac()) &&
           dist(v, getUnit(i).position) <= radius)
            res += getUnit(i).hp;
    return res;
}
std::vector<MoveOp>
calcTask(const std::vector<Group>& g,
         std::vector<int>& fa, int& mid) {
    std::set<int> idle;
    for(int i = 0; i < humanSiz(); ++i)
        if(getMyUnit(i).hp > 0)
            idle.insert(i);
    // steal/help/trans
    mid = -1;
    Point idleP = getMyTarget();
    for(int i = 0; i < facSiz(); ++i) {
        if(getCrystal(i).belong != -1 &&
           getFac(getCrystal(i).belong) ==
               getMyFac()) {
            mid = getNum(getCrystal(i).belong);
        }
        if(getCrystal(i).faction == getMyFac())
            idleP = getCrystal(i).position;
    }
    std::vector<MoveOp> mops;
    if(mid == -1 || !idle.count(mid)) {
        // steal
        MoveOp mop;
        trySteal(idle, mop);
        if(mop.id.size()) {
            idleP = mop.dst;
            mop.taskName = "steal";
            mops.push_back(mop);
        }
    } else {
        idleP = getMyUnit(mid).position;
        // trans and help
        {
            MoveOp mop;
            idle.erase(mid);
            mop.id.push_back(mid);
            mop.dst = getMyTarget();
            mop.radius = CONST::target_radius;
            mop.taskName = "trans";
            mops.push_back(mop);
        }
        {
            const std::vector<int>& ems =
                g[findFa(fa, getID(getMyFac(), mid))]
                    .emUnit;
            int shp = 0;
            for(int id : ems)
                shp += getUnit(id).hp;
            int helpHP = std::max(120, 80 + shp);
            MoveOp mop;
            moveSome(idle, getMyUnit(mid).position,
                     CONST::explode_radius * 3.0,
                     helpHP, mop);
            mop.taskName = "help";
            mops.push_back(mop);
        }
    }
    // intercept
    {
        for(const Crystal& cry : getLogic().crystal) {
            if(cry.faction == getMyFac()) {
                FT val = evalCrystal(cry);
                if(val < 0.0) {
                    MoveOp mop;
                    moveSome(idle, cry.position, 0.0,
                             countUnits(cry.position,
                                        linkDis,
                                        false) +
                                 50,
                             mop);
                    mops.push_back(mop);
                }
            }
        }
    }
    // idle
    {
        std::set<size_t> empBonus;
        for(size_t i = 0;
            i < getMap().bonus_places.size(); ++i)
            empBonus.insert(i);
        for(int uid : idle) {
            int id = -1;
            FT mval = 1e20;
            const Human& hu = getMyUnit(uid);
            auto esti = [&](const Point& b) {
                return MapHelper::estimateDis(
                    hu.position, b);
            };
            for(size_t i : empBonus) {
                FT cval =
                    esti(getMap().bonus_places[i]);
                if(cval < mval)
                    mval = cval, id = i;
            }
            empBonus.erase(id);
            MoveOp mop;
            mop.id.push_back(uid);
            if(id != -1) {
                mop.taskName = "idle";
                mop.dst = getMap().bonus_places[id];
                mop.radius = CONST::bonus_radius;
            } else {
                mop.taskName = "help";
                mop.dst = idleP;
                mop.radius =
                    3.0 * CONST::explode_radius;
            }
            mops.push_back(mop);
        }
    }
    return mops;
}
