#include "TaskScheduler.hpp"
#include "Map.hpp"
#include <algorithm>
#include <set>
int countUnits(const Point& v, FT radius, bool self) {
    int hsiz = getLogic().humans.size(), res = 0;
    for(int i = 0; i < hsiz; ++i)
        if(getUnit(i).hp > 0 &&
           (self ^ (getFac(i) != getMyFac())) &&
           dist(v, getUnit(i).position) <= radius)
            res += getUnit(i).hp;
    return res;
}
FT evalCrystal(const Crystal& ref) {
    return countUnits(ref.position, closeDis, true) -
        countUnits(ref.position, closeDis, false) -
        MapHelper::estimateDis(ref.position,
                               getMyTarget());
}
std::vector<MoveOp> calcTask(const std::vector<Group>&,
                             std::vector<int>&,
                             int& mid) {
    std::set<int> idle;
    for(int i = 0; i < humanSiz(); ++i)
        if(getMyUnit(i).hp > 0)
            idle.insert(i);
    debug() << "alive " << idle.size() << std::endl;
    mid = -1;
    for(int i = 0; i < facSiz(); ++i) {
        if(getCrystal(i).belong != -1 &&
           getFac(getCrystal(i).belong) == getMyFac())
            mid = getNum(getCrystal(i).belong);
    }
    std::vector<MoveOp> mops;
    if(mid == -1 || !idle.count(mid)) {
        // steal
        FT mval = -1e20;
        int cid = getMyFac();
        for(int i = 0; i < facSiz(); ++i) {
            if(getCrystal(i).faction == getMyFac())
                continue;
            FT cval = evalCrystal(getCrystal(i));
            if(cval > mval)
                cval = mval, cid = i;
        }
        MoveOp mop;
        mop.radius = CONST::ball_radius;
        mop.dst = getCrystal(cid).position;
        mop.taskName = "steal";
        mop.id.assign(idle.begin(), idle.end());
        mops.push_back(mop);
    } else {
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
            MoveOp mop;
            mop.id.assign(idle.begin(), idle.end());
            mop.dst = getMyUnit(mid).position;
            mop.radius = 3.0 * CONST::fireball_radius;
            mop.taskName = "help";
            mops.push_back(mop);
        }
    }
    return mops;
}
