#include "TaskScheduler.hpp"
#include "Map.hpp"
#include <algorithm>
#include <map>
#include <set>
int closest(const std::set<int>& idle, const Point& p) {
    return fromContainer(idle)
        .castPos(myUnitPos())
        .estimate(minv(distance(p)))
        .sample();
}
int countUnits(const Point& v, FT radius) {
    int hsiz = getLogic().humans.size(), res = 0;
    for(int i = 0; i < hsiz; ++i)
        if(getUnit(i).hp > 0 && getFac(i) != getMyFac() &&
           dist(v, getUnit(i).position) <= radius)
            ++res;
    return res;
}
std::vector<Task> calcTask() {
    std::set<int> idle, bak;
    int alive = 0;
    for(int i = 0; i < humanSiz(); ++i)
        if(getMyUnit(i).hp > 0)
            idle.insert(i), ++alive;
    bak = idle;
    std::vector<Task> mops;
    // bonus
    {
        std::vector<int> bp =
            DiscreteSampler(getLogic().bonus.size())
                .castPos(bonusPos())
                .request([&](const Point& p) {
                    return countUnits(p, CONST::bonus_radius) <= 1;
                })
                .estimate(minv<Point>([&](const Point& p) {
                    FT val = Math::inf;
                    for(int id : idle)
                        val = std::min(val, dist(getMyUnitPos(id), p));
                    return val;
                }))
                .sample(2);
        static std::map<int, int> his;
        for(int bid : bp) {
            bool flag = false;
            if(his.count(bid) && idle.count(his[bid])) {
                flag = true;
            } else {
                his.erase(bid);
                for(int id : idle)
                    if(getMyUnit(id).inv_time > 0) {
                        his[bid] = id;
                        flag = true;
                        break;
                    }
            }
            if(flag) {
                Task mop;
                mop.id.push_back(his[bid]);
                idle.erase(his[bid]);
                mop.extra = bid;
                mop.task = TaskType::Bonus;
                mops.push_back(mop);
            }
        }
        /*
        for(size_t i = 0; i < bp.size(); ++i) {
            Point bpos = bonusPos()(bp[i]);
            int tid = fromContainer(idle)
                          .castPos(myUnitPos())
                          .request([&](const Point& cp) {
                              FT cdis = dist(cp, bpos);
                              if(cid != -1 &&
                                 dist(cp, getCrystal(cid).position) <
                                     std::min(cdis, linkDis))
                                  return false;
                              return true;
                          })
                          .request([&](int id) {
                              Point cp = getMyUnitPos(id);
                              FT cdis = dist(cp, bpos);
                              if(cdis > CONST::bonus_radius * 4.0 &&
                                 getMyUnit(id).hp < CONST::human_hp * 0.8)
                                  return false;
                              return true;
                          })
                          .estimate(minv(distance(bpos)))
                          .sample();
            if(tid == -1)
                break;
            idle.erase(tid);
            Task mop;
            mop.id.push_back(tid);
            mop.extra = bp[i];
            mop.task = TaskType::Bonus;
            mops.push_back(mop);
        }
        */
    }
    int mid = -1;
    for(int i = 0; i < facSiz(); ++i) {
        if(getCrystal(i).belong != -1 &&
           getFac(getCrystal(i).belong) == getMyFac()) {
            mid = getNum(getCrystal(i).belong);
        }
    }
    int cid = DiscreteSampler(facSiz())
                  .request(isEnemyFac())
                  .estimate([](int id) {
                      const Crystal& ref = getCrystal(id);
                      FT res = minDist(ref.position, false) -
                          minDist(ref.position, true);
                      if(ref.belong != -1 && getFac(ref.belong) != getMyFac())
                          res -= getUnit(ref.belong).hp * 2.0;
                      return res - dist(ref.position, getMyTarget());
                  })
                  .sample();
    Point cryPos = cid == -1 ? Math::invalidPos : crystalInitPos()(cid);
    if(idle.count(mid)) {
        Point mp = getMyUnit(mid).position;
        FT mtdis = dist(mp, getMyTarget());
        bool flag = mtdis > minDist(getMyTarget(), false) &&
            getMyUnit(mid).hp > CONST::human_hp * 0.1;
        PointEvalFunc closestBonus = [](const Point& cp) {
            FT mdis = Math::inf;
            for(const Point& bp : getMap().bonus_places)
                mdis = std::min(mdis, dist(cp, bp));
            return mdis;
        };
        Task mop;
        mop.task = TaskType::Trans;
        idle.erase(mid);
        const FT flashVel = CONST::human_velocity +
            CONST::flash_distance / CONST::human_flash_interval;
        mop.id = fromContainer(idle)
                     .castPos(myUnitPos())
                     .request(closestBonus > 3.0 * CONST::bonus_radius)
                     .request([=](int id) {
                         if(flag)
                             return true;
                         if(dist(cryPos, getMyUnitPos(id)) / flashVel <
                            mtdis / CONST::human_velocity)
                             return true;
                         return getMyUnit(id).hp < CONST::human_hp * 0.2;
                     })
                     .estimate(minv(distance(mp)))
                     .sample(2);
        for(int id : mop.id)
            idle.erase(id);
        mop.id.push_back(mid);
        mop.extra = mid;
        mops.push_back(mop);
    }
    {
        Task mop;
        mop.id.assign(idle.begin(), idle.end());
        /*
        FT mdis = 0.0;
        for(int a : idle)
            for(int b : idle)
                mdis = std::max(
                    mdis, dist(getMyUnitPos(a), getMyUnitPos(b)));
        if(mdis > linkDis)
            mop.task = TaskType::Group;
        else
        */
        if(cid != -1) {
            if(bak.count(mid))
                mop.task = TaskType::NxtSteal;
            else
                mop.task = TaskType::Steal;
            mop.extra = cid;
        } else
            mop.task = TaskType::Guard;
        mops.push_back(mop);
    }
    int sub = 0;
    for(const Task& mop : mops)
        for(int id : mop.id)
            bak.erase(id), ++sub;
    if(bak.size() || sub != alive)
        ERROR << "Bad task schedule:tot=" << alive
              << " but remain=" << bak.size() << " sub=" << sub << std::endl;
    return mops;
}
