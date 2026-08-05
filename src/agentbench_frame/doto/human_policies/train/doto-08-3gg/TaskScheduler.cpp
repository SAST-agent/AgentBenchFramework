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
std::vector<Task> calcTask() {
    std::set<int> idle, bak;
    int alive = 0;
    for(int i = 0; i < humanSiz(); ++i)
        if(getMyUnit(i).hp > 0)
            idle.insert(i), ++alive;
    bak = idle;
    std::vector<Task> mops;
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
        bool helped = false;
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
                Point cp = getCrystal(cid).position;
                bool help = !helped && cid != -1 &&
                    dist(getMyUnitPos(his[bid]), cp) + 3 * vel <
                        minDist(cp, true) &&
                    getMyUnit(his[bid]).hp > 0.2 * CONST::human_hp;
                if(help) {
                    for(int i = 0; i < humanSiz(); ++i) {
                        const Human& hu = getUnit(cid, i);
                        if(hu.hp <= 0 || dist(hu.position, cp) < 3 * vel) {
                            help = false;
                            break;
                        }
                    }
                }
                if(help)
                    his.erase(bid), helped = true;
                else {
                    Task mop;
                    mop.id.push_back(his[bid]);
                    idle.erase(his[bid]);
                    mop.extra = bid;
                    mop.task = TaskType::Bonus;
                    mops.push_back(mop);
                }
            }
        }
    }
    int mid = -1;
    for(int i = 0; i < facSiz(); ++i) {
        if(getCrystal(i).belong != -1 &&
           getFac(getCrystal(i).belong) == getMyFac()) {
            mid = getNum(getCrystal(i).belong);
        }
    }
    if(idle.count(mid)) {
        Point mp = getMyUnit(mid).position;
        FT mtdis = dist(mp, getMyTarget()) - CONST::target_radius;
        const FT flashVel = CONST::human_velocity +
            CONST::flash_distance / CONST::human_flash_interval;
        Task mop;
        mop.task = TaskType::Trans;
        mop.id.push_back(mid);
        idle.erase(mid);
        for(int id : idle)
            if(mtdis / CONST::human_velocity >
                   (dist(getMyUnitPos(id), cryPos) - CONST::ball_radius +
                    CONST::flash_distance) /
                       flashVel ||
               getMyUnit(id).hp < CONST::human_hp * 0.2)
                mop.id.push_back(id);
        for(int id : mop.id)
            idle.erase(id);
        mop.extra = mid;
        mops.push_back(mop);
    }
    {
        Task mop;
        mop.id.assign(idle.begin(), idle.end());
        if(bak.count(mid))
            mop.task = TaskType::NxtSteal;
        else
            mop.task = TaskType::Steal;
        mop.extra = cid;
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
