#include "playerAI.h"
#include "Attack.hpp"
#include "Common.hpp"
#include "Map.hpp"
#include "TaskScheduler.hpp"
#include <csignal>
#include <exception>

void debugTime(const std::string& name,
               Clock::time_point& ts) {
    Clock::time_point cur = Clock::now();
    debug() << name << " time = "
            << std::chrono::duration_cast<
                   std::chrono::milliseconds>(cur - ts)
                   .count()
            << " ms" << std::endl;
    ts = cur;
}
void playerAIImpl() {
    static int lastFrame = 0, skiped = 0;
    skiped += getLogic().frame - lastFrame - 1;
    lastFrame = getLogic().frame;
    debug() << "Skiped frame = " << skiped
            << std::endl;
    debug() << "meteor = " << getLogic().meteors.size()
            << std::endl;
    debug() << "fireball = "
            << getLogic().fireballs.size()
            << std::endl;
    Clock::time_point now = Clock::now();
    int hsiz = getLogic().humans.size();
    std::vector<int> fa(hsiz);
    for(int i = 0; i < hsiz; ++i)
        fa[i] = i;
    for(int i = 0; i < hsiz; ++i) {
        if(getUnit(i).hp <= 0)
            continue;
        Point sp = getUnit(i).position;
        for(int j = i + 1; j < hsiz; ++j) {
            if(getUnit(j).hp <= 0 ||
               dist(sp, getUnit(j).position) > linkDis)
                continue;
            int fu = findFa(fa, i), fv = findFa(fa, j);
            if(fu != fv)
                fa[fu] = fv;
        }
    }
    std::vector<Group> groups(hsiz);
    for(int i = 0; i < hsiz; ++i)
        if(getUnit(i).hp > 0) {
            Group& addg = groups[findFa(fa, i)];
            (getFac(i) == getMyFac() ? addg.myUnit :
                                       addg.emUnit)
                .push_back(i);
        }
    debugTime("group", now);
    int mid = -1;
    std::vector<MoveOp> trans =
        calcTask(groups, fa, mid);
    debugTime("task", now);
    std::vector<Point> nxtPos(humanSiz());
    std::vector<Point> cnxt;
    for(const MoveOp& t : trans) {
        std::vector<PosInfo> res = teamMove(cnxt, t);
        for(const PosInfo& info : res)
            nxtPos[info.first] = info.second;
    }
    debugTime("move", now);
    calcAttack(mid, fa, nxtPos);
    debugTime("attack", now);
    updateSamples();
}
void signalHandler(int sig) {
    error() << "Signal " << sig << std::endl;
    throw std::runtime_error("signal");
}
void playerAI() {
    debug().str("");
    signal(SIGSEGV, signalHandler);
    signal(SIGABRT, signalHandler);
    signal(SIGILL, signalHandler);
    signal(SIGFPE, signalHandler);
    try {
        playerAIImpl();
    } catch(const std::exception& exc) {
        error() << "Exception:" << exc.what()
                << std::endl;
    } catch(...) {
        error() << "Unknown error" << std::endl;
    }
    getLogic().debug(debug().str());
}
