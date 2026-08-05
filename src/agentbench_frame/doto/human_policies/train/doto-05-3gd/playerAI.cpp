#include "Analyzer.hpp"
#include "AttackOperator.hpp"
#include "Common.hpp"
#include "MoveOperator.hpp"
#include "SafeOp.hpp"
#include "TaskScheduler.hpp"
#include <cerrno>
#include <csignal>
#include <cstring>
#include <exception>
#include <map>
void debugTime(const std::string& name, Clock::time_point& ts) {
    Clock::time_point cur = Clock::now();
    int64_t val =
        std::chrono::duration_cast<std::chrono::milliseconds>(cur - ts).count();
    static std::map<std::string, int64_t> maxv;
    maxv[name] = std::max(maxv[name], val);
    info() << name << " time = " << val << " ms"
           << "(peak = " << maxv[name] << "ms)" << std::endl;
    ts = cur;
}
std::map<int, Point> doMove(const Task& task) {
    std::map<int, Point> mps;
    if(task.id.empty())
        return mps;
    switch(task.task) {
        case TaskType::Trans: {
            if(task.extra >= 0 && task.extra < humanSiz())
                mps = doTrans(task.id, task.extra);
            else
                ERROR << "bad trans task" << std::endl;
        } break;
        case TaskType::Steal: {
            if(task.extra >= 0 && task.extra < facSiz() &&
               task.extra != getMyFac()) {
                mps = doSteal(task.id, task.extra);
            } else
                ERROR << "bad steal task" << std::endl;
        } break;
        case TaskType::NxtSteal: {
            if(task.extra >= 0 && task.extra < facSiz()) {
                mps = doNxtSteal(task.id, task.extra);
            } else
                ERROR << "bad nxtsteal task" << std::endl;
        } break;
        case TaskType::Bonus: {
            if(task.id.size() == 1 && task.extra >= 0 &&
               task.extra < static_cast<int>(getLogic().bonus.size())) {
                mps[task.id.front()] = doBonus(task.id.front(), task.extra);
            } else
                ERROR << "bad bonus task" << std::endl;
        } break;
        default: { ERROR << "bad unknown task" << std::endl; }
    }
    return mps;
}
void playerAIImpl() {
    static int lastFrame = 0, skipped = 0;
    SafeOp::begin();
    int skf = getLogic().frame - lastFrame - 1;
    skipped += skf;
    lastFrame = getLogic().frame;
    info() << "Skipped frame = " << skipped << std::endl;
    Clock::time_point now = Clock::now(), bak = now;
    updateState(skf);
    debugTime("analyse", now);
    std::vector<Task> tasks = calcTask();
    debugTime("task", now);
    std::map<int, Point> mps;
    size_t alive = 0;
    for(const Task& t : tasks) {
        alive += t.id.size();
        std::map<int, Point> mp = doMove(t);
        for(auto p : mp)
            mps[p.first] = p.second;
    }
    if(mps.size() != alive)
        ERROR << "mps size=" << mps.size() << " alive=" << alive << std::endl;
    debugTime("move", now);
    doAttack(mps);
    debugTime("attack", now);
    SafeOp::end();
    debugTime("frame", bak);
}
void signalHandler(int sig) {
    ERROR << "Signal " << sig
#ifndef _WIN32
          << " " << strsignal(sig)
#endif
          << std::endl;
    // throw std::runtime_error("signal");
}
template <typename T>
void setSignal(T handler) {
    signal(SIGSEGV, handler);
    // signal(SIGABRT, handler);
    signal(SIGILL, handler);
    signal(SIGFPE, handler);
// signal(SIGINT, handler);
// signal(SIGTERM, handler);
#ifndef _WIN32
    // signal(SIGQUIT, handler);
    signal(SIGPIPE, handler);
    signal(SIGSTKFLT, handler);
    signal(SIGIOT, handler);
    signal(SIGBUS, handler);
// signal(SIGKILL, handler);
#endif
}
void playerAI() {
    msg().str("");
    if(getLogic().frame == 1) {
        info() << "Version = " << __DATE__ << " " << __TIME__ << std::endl;
    }
    setSignal(signalHandler);
    errno = 0;
    try {
        playerAIImpl();
    } catch(const std::exception& exc) {
        ERROR << "Exception:" << exc.what() << std::endl;
    } catch(...) {
        ERROR << "Unknown error" << std::endl;
    }
    if(errno)
        ERROR << "Errno = " << errno << " " << strerror(errno) << std::endl;
    getLogic().debug(msg().str());
    setSignal(SIG_IGN);
}
