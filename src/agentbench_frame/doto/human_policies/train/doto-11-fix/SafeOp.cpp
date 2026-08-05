#include "SafeOp.hpp"
namespace SafeOp {
    void move(int num, const Point& p) {
        if(num >= humanSiz()) {
            error() << "move:invaild num" << std::endl;
            return;
        }
        if(MapHelper::isWall(p)) {
            error() << "move:invalid pos" << std::endl;
            return;
        }
        Point cp = getMyUnit(num).position;
        debug() << "move " << num << " from [" << cp.x
                << " " << cp.y << "] to [" << p.x
                << " " << p.y
                << "] dist=" << dist(cp, p)
                << std::endl;
        getLogic().move(num, p);
    }
    void fire(int num, const Point& p) {
        if(num >= humanSiz()) {
            error() << "fire:invaild num" << std::endl;
            return;
        }
        if(!canFire(getMyUnit(num))) {
            error() << "fire:can not fire"
                    << std::endl;
            return;
        }
        debug() << "shoot " << num << " at [" << p.x
                << " " << p.y << "]" << std::endl;
        getLogic().shoot(num, p);
    }
    void flash(int num) {
        if(num >= humanSiz()) {
            error() << "flash:invaild num"
                    << std::endl;
            return;
        }
        if(!canFlash(getMyUnit(num))) {
            error() << "flash:can not flash"
                    << std::endl;
            return;
        }
        debug() << "flash " << num << std::endl;
        getLogic().flash(num);
    }
    void meteor(int num, const Point& p) {
        if(num >= humanSiz()) {
            error() << "meteor:invaild num"
                    << std::endl;
            return;
        }
        if(!canMeteor(getMyUnit(num))) {
            error() << "meteor:can not meteor"
                    << std::endl;
            return;
        }
        debug() << "meteor " << num << " in [" << p.x
                << " " << p.y << "]" << std::endl;
        getLogic().meteor(num, p);
    }
}
