#include "SafeOp.hpp"
#include <cmath>
namespace SafeOp {
    static FT sumMove = 0.0;
    static int fireCnt = 0, meteorCnt = 0, frameCnt = 0;
    static std::vector<bool> mopF, flashF, meteorF, fireF;
    static std::vector<int> lastFire, lastMeteor, lastFlash;
    static std::vector<Point> lastMove;
    struct FireHis {
        int from;
        Point base;
        SourceLocation loc;
        FireHis(int from, const Point& base, const SourceLocation& loc)
            : from(from), base(base), loc(loc) {}
    };
    static std::vector<FireHis> fireHis;
    bool checkFT(FT k) {
        return isnormal(k);
    }
    bool checkPoint(const Point& p) {
        return checkFT(p.x) && checkFT(p.y);
    }
    void move(int num, const Point& p, const SourceLocation& callPos) {
        if(num >= humanSiz()) {
            error(callPos) << "move:invaild num" << std::endl;
            return;
        }
        Point cp = getMyUnitPos(num);
        info() << "move " << num << " from [" << cp.x << " " << cp.y << "] to ["
               << p.x << " " << p.y << "] dist=" << dist(cp, p) << std::endl;
        if(!checkPoint(p) || isWall(p) || dist(cp, p) > CONST::flash_distance) {
            error(callPos) << "move:invalid pos " << std::endl;
            return;
        }
        if(mopF[num]) {
            error(callPos) << "re-move" << std::endl;
            return;
        } else
            mopF[num] = true;
        sumMove += dist(cp, p);
        getLogic().move(num, p);
    }
    void fire(int num, const Point& base, const Point& p,
              const SourceLocation& callPos) {
        if(num >= humanSiz()) {
            error(callPos) << "fire:invaild num" << std::endl;
            return;
        }
        if(!checkPoint(p) || !canFire(getMyUnit(num)) || !canFireAt(base, p)) {
            error(callPos) << "fire:can not fire [" << p.x << "," << p.y << "]"
                           << std::endl;
            return;
        }
        if(fireF[num]) {
            error(callPos) << "re-fire" << std::endl;
            return;
        } else if(lastFire[num] + CONST::human_fireball_interval >
                  getLogic().frame) {
            error(callPos) << "no fire delay but can fire:id=" << num
                           << ",last=" << lastFire[num] << std::endl;
        } else {
            fireF[num] = true;
            lastFire[num] = getLogic().frame;
            ++fireCnt;
        }
        debug() << "@shoot@ " << num << " at [" << p.x << " " << p.y << "]"
                << std::endl;
        getLogic().shoot(num, p);
        fireHis.emplace_back(
            num, base + Math::scale(p - base, CONST::splash_radius), callPos);
    }
    void flash(int num, const SourceLocation& callPos) {
        if(num >= humanSiz()) {
            error(callPos) << "flash:invaild num" << std::endl;
            return;
        }
        if(!canFlash(getMyUnit(num))) {
            error(callPos) << "flash:can not flash" << std::endl;
            return;
        }
        if(flashF[num]) {
            error(callPos) << "re-flash" << std::endl;
            return;
        } else if(lastFlash[num] + CONST::human_flash_interval >
                  getLogic().frame) {
            error(callPos) << "no flash delay but can flash:id=" << num
                           << ",last=" << lastFlash[num] << std::endl;
        } else {
            flashF[num] = true;
            lastFlash[num] = getLogic().frame;
        }
        debug() << "@flash@ " << num << std::endl;
        getLogic().flash(num);
    }
    void meteor(int num, const Point& p, const SourceLocation& callPos) {
        if(num >= humanSiz()) {
            error(callPos) << "meteor:invaild num" << std::endl;
            return;
        }
        if(!checkPoint(p) || !canMeteor(getMyUnit(num))) {
            error(callPos) << "meteor:can not meteor [" << p.x << "," << p.y
                           << "]" << std::endl;
            return;
        }
        if(meteorF[num])
            error(callPos) << "re-meteor" << std::endl;
        else if(lastMeteor[num] + CONST::human_meteor_interval >
                getLogic().frame) {
            error(callPos) << "no meteor delay but can meteor:id=" << num
                           << ",last=" << lastMeteor[num] << std::endl;
        } else {
            meteorF[num] = true;
            lastMeteor[num] = getLogic().frame;
            ++meteorCnt;
        }
        debug() << "@meteor@ " << num << " in [" << p.x << " " << p.y << "]"
                << std::endl;
        getLogic().meteor(num, p);
    }
    void begin() {
        int siz = humanSiz();
        std::vector<bool> empty(siz);
        mopF = flashF = fireF = meteorF = empty;
        if(static_cast<int>(lastFire.size()) != siz)
            lastFire.assign(siz, -100);
        if(static_cast<int>(lastMeteor.size()) != siz)
            lastMeteor.assign(siz, -100);
        if(static_cast<int>(lastFlash.size()) != siz)
            lastFlash.assign(siz, -100);
        if(static_cast<int>(lastMove.size()) == siz) {
            for(int i = 0; i < siz; ++i) {
                if(getMyUnit(i).hp <= 0)
                    continue;
                if(dist(lastMove[i], Point(-1, -1)) < Math::bias)
                    continue;
                Point exc = lastMove[i], cur = getMyUnit(i).position;
                if(dist(cur, exc) > minTrans * 0.1)
                    ERROR << "invalid move " << i << " exc:[" << exc.x << ","
                          << exc.y << "] cur:[" << cur.x << "," << cur.y << "]"
                          << std::endl;
            }
        }
        for(const FireHis& his : fireHis) {
            bool found = false;
            for(const Fireball& fb : getLogic().fireballs) {
                if(fb.from_number != getID(getMyFac(), his.from))
                    continue;
                if(dist(fb.position, his.base) < 1.0) {
                    found = true;
                    break;
                }
            }
            if(!found) {
                error(his.loc) << "invalid fireball from unit " << his.from
                               << "(last frame)" << std::endl;
            }
        }
        fireHis.clear();
    }
    void end() {
        FT frame = getLogic().frame;
        for(int i = 0; i < humanSiz(); ++i)
            frameCnt += getMyUnit(i).hp > 0;
        info() << "frame cnt:" << frameCnt << std::endl;
        info() << "alive ratio:" << 100.0 * frameCnt / (frame * humanSiz())
               << "%" << std::endl;
        FT totFlash = frameCnt / CONST::human_flash_interval;
        FT totMove = (frameCnt - totFlash) * CONST::human_velocity +
            totFlash * CONST::flash_distance;
        info() << "move ratio:" << 100.0 * sumMove / totMove << "%"
               << std::endl;
        int totFire = std::max(1, frameCnt / CONST::human_fireball_interval);
        info() << "fire ratio:" << 100.0 * fireCnt / totFire << "%"
               << "(" << fireCnt << "/" << totFire << ")" << std::endl;
        int totMeteor = std::max(1, frameCnt / CONST::human_meteor_interval);
        info() << "meteor ratio:" << 100.0 * meteorCnt / totMeteor << "%"
               << "(" << meteorCnt << "/" << totMeteor << ")" << std::endl;
        lastMove = getLogic().ope.move;
        static std::vector<Point> posHis;
        if(getLogic().frame % (CONST::frames_per_second * 10) == 0) {
            if(posHis.size() != getLogic().humans.size()) {
                posHis.clear();
                for(const Human& hu : getLogic().humans)
                    posHis.push_back(hu.position);
            } else {
                for(int i = 0; i < humanSiz(); ++i) {
                    const Human& hu = getMyUnit(i);
                    if(hu.hp <= 0)
                        continue;
                    Point cp = hu.position;
                    bool flag = true;
                    for(Point bp : getMap().bonus_places)
                        if(dist(bp, cp) < CONST::bonus_radius * 2.0) {
                            flag = false;
                            break;
                        }
                    if(flag &&
                       dist(cp, posHis[i]) < CONST::human_velocity * 2.0) {
                        ERROR << "unit " << i << " locked!!!" << std::endl;
                    }
                    posHis[i] = cp;
                }
            }
        }
    }
}
