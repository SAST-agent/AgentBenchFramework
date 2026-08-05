#pragma once
#include "SDK/const.h"
#include "SDK/logic.h"
#include <algorithm>
#include <chrono>
#include <sstream>

using Clock = std::chrono::high_resolution_clock;
using FT = double;
const FT vel = CONST::frames_per_second * CONST::human_velocity;
const FT linkDis = vel * 2.0;
const FT closeDis = vel * 3.0;
const FT panic = 10.0;
const FT minTrans = CONST::human_velocity / 5.0;
const FT keep = 2.5 * std::max(CONST::fireball_radius, CONST::explode_radius);
inline Logic& getLogic() {
    return *Logic::Instance();
}
inline const Map& getMap() {
    return getLogic().map;
}
inline int facSiz() {
    return getMap().faction_number;
}
inline int humanSiz() {
    return getMap().human_number;
}
inline int getID(int faction, int num) {
    return num * facSiz() + faction;
}
inline const Human& getUnit(int id) {
    return getLogic().humans[id];
}
inline const Human& getUnit(int faction, int num) {
    return getUnit(getID(faction, num));
}
inline int getMyFac() {
    return getLogic().faction;
}
inline const Human& getMyUnit(int num) {
    return getUnit(getMyFac(), num);
}
inline Point getMyUnitPos(int num) {
    return getMyUnit(num).position;
}
inline int getFac(int id) {
    return id % facSiz();
}
inline int getNum(int id) {
    return id / facSiz();
}
inline const Crystal& getCrystal(int id) {
    return getLogic().crystal[id];
}
inline Point getMyTarget() {
    return getMap().target_places[getMyFac()];
}
inline bool isAnti(int id) {
    return getMyFac() != getFac(id);
}
bool canFlash(const Human& hu);
bool canFire(const Human& hu);
bool canMeteor(const Human& hu);
FT minDist(const Point& pos, bool self);
int countUnits(const Point& v, FT radius);

struct SourceLocation final {
    const char* file;
    int line;
    SourceLocation(const char* file, int line) : file(file), line(line) {}
};

std::ostringstream& msg();
std::ostream& info();
std::ostream& debug();
std::ostream& error(const SourceLocation& src);

#define SRCLOC SourceLocation(__FILE__, __LINE__)
#define ERROR error(SRCLOC)
