#pragma once
#include "const.h"
#include "logic.h"
#include <algorithm>
#include <chrono>
#include <sstream>
#include <vector>
using Clock = std::chrono::high_resolution_clock;
using FT = double;

const FT vel =
    CONST::frames_per_second * CONST::human_velocity;
const FT linkDis = vel * 4.0;
const FT closeDis = vel * 7.0;
const FT panic = 10.0;
const FT minTrans = CONST::human_velocity / 5.0;
const FT keep = 2.5 * std::max(CONST::fireball_radius,
                               CONST::explode_radius);
std::ostringstream& debug();
std::ostream& error();
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
bool canFlash(const Human& hu);
bool canFire(const Human& hu);
bool canMeteor(const Human& hu);

struct MoveOp {
    std::vector<int> id;
    Point dst;
    FT radius;
    std::string taskName;
};
struct Group {
    std::vector<int> myUnit, emUnit;
};
int findFa(std::vector<int>& fa, int x);
