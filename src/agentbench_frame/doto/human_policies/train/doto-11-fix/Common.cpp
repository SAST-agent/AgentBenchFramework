#include "Common.hpp"
std::ostringstream& debug() {
    static std::ostringstream ss;
    return ss;
}
std::ostream& error() {
    return debug() << "#ERROR# ";
}
bool canFlash(const Human& hu) {
    if(hu.hp <= 0 || hu.flash_num <= 0 ||
       hu.flash_time > 0)
        return false;
    for(const Crystal& cry : getLogic().crystal)
        if(cry.belong == hu.number)
            return false;
    return true;
}
bool canFire(const Human& hu) {
    return hu.fire_time <= 0 && hu.hp > 0;
}
bool canMeteor(const Human& hu) {
    return hu.meteor_number > 0 && hu.hp > 0 &&
        hu.meteor_time <= 0;
}
int findFa(std::vector<int>& fa, int x) {
    return fa[x] == x ? x : fa[x] = findFa(fa, fa[x]);
}
