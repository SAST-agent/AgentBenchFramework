#include "Analyzer.hpp"
#include "Common.hpp"
#include "Math.hpp"
#include <algorithm>
#include <cmath>
#include <map>
const Point invalidPos = Point(-1, -1);
std::vector<Fireball> newFireBall() {
    static std::vector<Fireball> ofbs;
    std::vector<Fireball> res;
    for(const Fireball& nfb : getLogic().fireballs) {
        if(std::find_if(ofbs.begin(), ofbs.end(), [&](const Fireball& ofb) {
               return ofb.from_number == nfb.from_number &&
                   fabs(nfb.rotation - ofb.rotation) < Math::bias &&
                   dist(nfb.position, ofb.position) <
                   CONST::fireball_velocity + Math::hurtBias;
           }) == ofbs.end())
            res.push_back(nfb);
    }
    ofbs = getLogic().fireballs;
    return res;
}
std::vector<Meteor> newMeteor() {
    static std::vector<Meteor> omes;
    std::vector<Meteor> res;
    for(const Meteor& nme : getLogic().meteors) {
        if(std::find_if(omes.begin(), omes.end(), [&](const Meteor& ome) {
               return ome.from_number == nme.from_number &&
                   dist(nme.position, ome.position) < Math::bias;
           }) == omes.end())
            res.push_back(nme);
    }
    omes = getLogic().meteors;
    return res;
}
void analyseUsage(int fac, int nfb, int nme) {
    // alive
    static std::map<int, int> frames;
    for(int i = 0; i < humanSiz(); ++i) {
        if(getUnit(getID(fac, i)).hp > 0)
            ++frames[fac];
    }
    info() << "alive ratio:"
           << 100.0 * frames[fac] / (getLogic().frame * humanSiz()) << "%"
           << std::endl;
    // move
    static std::map<int, FT> move;
    static std::map<int, Point> pos;
    for(int i = 0; i < humanSiz(); ++i) {
        int id = getID(fac, i);
        if(getUnit(id).hp > 0) {
            Point cp = getUnit(id).position;
            if(pos.count(id))
                move[fac] += dist(cp, pos[id]);
            pos[id] = cp;
        } else
            pos.erase(id);
    }
    FT totFlash = frames[fac] / CONST::human_flash_interval;
    FT totMove = (frames[fac] - totFlash) * CONST::human_velocity +
        totFlash * CONST::flash_distance;
    info() << "move ratio:" << 100.0 * move[fac] / totMove << "%" << std::endl;
    // fire
    static std::map<int, int> fireCnt;
    fireCnt[fac] += nfb;
    int totFire = std::max(1, frames[fac] / CONST::human_fireball_interval);
    info() << "fire ratio:" << 100.0 * fireCnt[fac] / totFire << "%"
           << "(" << fireCnt[fac] << "/" << totFire << ")" << std::endl;
    // meteor
    static std::map<int, int> meteorCnt;
    meteorCnt[fac] += nme;
    int totMeteor = std::max(1, frames[fac] / CONST::human_meteor_interval);
    info() << "meteor ratio:" << 100.0 * meteorCnt[fac] / totMeteor << "%"
           << "(" << meteorCnt[fac] << "/" << totMeteor << ")" << std::endl;
}
std::map<int, std::map<int, int>> enemys;
void analyseEnemy(const std::vector<Fireball>& nfb) {
    for(const Fireball& fb : nfb) {
        const Human& hu = getUnit(fb.from_number);
        if(hu.hp < 0 || getFac(hu.number) == getMyFac())
            continue;
        Point cp = hu.position;
        Point sdir(cos(fb.rotation), sin(fb.rotation));
        int obj = -1;
        FT mang = 1e20;
        for(int i = 0; i < humanSiz(); ++i) {
            if(getMyUnit(i).hp < 0)
                continue;
            Point op = getMyUnitPos(i);
            Point dir = op - cp;
            FT cang = fabs(atan2(Math::cross(sdir, dir), Math::dot(sdir, dir)));
            if(isfinite(cang) && cang < mang)
                obj = i, mang = cang;
        }
        if(obj != -1 && mang < Math::pi / 18.0)
            enemys[obj][hu.number] = getLogic().frame;
    }
}
std::vector<int> listEnemys(int id) {
    std::vector<int> res;
    for(auto em : enemys[id])
        if(getLogic().frame - em.second < 3 * CONST::frames_per_second &&
           getUnit(em.first).hp > 0)
            res.push_back(em.first);
    return res;
}
template <typename T>
std::vector<T> filter(const std::vector<T>& A, int fac) {
    std::vector<T> res;
    for(const T& a : A)
        if(getFac(a.from_number) == fac)
            res.push_back(a);
    return res;
}
std::map<int, std::pair<Point, Point>> samples;
Point estimate(int id) {
    return samples[id].first - samples[id].second;
}
void updateSamples(bool forceRep) {
    size_t hsiz = getLogic().humans.size();
    static int cacheRef = 1, cacheMiss = 0;
    for(size_t i = 0; i < hsiz; ++i) {
        Point cp = getUnit(i).position;
        Point old = estimate(i);
        if(!forceRep &&
           dist(cp, samples[i].first) < CONST::human_velocity * 1.1)
            samples[i].second = samples[i].first;
        else
            samples[i].second = cp;
        samples[i].first = cp;
        if(getFac(i) != getMyFac()) {
            Point cur = estimate(i);
            if(dist(old, cur) > CONST::human_velocity * 0.2)
                ++cacheMiss;
            ++cacheRef;
        }
    }
    info() << "sample miss = " << 100.0 * cacheMiss / cacheRef << "%"
           << std::endl;
}
std::vector<int> lastBonus;
void updateBonus() {
    static std::vector<bool> state;
    size_t bcnt = getLogic().bonus.size();
    if(state.size() != bcnt)
        state.assign(bcnt, false);
    if(lastBonus.size() != bcnt)
        lastBonus.assign(bcnt, 0);
    for(size_t i = 0; i < bcnt; ++i)
        if(state[i] && !getLogic().bonus[i]) {
            info() << "bonus " << i << " disappeared." << std::endl;
            lastBonus[i] = getLogic().frame;
        }
    state = getLogic().bonus;
}
int getLastBonus(int id) {
    if(lastBonus.size() != getMap().bonus_places.size())
        return 0;
    return lastBonus[id];
}
void updateState(bool forceRep) {
    updateBonus();
    updateSamples(forceRep);
    std::vector<Fireball> nfb = newFireBall();
    std::vector<Meteor> nme = newMeteor();
    for(int i = 0; i < facSiz(); ++i) {
        if(i == getMyFac())
            continue;
        info() << "analyse fac " << i << std::endl;
        std::vector<Fireball> infb = filter(nfb, i);
        std::vector<Meteor> inme = filter(nme, i);
        analyseUsage(i, infb.size(), inme.size());
    }
    analyseEnemy(nfb);
}
