#include "Map.hpp"
#include "SafeOp.hpp"
#include <cstring>
#include <map>
#include <queue>
bool isWallInt(int x, int y) {
    if(x < 0 || y < 0 || x >= getMap().width || y >= getMap().height)
        return true;
    return getMap().pixels[x][y];
}
bool isWall(const Point& v) {
    return isWallInt(v.x, v.y);
}
bool noWall(const Point& s, const Point& t) {
    const int maxSample = 1000;
    int sampleCnt = ceil(dist(s, t) / CONST::room_size * 1.1);
    sampleCnt = std::max(3, std::min(sampleCnt, maxSample));
    for(int i = 0; i <= sampleCnt; ++i)
        if(isWall(Math::lerp(s, t, i / static_cast<FT>(sampleCnt))))
            return false;
    return true;
}
bool inMeteor(const Point& v) {
    for(const Meteor& m : getLogic().meteors) {
        if(isAnti(m.from_number)) {
            FT dis = dist(m.position, v);
            if(dis < CONST::explode_radius + Math::hurtBias &&
               CONST::explode_radius + Math::hurtBias + dis >
                   (m.last_time - 1) * CONST::human_velocity * 0.7)
                return true;
        }
    }
    return false;
}
bool canFireAt(const Point& cp, const Point& obj) {
    return !isWall(cp + Math::scale(obj - cp, CONST::splash_radius));
}
bool canFireTo(const Point& cp, const Point& obj) {
    Point off = Math::scale(obj - cp, CONST::fireball_velocity);
    Point pos = cp + off;
    if(isWall(pos))
        return false;
    int cnt = ceil(dist(pos, obj) / CONST::fireball_velocity);
    for(int i = 0; i < cnt; ++i) {
        pos = pos + off;
        if(dist(pos, obj) < CONST::fireball_radius)
            return true;
        if(isWall(pos))
            return false;
    }
    return false;
}
bool inFire(const Point& p) {
    for(const Fireball& fb : getLogic().fireballs) {
        if(!isAnti(fb.from_number))
            continue;
        Point nxt = fb.position +
            Point(cos(fb.rotation), sin(fb.rotation)) *
                CONST::fireball_velocity;
        if(dist(nxt, p) < CONST::splash_radius + Math::hurtBias)
            return true;
    }
    return false;
}
struct Posi {
    int x, y;
    Posi(int x, int y) : x(x), y(y) {}
    explicit Posi(const Point& p) : x(p.x), y(p.y) {}
    explicit Posi(int id) : x(id / getMap().height), y(id % getMap().height) {}
    int getId() const {
        return x * getMap().height + y;
    }
    Posi operator-(const Posi& rhs) const {
        return Posi(x - rhs.x, y - rhs.y);
    }
};
const int off[8][2] = { { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 },
                        { 0, 1 },   { 1, -1 }, { 1, 0 },  { 1, 1 } };
struct Info {
    int id;
    FT f, g;
    Info(int id, FT f, FT g) : id(id), f(f), g(g) {}
    bool operator<(const Info& rhs) const {
        return f > rhs.f;
    }
};
int iabs(int a) {
    return a >= 0 ? a : -a;
}
Point astar(const Posi& src, const Posi& dst, bool mode, const Point& psrc) {
    static std::vector<int> trans, cts, ots;
    static std::vector<FT> cdis;
    int mapSiz = getMap().width * getMap().height;
    trans.resize(mapSiz);
    ots.resize(mapSiz);
    cts.resize(mapSiz);
    cdis.resize(mapSiz);
    static int timeStamp = 0;
    ++timeStamp;
    Posi dir = dst - src;
    auto esti = [&](const Posi& p) -> FT {
        int dx = iabs(p.x - dst.x), dy = iabs(p.y - dst.y);
        int wa = std::min(dx, dy);
        int wb = dx + dy;
        Posi line = p - dst;
        int wc = iabs(line.x * dir.y - line.y * dir.x);
        FT hv = wa * Math::sqrt2 + (wb - 2.0 * wa) + wc * 1e-3 +
            Math::uniform(0.0, 0.1);
        return hv * 1.003;
    };
    std::priority_queue<Info> heap;
    heap.emplace(src.getId(), esti(src), 0.0);
    ots[src.getId()] = timeStamp;
    int iter = 0;
    while(heap.size()) {
        int u = heap.top().id;
        FT g = heap.top().g;
        heap.pop();
        if(ots[u] == 0)
            continue;
        cts[u] = timeStamp;
        ots[u] = 0;
        if(u == dst.getId())
            break;
        Posi up(u);
        for(int i = 0; i < 8; ++i) {
            int cx = up.x + off[i][0], cy = up.y + off[i][1];
            if(isWallInt(cx, cy))
                continue;
            Posi vp(cx, cy);
            int v = vp.getId();
            FT cg = g + (off[i][0] && off[i][1] ? Math::sqrt2 : 1.0);
            if(ots[v] == timeStamp && cg < cdis[v] - Math::bias)
                ots[v] = 0;
            if(cts[v] == timeStamp && cg < cdis[v] - Math::bias)
                cts[v] = 0;
            if(ots[v] != timeStamp && cts[v] != timeStamp) {
                cdis[v] = cg;
                ots[v] = timeStamp;
                heap.emplace(v, cg + esti(vp), cg);
                trans[v] = u;
            }
        }
        ++iter;
        if(iter > 20000)
            return Math::invalidPos;
    }
    info() << "[" << src.x << "," << src.y << "]->[" << dst.x << "," << dst.y
           << "] iter=" << iter << std::endl;
    int sid = src.getId(), did = dst.getId();
    if(cts[did] != timeStamp)
        return Math::invalidPos;
    auto off = [] { return Math::uniform(Math::bias, 1.0 - Math::bias); };
    const FT fac = CONST::room_size;
    Point mp(fac * (dst.x + off()), fac * (dst.y + off()));
    if(mode) {
        FT mdis = 0.0,
           end = CONST::flash_distance * CONST::flash_distance - Math::bias;
        while(sid != did) {
            Posi cp(did);
            FT cx = fac * (cp.x + off()), cy = fac * (cp.y + off()),
               dx = psrc.x - cx, dy = psrc.y - cy, w = dx * dx + dy * dy;
            if(w < end && w > mdis)
                mp = Point(cx, cy), mdis = w;
            did = trans[did];
        }
    } else {
        while(sid != did) {
            Posi cp(did);
            FT cx = (cp.x + off()) * fac, cy = (cp.y + off()) * fac;
            if(noWall(psrc, Point(cx, cy))) {
                mp = Point(cx, cy);
                break;
            }
            did = trans[did];
        }
    }
    return mp;
}
using Cache = std::vector<Point>;
Cache& getSampleCache(const std::string& uid) {
    static std::map<std::string, Cache> cs;
    Cache& cache = cs[uid];
    cache.resize(humanSiz());
    return cache;
}
Point sampleDst(int id, const Point& base, FT minRad, FT maxRad) {
    Point& pos = getSampleCache("dst sample res")[id];
    Point& odst = getSampleCache("dst sample old")[id];
    if(dist(odst, base) < Math::bias &&
       dist(pos, getMyUnitPos(id)) > minTrans * 1.1 &&
       dist(pos, base) <= maxRad && dist(pos, base) >= minRad)
        return pos;
    odst = base;
    pos = PointSampler([=](int, int) {
              static uint32_t index = 0;
              ++index;
              FT w = Math::halton3(index);
              FT rad = minRad * w + maxRad * (1.0 - w);
              FT ang = Math::halton7(index) * Math::twoPi;
              return base + Point(cos(ang) * rad, sin(ang) * rad);
          })
              .request(!isWallFunc())
              .sample(512);
    if(pos != Math::invalidPos)
        return pos;
    return base;
}
Point path(int id, const Point& dst, FT minRad, FT maxRad) {
    Point cdst = sampleDst(id, dst, minRad, maxRad);
    const Human& hu = getMyUnit(id);
    if(noWall(hu.position, cdst))
        return cdst;
    else {
        Clock::time_point beg = Clock::now();
        Point res =
            astar(Posi(hu.position), Posi(cdst), canFlash(hu), hu.position);
        Clock::duration dur = Clock::now() - beg;
        info() << "astar "
               << std::chrono::duration_cast<std::chrono::microseconds>(dur)
                      .count()
               << "us" << std::endl;
        return res;
    }
}
void doMove(int num, const Point& dst, const SourceLocation& callPos) {
    if(canFlash(getMyUnit(num)) &&
       dist(dst, getMyUnitPos(num)) > CONST::human_velocity)
        SafeOp::flash(num, callPos);
    SafeOp::move(num, dst, callPos);
}
PointSampler makeMoveBaseSampler(int id) {
    const Human& hu = getMyUnit(id);
    return moveSampler(hu.position, canFlash(hu) ? CONST::flash_distance :
                                                   CONST::human_velocity)
        .request(!isWallFunc())
        .request(distance(hu.position) > minTrans);
}
const FT fbRadius = CONST::splash_radius + Math::hurtBias;
bool evalFireBall(const Point& base, const Fireball& x, FT& dis) {
    if(!isAnti(x.from_number))
        return false;
    Point xp = x.position;
    if(dist(xp, base) < fbRadius)
        return false;
    Point dir = Point(cos(x.rotation), sin(x.rotation));
    FT d = ptoldist(base, Lineseg(xp, xp + dir));
    if(d > fbRadius)
        return false;
    dis = Math::dot(base - xp, dir) -
        sqrt(std::max(0.0, fbRadius * fbRadius - d * d));
    if(dis <= 0.0)
        return false;
    Point cp = xp;
    for(int i = 0; i * CONST::fireball_velocity < dis; ++i) {
        cp = cp + dir * CONST::fireball_velocity;
        if(dist(cp, base) > fbRadius && isWall(cp))
            return false;
    }
    return true;
}
const int predict = 20;
FT evalSafeFacFlash(const Point& base) {
    FT mint = predict;
    for(const Fireball& x : getLogic().fireballs) {
        FT dis;
        if(evalFireBall(base, x, dis)) {
            FT ty = dis / CONST::fireball_velocity;
            mint = std::min(mint, ceil(ty));
        }
    }
    return mint;
}
std::vector<std::pair<Point, Point>> preSafeFac(const Point& base) {
    std::vector<std::pair<Point, Point>> res;
    for(const Fireball& x : getLogic().fireballs) {
        if(!isAnti(x.from_number))
            continue;
        Point fdir(cos(x.rotation), sin(x.rotation));
        Point foff = fdir * CONST::fireball_velocity;
        bool flagA = true, flagB = false;
        const int sampleCnt = 12;
        for(int i = 0; i < sampleCnt; ++i) {
            FT ang = Math::twoPi * i / sampleCnt;
            Point mdir(cos(ang), sin(ang));
            Point moff = mdir * CONST::human_velocity, cp = base,
                  fp = x.position;
            int ht = 0;
            for(int j = 1; j <= predict; ++j) {
                fp = fp + foff, cp = cp + moff;
                if(dist(fp, cp) < CONST::splash_radius) {
                    ht = j;
                    flagB = true;
                    break;
                }
                if(isWall(fp) || isWall(cp))
                    break;
            }
            if(ht == 0) {
                flagA = false;
                if(flagB)
                    break;
            }
        }
        if(!flagA && flagB)
            res.emplace_back(x.position, foff);
    }
    return res;
}
FT evalSafeFacMove(const Point& base, const Point& mdir,
                   const std::vector<std::pair<Point, Point>>& fbs) {
    Point moff = mdir * CONST::human_velocity;
    int mt = predict + 1;
    for(std::pair<Point, Point> fb : fbs) {
        Point cp = base;
        for(int i = 1; i < mt; ++i) {
            fb.first = fb.first + fb.second, cp = cp + moff;
            if(dist(fb.first, cp) < fbRadius || isWall(cp)) {
                mt = i;
                break;
            }
            if(isWall(fb.first))
                break;
        }
    }
    return mt;
}
bool hasWall(const Point& p) {
    int x = p.x, y = p.y;
    for(int i = -3; i <= 3; ++i)
        for(int j = -3; j <= 3; ++j)
            if(isWallInt(x + i, y + j))
                return true;
    return false;
}
int maxFire(const Point& base, const Point& dir) {
    if(!canFireAt(base, dir))
        return -1;
    Point off = dir * CONST::fireball_velocity,
          cur = base + dir * CONST::splash_radius;
    int cnt = 0;
    while(!isWall(cur)) {
        ++cnt;
        cur = cur + off;
    }
    return cnt;
}
Point maxFire(const Point& base) {
    Point res(0.0, 0.0);
    int mcnt = -(1 << 30);
    const int sampleCnt = 128;
    for(int i = 0; i < sampleCnt; ++i) {
        FT ang = Math::twoPi * i / sampleCnt;
        Point dir = Point(cos(ang), sin(ang));
        int cnt = maxFire(base, dir);
        if(cnt > mcnt)
            mcnt = cnt, res = base + dir;
    }
    return res;
}
Point scanCorner(const Point& base, const Point& dst) {
    Point bdir = normalize(dst - base);
    int src = maxFire(base, bdir);
    int cl = src, cr = src;
    const FT off = 0.3;
    for(FT i = off; i < 60.0; i += off) {
        FT ang = Math::pi / 180.0 * i;

        Point cdl = Math::rotate(bdir, ang);
        int ccl = maxFire(base, cdl);
        if(ccl > cl * 2)
            return cdl;
        cl = ccl;

        Point cdr = Math::rotate(bdir, -ang);
        int ccr = maxFire(base, cdr);
        if(ccr > cr * 2)
            return cdr;
        cr = ccr;
    }
    return bdir;
}

PointPredFunc isWallFunc() {
    return isWall;
}
PointPredFunc hasWallFunc() {
    return hasWall;
}
PointPredFunc inFireFunc() {
    return inFire;
}
PointPredFunc inMeteorFunc() {
    return inMeteor;
}
PointPredFunc canFireAtFunc(const Point& src) {
    return [src](const Point& dst) { return canFireAt(src, dst); };
}
PointPredFunc noWallFunc(const Point& src) {
    return [src](const Point& dst) { return noWall(src, dst); };
}
PointPredFunc canFireToFunc(const Point& src) {
    return [src](const Point& dst) { return canFireTo(src, dst); };
}
