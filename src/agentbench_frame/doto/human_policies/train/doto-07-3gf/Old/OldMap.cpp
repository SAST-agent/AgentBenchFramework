#include "Map.hpp"
#include "SafeOp.hpp"
#include <cstring>
#include <map>
#include <queue>
namespace MapHelper {
    bool isWall(int x, int y) {
        if(x < 0 || y < 0 || x >= getMap().width ||
           y >= getMap().height)
            return true;
        return getMap().pixels[x][y];
    }
    bool isWall(const Point& v) {
        return isWall(v.x, v.y);
    }
    bool noWall(const Point& s, const Point& t) {
        const int maxSample = 1000;
        int sampleCnt =
            ceil(dist(s, t) / CONST::room_size * 1.1);
        sampleCnt = std::max(
            3, std::min(sampleCnt, maxSample));
        for(int i = 0; i <= sampleCnt; ++i)
            if(isWall(Math::lerp(
                   s, t,
                   i / static_cast<FT>(sampleCnt))))
                return false;
        return true;
    }
    bool inMeteor(const Point& v) {
        for(const Meteor& m : getLogic().meteors) {
            if(isAnti(m.from_number)) {
                FT dis = dist(m.position, v);
                if(dis < CONST::explode_radius +
                           Math::hurtBias &&
                   CONST::explode_radius +
                           Math::hurtBias + dis >
                       (m.last_time - 1) *
                           CONST::human_velocity * 0.7)
                    return true;
            }
        }
        return false;
    }
    bool isWalkWall(const Point& v) {
        if(isWall(v))
            return true;
        return inMeteor(v);
    }
    Point maxFire(const Point& base) {
        Point res(0.0, 0.0);
        int mcnt = -(1 << 30);
        const int sampleCnt = 128;
        for(int i = 0; i < sampleCnt; ++i) {
            FT ang = Math::twoPi * i / sampleCnt;
            Point dir = Point(cos(ang), sin(ang)),
                  off = dir * CONST::room_size,
                  cur = base +
                dir * CONST::splash_radius;
            int cnt = 0;
            while(!isWall(cur)) {
                ++cnt;
                cur = cur + off;
            }
            if(cnt > mcnt)
                mcnt = cnt, res = base + dir;
        }
        return res;
    }
    Point clampMap(const Point& p) {
        using Math::bias;
        FT cx = std::max(
            bias,
            std::min(p.x, getMap().width - bias));
        FT cy = std::max(
            bias,
            std::min(p.y, getMap().height - bias));
        return Point(cx, cy);
    }
    bool canFireAt(const Point& cp, const Point& obj) {
        return !isWall(
            cp + Math::scale(obj - cp,
                             CONST::splash_radius));
    }
    bool inFire(const Point& p) {
        for(const Fireball& fb :
            getLogic().fireballs) {
            if(!isAnti(fb.from_number))
                continue;
            Point nxt = fb.position +
                Point(cos(fb.rotation),
                      sin(fb.rotation)) *
                    CONST::fireball_velocity;
            if(dist(nxt, p) <
               CONST::splash_radius + Math::hurtBias)
                return true;
        }
        return false;
    }
}
struct Posi {
    int x, y;
    Posi(int x, int y) : x(x), y(y) {}
    explicit Posi(const Point& p) : x(p.x), y(p.y) {}
    explicit Posi(int id)
        : x(id / getMap().height),
          y(id % getMap().height) {}
    int getId() const {
        return x * getMap().height + y;
    }
    Posi operator-(const Posi& rhs) const {
        return Posi(x - rhs.x, y - rhs.y);
    }
};
const int off[8][2] = { { -1, -1 }, { -1, 0 },
                        { -1, 1 },  { 0, -1 },
                        { 0, 1 },   { 1, -1 },
                        { 1, 0 },   { 1, 1 } };
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
std::vector<int> trans, cts, ots;
std::vector<FT> cdis;
Point astar(const Posi& src, const Posi& dst,
            bool mode, const Point& psrc) {
    int mapSiz = getMap().width * getMap().height;
    trans.resize(mapSiz);
    ots.resize(mapSiz);
    cts.resize(mapSiz);
    cdis.resize(mapSiz);
    static int timeStamp = 0;
    ++timeStamp;
    Posi dir = dst - src;
    auto esti = [&](const Posi& p) -> FT {
        int dx = iabs(p.x - dst.x),
            dy = iabs(p.y - dst.y);
        int wa = std::min(dx, dy);
        int wb = dx + dy;
        Posi line = p - dst;
        int wc = iabs(line.x * dir.y - line.y * dir.x);
        FT hv = wa * Math::sqrt2 + (wb - 2.0 * wa) +
            wc * 1e-3 + Math::uniform(0.0, 0.1);
        return hv * 1.001;
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
            int cx = up.x + off[i][0],
                cy = up.y + off[i][1];
            if(MapHelper::isWall(cx, cy))
                continue;
            Posi vp(cx, cy);
            int v = vp.getId();
            FT cg = g + (off[i][0] && off[i][1] ?
                             Math::sqrt2 :
                             1.0);
            if(ots[v] == timeStamp &&
               cg < cdis[v] - Math::bias)
                ots[v] = 0;
            if(cts[v] == timeStamp &&
               cg < cdis[v] - Math::bias)
                cts[v] = 0;
            if(ots[v] != timeStamp &&
               cts[v] != timeStamp) {
                cdis[v] = cg;
                ots[v] = timeStamp;
                heap.emplace(v, cg + esti(vp), cg);
                trans[v] = u;
            }
        }
        ++iter;
    }
    debug() << "[" << src.x << "," << src.y << "]->["
            << dst.x << "," << dst.y
            << "] iter=" << iter << std::endl;
    int sid = src.getId(), did = dst.getId();
    if(cts[did] != timeStamp)
        return Point(0.0, 0.0);
    auto off = [] {
        return Math::uniform(Math::bias,
                             1.0 - Math::bias);
    };
    const FT fac = CONST::room_size;
    Point mp(fac * (dst.x + off()),
             fac * (dst.y + off()));
    if(mode) {
        FT mdis = 0.0, end = CONST::flash_distance *
                CONST::flash_distance -
            Math::bias;
        while(sid != did) {
            Posi cp(did);
            FT cx = fac * (cp.x + off()),
               cy = fac * (cp.y + off()),
               dx = psrc.x - cx, dy = psrc.y - cy,
               w = dx * dx + dy * dy;
            if(w < end && w > mdis)
                mp = Point(cx, cy), mdis = w;
            did = trans[did];
        }
    } else {
        while(sid != did) {
            Posi cp(did);
            FT cx = (cp.x + off()) * fac,
               cy = (cp.y + off()) * fac;
            if(MapHelper::noWall(psrc,
                                 Point(cx, cy))) {
                mp = Point(cx, cy);
                break;
            }
            did = trans[did];
        }
    }
    return mp;
}
void doMove(int num, const Point& dst,
            const SourceLocation& callPos) {
    if(canFlash(getMyUnit(num)) &&
       dist(dst, getMyUnit(num).position) >
           CONST::human_velocity - Math::bias)
        SafeOp::flash(num, callPos);
    debug() << "moveop line " << callPos.line
            << std::endl;
    SafeOp::move(num, dst, callPos);
}
Point tryMove(int num, Point dir,
              const std::vector<Point>& cnxt,
              bool relax, bool stop,
              const SourceLocation& src) {
    FT rad = CONST::human_velocity;
    if(canFlash(getMyUnit(num)))
        rad = CONST::flash_distance;
    rad -= Math::bias;
    const Human& hu = getMyUnit(num);
    if(length(dir) > Math::bias) {
        auto accept = [&](const Point& p) {
            if(MapHelper::isWalkWall(p))
                return false;
            if(relax)
                return true;
            for(Point old : cnxt) {
                FT dis = dist(old, p);
                if(dis < keep)
                    return false;
            }
            return true;
        };
        if(length(dir) > rad)
            dir = dir * (rad / length(dir));
        do {
            Point cp = hu.position + dir;
            if(accept(cp)) {
                doMove(num, cp, SRCLOC);
                return cp;
            }
            dir = dir * 0.95;
        } while(length(dir) > minTrans);
    }
    if(!stop) {
        info() << "no way" << std::endl;
        const int sampleCnt = 1024;
        FT mdis = -Math::inf, mval = -Math::inf;
        Point mpA, mpB;
        bool flagA = false, flagB = false;
        for(int i = 0; i < sampleCnt; ++i) {
            Point cp = Math::sampleCircle(
                hu.position, minTrans, rad);
            if(MapHelper::isWalkWall(cp))
                continue;
            FT cdis = Math::inf;
            for(Point old : cnxt)
                cdis = std::min(cdis, dist(old, cp));
            if(cdis > mdis)
                mdis = cdis, mpB = cp;
            flagB = true;
            if(cdis > keep) {
                FT cval =
                    Math::dot(cp - hu.position, dir);
                if(cval > mval)
                    mval = cval, mpA = cp;
                flagA = true;
            }
        }
        if(flagA) {
            debug() << "use Point A" << std::endl;
            doMove(num, mpA, SRCLOC);
            return mpA;
        }
        if(flagB) {
            debug() << "use Point B" << std::endl;
            doMove(num, mpB, SRCLOC);
            return mpB;
        }
        error(src) << "No move B" << std::endl;
    }
    return hu.position;
}
const FT fbRadius = std::max(CONST::splash_radius,
                             CONST::fireball_radius) +
    Math::hurtBias;
bool evalFireBall(const Point& base, const Fireball& x,
                  FT& d, FT& dis, Point& dir,
                  bool def) {
    if(!isAnti(x.from_number))
        return false;
    Point xp = x.position;
    if(dist(xp, base) < fbRadius)
        return def;
    dir = Point(cos(x.rotation), sin(x.rotation));
    d = ptoldist(base, Lineseg(xp, xp + dir));
    if(d > fbRadius)
        return false;
    dis = Math::dot(base - xp, dir) -
        sqrt(std::max(0.0,
                      fbRadius * fbRadius - d * d));
    if(dis <= 0.0)
        return false;
    Point cp = xp;
    for(int i = 0; i * CONST::fireball_velocity < dis;
        ++i) {
        cp = cp + dir * CONST::fireball_velocity;
        if(dist(cp, base) > fbRadius &&
           MapHelper::isWall(cp))
            return false;
    }
    return true;
}
int evalSafeFac(const Point& base, const Point& mdir) {
    int mint = 10000;
    for(const Fireball& x : getLogic().fireballs) {
        FT d, dis;
        Point xp = x.position, dir;
        if(evalFireBall(base, x, d, dis, dir, false)) {
            FT vx = fabs(Math::dot(
                        mdir, Point(dir.y, -dir.x))) *
                CONST::human_velocity;
            FT dx = SameSide(mdir, base - xp,
                             makeLine(Point(0.0, 0.0),
                                      dir)) ?
                fbRadius - d :
                fbRadius + d;
            FT vy = CONST::fireball_velocity -
                Math::dot(mdir, dir) *
                    CONST::human_velocity;
            FT ty = fabs(vy) > Math::bias ? dis / vy :
                                            Math::inf,
               tx = fabs(vx) > Math::bias ? dx / vx :
                                            Math::inf;
            if(ty < tx + Math::bias && ty < Math::inf)
                mint = std::min(
                    mint, static_cast<int>(std::min(
                              10000.0, ceil(ty))));
        }
    }
    return mint;
}
using Cache = std::vector<Point>;
Cache& getSampleCache(const std::string& uid) {
    static std::map<std::string, Cache> cs;
    Cache& cache = cs[uid];
    cache.resize(humanSiz());
    return cache;
}
Point sampleDst(int id, const Point& base, FT radius) {
    Point& pos = getSampleCache("dst sample res")[id];
    Point& odst = getSampleCache("dst sample old")[id];
    if(dist(odst, base) < Math::bias &&
       dist(pos, getMyUnit(id).position) >
           minTrans * 1.1 &&
       dist(pos, base) < radius)
        return pos;
    odst = base;
    for(int i = 0; i < 512; ++i) {
        Point cp =
            Math::sampleCircle(base, 0.0, radius);
        if(!MapHelper::isWall(cp))
            return pos = cp;
    }
    pos = Point(-1.0, -1.0);
    return base;
}
bool fleeMeteor(int id, const Human& hu, Point& res) {
    Cache& mcache = getSampleCache("in Meteor");
    auto eval = [&](const Point& cp) {
        if(MapHelper::isWall(cp))
            return Math::inf;
        for(const Meteor& me : getLogic().meteors)
            if(isAnti(me.from_number) &&
               dist(me.position, cp) <
                   CONST::explode_radius +
                       Math::hurtBias)
                return Math::inf;
        if(!canFlash(hu) &&
           !MapHelper::noWall(hu.position, cp))
            return Math::inf;
        return dist(hu.position, cp);
    };
    Point& pos = mcache[id];
    FT mdis = eval(pos);
    for(int i = 0; i < 4096; ++i) {
        Point cp = Math::sampleCircle(
            hu.position, minTrans, canFlash(hu) ?
                CONST::flash_distance :
                CONST::explode_radius * 5.0);
        FT cdis = eval(cp);
        if(cdis < mdis)
            mdis = cdis, pos = cp;
    }
    if(mdis != Math::inf) {
        Point off = pos - hu.position;
        FT offd =
            (canFlash(hu) ? CONST::flash_distance :
                            CONST::human_velocity) -
            Math::bias;
        if(offd < mdis)
            off = off * (offd / mdis);
        do {
            Point odst = hu.position + off;
            if(!MapHelper::isWall(odst)) {
                doMove(id, odst, SRCLOC);
                res = odst;
                return true;
            }
            off = off * 0.95;
        } while(length(off) > minTrans);
    }
    ERROR << "No move A" << std::endl;
    return false;
}
Point fleeFireball(const std::vector<Point>& cnxt,
                   int id, const Point& dst,
                   const Human& hu, int minFac) {
    Point pos = hu.position;
    if(canFlash(hu)) {
        Point mp(0.0, 0.0);
        bool flag = false;
        FT mtrans = -Math::inf;
        for(int i = 0; i < 4096; ++i) {
            Point cp = Math::sampleCircle(
                pos, minTrans,
                CONST::flash_distance - Math::bias);
            if(MapHelper::isWalkWall(cp))
                continue;
            if(evalSafeFac(cp, Point(0.0, 0.0)) >
               minFac) {
                FT ctrans =
                    Math::dot(cp - pos, dst - pos);
                if(ctrans > mtrans)
                    mtrans = ctrans, mp = cp;
                flag = true;
            }
        }
        if(flag) {
            doMove(id, mp, SRCLOC);
            return mp;
        }
    }
    Point dir(0.0, 0.0);
    int msfac = -100, mstep = -100;
    FT mdotv = -Math::inf;
    const int sampleCnt = 1024;
    for(int i = 0; i < sampleCnt; ++i) {
        FT angle = Math::twoPi * i / sampleCnt;
        Point cd(cos(angle), sin(angle));
        int csfac =
            std::min(minFac, evalSafeFac(pos, cd));
        if(csfac < msfac)
            continue;
        int cstep = 0;
        for(int j = 1; j <= 20; ++j) {
            if(MapHelper::isWalkWall(
                   pos +
                   cd * (CONST::human_velocity * j)))
                break;
            else
                cstep = j;
        }
        if(csfac == msfac && cstep < mstep)
            continue;
        FT cdotv = Math::dot(cd, dst - pos);
        if(csfac > msfac || cstep > mstep ||
           cdotv > mdotv) {
            msfac = csfac, mstep = cstep,
            mdotv = cdotv;
            dir = cd;
        }
    }
    debug() << "max safe fac = " << msfac << std::endl;
    return tryMove(id, dir * CONST::human_velocity,
                   cnxt, true, false, SRCLOC);
}
Point calcMove(const std::vector<Point>& cnxt, int id,
               const Point& dst, FT radius,
               bool stop) {
    const Human& hu = getMyUnit(id);
    if(MapHelper::inMeteor(hu.position)) {
        Point res;
        if(fleeMeteor(id, hu, res))
            return res;
    }
    if(MapHelper::inFire(hu.position)) {
        Point dir = dst - hu.position;
        FT offd =
            (canFlash(hu) ? CONST::flash_distance :
                            CONST::human_velocity) -
            Math::bias;
        FT mang = -Math::inf;
        Point mp;
        for(int i = 0; i < 4096; ++i) {
            Point cp = Math::sampleCircle(
                hu.position, minTrans, offd);
            if(MapHelper::isWalkWall(cp) ||
               MapHelper::inFire(cp))
                continue;
            FT cang = Math::dot(cp - hu.position, dir);
            if(cang > mang)
                mang = cang, mp = cp;
        }
        if(mang != -Math::inf) {
            doMove(id, mp, SRCLOC);
            return mp;
        }
    }
    if(radius < 0.0) {
        FT offd =
            (canFlash(hu) ? CONST::flash_distance :
                            CONST::human_velocity) -
            Math::bias;
        Point off = dst - hu.position;
        if(length(off) > offd)
            off = Math::scale(off, offd);
        while(length(off) > minTrans) {
            Point odst = hu.position + off;
            if(!MapHelper::isWalkWall(odst) &&
               !MapHelper::inFire(odst)) {
                doMove(id, odst, SRCLOC);
                return odst;
            }
            off = off * 0.95;
        }
        return hu.position;
    }
    int minFac = CONST::frames_per_second;
    if(hu.hp <= CONST::human_hp * 0.5)
        minFac *= 2;
    if(hu.hp <= CONST::human_hp * 0.25)
        minFac *= 2;
    if(hu.hp <= CONST::human_hp * 0.15)
        minFac *= 2;
    int sfac =
        evalSafeFac(hu.position, Point(0.0, 0.0));
    debug() << "Safe fac " << id << "=" << sfac
            << std::endl;
    if(sfac > minFac || hu.inv_time > 0) {
        Point cdst = sampleDst(id, dst, radius);
        if(MapHelper::noWall(hu.position, cdst))
            return tryMove(id, cdst - hu.position,
                           cnxt, false, stop, SRCLOC);
        else {
            Point pos =
                astar(Posi(hu.position), Posi(cdst),
                      canFlash(hu), hu.position);
            return tryMove(id, pos - hu.position, cnxt,
                           false, stop, SRCLOC);
        }
    } else
        return fleeFireball(cnxt, id, dst, hu, minFac);
}
std::vector<PosInfo> teamMove(std::vector<Point>& cnxt,
                              const Task& op) {
    std::vector<PosInfo> res;
    if(op.id.empty())
        return res;
    std::vector<int> ids = op.id;
    std::sort(
        ids.begin(), ids.end(), [&](int a, int b) {
            return getMyUnit(a).hp > getMyUnit(b).hp;
        });
    for(int id : ids) {
        info() << std::endl
               << "unit " << id << " task "
               << op.taskName << std::endl;
        res.emplace_back(
            id,
            calcMove(cnxt, id, op.dst, op.radius,
                     op.taskName == "bonus" &&
                         dist(getMyUnit(id).position,
                              op.dst) <
                             CONST::bonus_radius));
        cnxt.push_back(res.back().second);
    }
    return res;
}
