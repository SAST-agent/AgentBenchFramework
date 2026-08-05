#include "Map.hpp"
#include "SafeOp.hpp"
#include <cstring>
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
    bool isWalkWall(const Point& v, int invTime) {
        if(isWall(v))
            return true;
        for(const Meteor& m : getLogic().meteors) {
            if(getFac(m.from_number) != getMyFac() &&
               dist(m.position, v) <=
                   CONST::explode_radius +
                       Math::bias &&
               m.last_time > invTime)
                return true;
        }
        return false;
    }
    FT estimateDis(const Point& a, const Point& b) {
        return dist(a, b);
    }
    Point maxFire(const Point& base) {
        Point res(0.0, 0.0);
        int mcnt = -(1 << 30);
        const int sampleCnt = 128.0;
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
                mcnt = cnt, res = cur - off;
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
}
void doMove(int num, const Point& dst) {
    if(canFlash(getMyUnit(num)) &&
       dist(dst, getMyUnit(num).position) >
           CONST::human_velocity - Math::bias)
        SafeOp::flash(num);
    SafeOp::move(num, dst);
}
Point tryMove(int num, Point dir,
              const std::vector<Point>& cnxt) {
    FT rad = CONST::human_velocity;
    if(canFlash(getMyUnit(num)))
        rad = CONST::flash_distance;
    rad -= Math::bias;
    const Human& hu = getMyUnit(num);
    if(length(dir) > Math::bias &&
       !MapHelper::isWalkWall(hu.position,
                              hu.inv_time)) {
        auto accept = [&](const Point& p) {
            if(MapHelper::isWalkWall(p, hu.inv_time))
                return false;
            for(Point old : cnxt) {
                FT dis = dist(old, p);
                if(dis < 2.0 *
                       std::max(CONST::fireball_radius,
                                CONST::explode_radius))
                    return false;
            }
            return true;
        };
        if(length(dir) > rad)
            dir = dir * (rad / length(dir));
        while(length(dir) > 0.2) {
            Point cp = hu.position + dir;
            if(accept(cp)) {
                doMove(num, cp);
                return cp;
            }
            dir = dir * 0.95;
        }
    }
    debug() << "Warning:no way" << std::endl;
    const int sampleCnt = 128;
    for(int i = 0; i < sampleCnt; ++i) {
        Point cp =
            Math::sampleCircle(hu.position, rad);
        if(!MapHelper::isWalkWall(cp, hu.inv_time)) {
            doMove(num, cp);
            return cp;
        }
    }
    return hu.position;
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
Point sampleDst(const Point& base, FT radius) {
    for(int i = 0; i < 32; ++i) {
        Point cp = Math::sampleCircle(base, radius);
        if(!MapHelper::isWall(cp))
            return cp;
    }
    return base;
}
FT evalSafeFac(const Point& base) {
    FT val = 0.0;
    for(const Fireball& x : getLogic().fireballs) {
        if(getFac(x.from_number) == getMyFac())
            continue;
        Point xp = x.position;
        FT dis = dist(xp, base);
        if(dis > 60.0)
            continue;
        Point dir(cos(x.rotation), sin(x.rotation));
        {
            Point off = base -
                (xp - dir * CONST::fireball_radius);
            FT dotv = off.x * dir.x + off.y * dir.y;
            if(dotv < -Math::bias)
                continue;
        }
        FT d = ptoldist(base, Lineseg(xp, xp + dir));
        if(d > CONST::fireball_radius + Math::bias)
            continue;
        bool flag = true;
        for(int i = 0;
            i < dis / CONST::fireball_velocity; ++i) {
            xp = xp + dir * CONST::fireball_velocity;
            if(dist(xp, base) >
                   CONST::fireball_radius +
                       Math::bias &&
               MapHelper::isWall(xp)) {
                flag = false;
                break;
            }
        }
        if(flag)
            val += (60.0 - dis + Math::bias) *
                (CONST::fireball_radius - d +
                 Math::bias);
    }
    return val;
}
std::vector<PosInfo> teamMove(const MoveOp& op) {
    std::vector<PosInfo> res;
    if(op.id.empty())
        return res;
    std::vector<Point> cnxt;
    for(int id : op.id) {
        debug() << "unit " << id << " task "
                << op.taskName << std::endl;
        const Human& hu = getMyUnit(id);
        FT csf = evalSafeFac(hu.position);
        if(csf < Math::bias) {
            Point cdst = sampleDst(op.dst, op.radius);
            if(MapHelper::noWall(hu.position, cdst)) {
                res.emplace_back(
                    id, tryMove(id, cdst - hu.position,
                                cnxt));
            } else {
                Point pos = astar(
                    Posi(hu.position), Posi(cdst),
                    canFlash(hu), hu.position);
                res.emplace_back(
                    id, tryMove(id, pos - hu.position,
                                cnxt));
            }
        } else {
            FT rad = (canFlash(hu) ?
                          CONST::flash_distance :
                          CONST::human_velocity) -
                Math::bias;
            Point pos = hu.position;
            FT mval = 1e20;
            for(int i = 0; i < 2048; ++i) {
                Point cp = Math::sampleCircle(
                    hu.position, rad);
                if(MapHelper::isWalkWall(cp,
                                         hu.inv_time))
                    continue;
                FT cval = evalSafeFac(cp);
                if(cval < mval)
                    mval = cval, pos = cp;
            }
            doMove(id, pos);
            res.emplace_back(id, pos);
        }
        cnxt.push_back(res.back().second);
    }
    return res;
}
