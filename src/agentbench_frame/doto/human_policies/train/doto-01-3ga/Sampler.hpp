#pragma once
#include "Common.hpp"
#include "Math.hpp"
#include <functional>
#include <vector>

class Bool final {
private:
    bool val;

public:
    Bool(bool val) : val(val) {}

    bool test() const {
        return val;
    }
};

class FloatingPoint final {
private:
    FT val;

public:
    FloatingPoint(FT val) : val(val) {}

    FT get() const {
        return val;
    }
};

template <typename Input>
using PredFunc = std::function<Bool(const Input& in)>;
// Maximum
template <typename Input>
using EvalFunc = std::function<FloatingPoint(const Input& in)>;

using PointPredFunc = PredFunc<Point>;
using PointEvalFunc = EvalFunc<Point>;
class PointSampler;
using PointApplyor = std::function<void(PointSampler& sampler)>;
using PointGenerator = std::function<Point(int index, int all)>;

class PointSampler final {
private:
    PointGenerator mGen;
    std::vector<PointPredFunc> mReqs;
    std::vector<PointEvalFunc> mEvals;

public:
    explicit PointSampler(const PointGenerator& src);
    PointSampler& request(const PointPredFunc& req);
    PointSampler& estimate(const PointEvalFunc& eval);
    PointSampler& optional(const PointPredFunc& test);
    PointSampler& apply(const PointApplyor& func);
    Point sample(size_t sampleCnt,
                 const std::function<Point()>& expect = {}) const;
};

using DiscretePredFunc = PredFunc<int>;
using DiscreteEvalFunc = EvalFunc<int>;
using Int2PointFunc = std::function<Point(int id)>;
class DiscreteSampler;
using DiscreteApplyor = std::function<void(DiscreteSampler& sampler)>;

class DiscreteSampler final {
private:
    std::vector<int> mPoints;
    Int2PointFunc mCast;
    std::vector<DiscretePredFunc> mReqs;
    std::vector<DiscreteEvalFunc> mEvals;

public:
    explicit DiscreteSampler(int siz);
    explicit DiscreteSampler(const std::vector<int>& points);
    DiscreteSampler& castPos(const Int2PointFunc& cast);
    DiscreteSampler& request(const DiscretePredFunc& req);
    DiscreteSampler& optional(const DiscretePredFunc& test);
    DiscreteSampler& estimate(const DiscreteEvalFunc& eval);
    DiscreteSampler& request(const PointPredFunc& req);
    DiscreteSampler& optional(const PointPredFunc& test);
    DiscreteSampler& estimate(const PointEvalFunc& eval);
    DiscreteSampler& apply(const DiscreteApplyor& func);
    int sample() const;
    std::vector<int> sample(size_t cnt) const;
    Point castToPos(int id) const;
};

// Toolbar
PointSampler circleSampler(const Point& base, FT radius);
PointSampler moveSampler(const Point& base, FT rad);
PointSampler stableSampler(const Point& base, FT rad);

template <typename Container>
DiscreteSampler fromContainer(const Container& cont) {
    return DiscreteSampler{ std::vector<int>{ std::begin(cont),
                                              std::end(cont) } };
}

PointEvalFunc distance(const Point& base);
PointEvalFunc dot(const Point& base, const Point& dst);
PointEvalFunc angle(const Point& base, const Point& dst);
PointEvalFunc distanceCloseTo(const Point& base, FT rad);

template <typename Input>
EvalFunc<Input> prec(const EvalFunc<Input>& func, FT k) {
    return [func, k](const Input& in) { return round(func(in).get() / k); };
}

template <typename Input>
EvalFunc<Input> minv(const EvalFunc<Input>& func) {
    return [func](const Input& in) { return -func(in).get(); };
}

template <typename Input>
PredFunc<Input> operator>(const EvalFunc<Input>& fa,
                          const EvalFunc<Input>& fb) {
    return [=](const Input& in) { return fa(in).get() > fb(in).get(); };
}
template <typename Input>
PredFunc<Input> operator<(const EvalFunc<Input>& fa,
                          const EvalFunc<Input>& fb) {
    return [=](const Input& in) { return fa(in).get() < fb(in).get(); };
}
template <typename Input>
PredFunc<Input> operator>(const EvalFunc<Input>& fa, FT val) {
    return [=](const Input& in) { return fa(in).get() > val; };
}
template <typename Input>
PredFunc<Input> operator<(const EvalFunc<Input>& fa, FT val) {
    return [=](const Input& in) { return fa(in).get() < val; };
}

template <typename Input>
PredFunc<Input> operator!(const PredFunc<Input>& func) {
    return [=](const Input& in) { return !func(in).test(); };
}

template <typename Input>
PredFunc<Input> operator|(const PredFunc<Input>& fa,
                          const PredFunc<Input>& fb) {
    return [=](const Input& in) { return fa(in).test() || fb(in).test(); };
}

template <typename Input>
PredFunc<Input> operator|(const PredFunc<Input>& fa, bool val) {
    return val ? PredFunc<Input>([](const Input&) { return true; }) : fa;
}

template <typename Input>
PredFunc<Input> operator&(const PredFunc<Input>& fa,
                          const PredFunc<Input>& fb) {
    return [=](const Input& in) { return fa(in).test() && fb(in).test(); };
}

template <typename Input>
PredFunc<Input> operator&(const PredFunc<Input>& fa, bool val) {
    return val ? fa : PredFunc<Input>([](const Input&) { return false; });
}

Int2PointFunc myUnitPos();
Int2PointFunc allUnitPos();
Int2PointFunc estiUnitPos();
Int2PointFunc bonusPos();
Int2PointFunc crystalPos();
Int2PointFunc crystalInitPos();

DiscretePredFunc isSelf();
DiscretePredFunc isEnemy();
DiscretePredFunc isNotEq(int id);
DiscretePredFunc isEnemyFac();
DiscretePredFunc isAlive();
DiscreteEvalFunc unitHP();
