"""Interval-aware numerical definitions; no production-solver dependencies."""
import numpy as np
__all__ = ["events", "linear_integral", "linear_abs_difference", "step_abs_difference"]

def events(left, right, probability, a=0., b=None, P0=1.):
    """Constant within-step event rates integrated with exact interval overlap."""
    left, right, probability = [np.asarray(v, dtype=np.float64) for v in (left, right, probability)]
    if not (left.shape == right.shape == probability.shape and left.ndim == 1 and left.size):
        raise ValueError("Expected matching nonempty one-dimensional event arrays")
    if not (P0 > 0 and np.isfinite(P0) and all(np.all(np.isfinite(v)) for v in (left, right, probability)) and np.all(right > left)):
        raise ValueError("Invalid launch norm, event values or interval widths")
    if np.any(np.diff(left) < 0) or np.any(np.abs(left[1:] - right[:-1]) > 1e-10):
        raise ValueError("Native event intervals must be ordered and contiguous")
    b = float(right[-1]) if b is None else b
    if b < a or a < left[0] - 1e-12 or b > right[-1] + 1e-10:
        raise ValueError("Window outside native event support")
    lo, hi = np.maximum(left, a), np.minimum(right, b)
    width = np.maximum(hi-lo, 0.)
    rate = probability / (right-left) / P0
    D = float(np.sum(rate*width, dtype=np.float64))
    M1 = float(np.sum(rate*width*(hi+lo)/2, dtype=np.float64))
    return {"probability": D, "first_moment": M1, "conditional_mean": M1/D if D else float("nan")}

def _window(x, y, a, b):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not np.all(np.diff(x) > 0):
        raise ValueError("Linear interpolation requires equal vectors and increasing times")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Nonfinite trace: respect validity masks before integration")
    if a < x[0]-1e-12 or b > x[-1]+1e-10 or b < a:
        raise ValueError("Window outside sample support")
    t = np.r_[a, x[(x>a)&(x<b)], b]
    return t, np.interp(t, x, y)

def linear_integral(x, y, a=0., b=None):
    """Exact integral of piecewise-linear samples including window endpoints."""
    b = float(np.asarray(x)[-1]) if b is None else b
    t,v = _window(x,y,a,b)
    return float(np.sum(np.diff(t)*(v[:-1]+v[1:])/2, dtype=np.float64))

def linear_abs_difference(xa, ya, xb, yb, a, b):
    """Absolute linear-trace difference on the union grid with zero splitting."""
    ta,_ = _window(xa,ya,a,b); tb,_ = _window(xb,yb,a,b)
    t = np.union1d(ta,tb)
    d = np.interp(t,xa,ya)-np.interp(t,xb,yb)
    u,v = np.abs(d[:-1]),np.abs(d[1:])
    same = (np.signbit(d[:-1]) == np.signbit(d[1:])) | (d[:-1] == 0) | (d[1:] == 0)
    area = np.empty_like(u)
    area[same]=(u[same]+v[same])/2
    other=~same
    scale=np.maximum(u[other],v[other])
    un,vn=u[other]/scale,v[other]/scale
    area[other]=scale*(un**2+vn**2)/(2*(un+vn))
    return float(np.sum(np.diff(t)*area,dtype=np.float64))

def step_abs_difference(la, ra, pa, lb, rb, pb, a, b, P0a=1., P0b=1.):
    """Absolute difference of piecewise-constant native event-rate histories."""
    events(la,ra,pa,a,b,P0a); events(lb,rb,pb,a,b,P0b)
    la,ra,pa,lb,rb,pb = map(np.asarray,(la,ra,pa,lb,rb,pb))
    t=np.unique(np.r_[a,b,la[(la>a)&(la<b)],ra[(ra>a)&(ra<b)],lb[(lb>a)&(lb<b)],rb[(rb>a)&(rb<b)]])
    mid=(t[:-1]+t[1:])/2
    ia=np.clip(np.searchsorted(ra,mid,side="right"),0,len(ra)-1)
    ib=np.clip(np.searchsorted(rb,mid,side="right"),0,len(rb)-1)
    da=pa[ia]/(ra[ia]-la[ia])/P0a; db=pb[ib]/(rb[ib]-lb[ib])/P0b
    return float(np.sum(np.diff(t)*np.abs(da-db),dtype=np.float64))
