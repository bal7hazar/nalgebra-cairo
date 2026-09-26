"""Design model of `SchurN` (WP 8.5-P16): the Cairo iteration re-implemented on Python integers
(raw Q32.32, floor / nearest exactly as the `Real` kernels), bit-faithful to the generated code with
the flags `STUCK_K = 8`, `NEAREST = True`, `HESS_NEAREST = False`, `EXC = True`, `thr_abs = 64`
(validated pass for pass against `tests_linalg_schur*`). The other flag values are the variants
measured in the WP 8.5-P16 report (floor rounding, no exceptional shift, no stall doubling, one-pass
axes with `TWO_PASS = False`, unscaled shift vector with `SCALED = False`). `schur` returns `(q, t,
passes)`; where `try_new` returns `None` it returns `("STATE", q, t)` (inspection), and `("DIVERGED",
passes)` past `max_guard` passes. Not part of the build.
"""
import math, random
S = 32
ONE = 1 << S
EPS = 1
TWO_PASS = True
SCALED = True

def mul(a, b): return (a * b) >> S
NEAREST = True
HALF = (1 << (S - 1))
def fused(prods, adds=()):  # prods: list of (a,b) (sign folded), adds raw
    return (sum(a * b for a, b in prods) + (sum(adds) << S) + (HALF if NEAREST else 0)) >> S
def mul_add(a, b, c): return (a * b + (c << S) + (HALF if NEAREST else 0)) >> S
def div(a, b):
    num = a << S
    q, r = divmod(num, b)  # floor
    # round to nearest ties even
    if b < 0: q, r = divmod(-num, -b)
    twice = 2 * r
    d = abs(b)
    if twice > d or (twice == d and q & 1): q += 1
    return q
def isqrt(w): return math.isqrt(w)
def norm(xs): return isqrt(sum(x * x for x in xs))

def refl_axis(xs):
    n = norm(xs)
    if n == 0: return 0, False, xs
    signed = -n if xs[0] < 0 else n
    y = [xs[0] + signed] + xs[1:]
    d = norm(y)
    v = [div(x, d) for x in y]
    if TWO_PASS:
        d2 = norm(v)
        v = [div(x, d2) for x in v]
    return -signed, True, v

class Mat:
    def __init__(s, n, a=None):
        s.n = n; s.a = a if a else [[0]*n for _ in range(n)]
    def copy(s): return Mat(s.n, [r[:] for r in s.a])

def reflect_left(t, u, rows, cols):
    # rows: list of row indices (axis), cols: columns
    for j in cols:
        h = fused([(u[k], t.a[r][j]) for k, r in enumerate(rows)])
        w = h + h
        for k, r in enumerate(rows):
            t.a[r][j] = mul_add(-w, u[k], t.a[r][j])

def reflect_right(t, u, cols, rows):
    for i in rows:
        h = fused([(t.a[i][c], u[k]) for k, c in enumerate(cols)])
        w = h + h
        for k, c in enumerate(cols):
            t.a[i][c] = mul_add(-w, u[k], t.a[i][c])

HESS_NEAREST = None
def hessenberg(m):
    """returns (q, h) : unpack of upstream Hessenberg (sign conventions)"""
    global NEAREST
    saved = NEAREST
    if HESS_NEAREST is not None: NEAREST = HESS_NEAREST
    try:
        return _hessenberg(m)
    finally:
        NEAREST = saved
def _hessenberg(m):
    n = m.n; a = m.copy(); sub = []
    axes = []
    for i in range(n - 1):
        rows = list(range(i + 1, n))
        col = [a.a[r][i] for r in rows]
        nrm, nz, u = refl_axis(col)
        sub.append(nrm)
        if nz:
            sign = -1 if nrm < 0 else 1
            # bilateral: reflect_rows_with_sign on right = columns i+1.. of all rows, sign
            # lhs[r][c] = sign*lhs - 2 sign (lhs·u) u
            for r in range(n):
                h = fused([(a.a[r][c], u[k]) for k, c in enumerate(rows)])
                w = h + h
                for k, c in enumerate(rows):
                    x = a.a[r][c]
                    a.a[r][c] = mul_add(-w, u[k], x) if sign > 0 else mul_add(w, u[k], -x)
            # reflect_with_sign(right.rows(i+1..), sign)
            for c in range(i + 1, n):
                h = fused([(u[k], a.a[r][c]) for k, r in enumerate(rows)])
                w = h + h
                for k, r in enumerate(rows):
                    x = a.a[r][c]
                    a.a[r][c] = mul_add(-w, u[k], x) if sign > 0 else mul_add(w, u[k], -x)
            for k, r in enumerate(rows): a.a[r][i] = u[k]
            axes.append(u)
        else:
            axes.append(None)
    # q = assemble_q
    q = Mat(n, [[ONE if i == j else 0 for j in range(n)] for i in range(n)])
    for i in reversed(range(n - 1)):
        u = axes[i]
        if u is None:
            # upstream: axis is the (zero) column, reflect with it: x - 2(u.x)u = x ; with sign
            sign = -1 if sub[i] < 0 else 1
            if sign < 0:
                for r in range(i+1, n):
                    for c in range(i, n): q.a[r][c] = -q.a[r][c]
            continue
        sign = -1 if sub[i] < 0 else 1
        rows = list(range(i + 1, n))
        for c in range(i, n):
            h = fused([(u[k], q.a[r][c]) for k, r in enumerate(rows)])
            w = h + h
            for k, r in enumerate(rows):
                x = q.a[r][c]
                q.a[r][c] = mul_add(-w, u[k], x) if sign > 0 else mul_add(w, u[k], -x)
    h = a.copy()
    for i in range(n):
        for j in range(n):
            if i > j + 1: h.a[i][j] = 0
    for i in range(n - 1): h.a[i + 1][i] = abs(sub[i])
    return q, h

def delimit(t, eps, end, thr_abs):
    n = end
    while n > 0:
        m = n - 1
        od = abs(t.a[n][m])
        if od <= thr_abs or od <= mul(eps, abs(t.a[n][n]) + abs(t.a[m][m])):
            t.a[n][m] = 0
        else:
            break
        n -= 1
    if n == 0: return 0, 0
    ns = n - 1
    while ns > 0:
        m = ns - 1
        od = t.a[ns][m]
        odn = abs(od)
        if od == 0 or odn <= thr_abs or odn <= mul(eps, abs(t.a[ns][ns]) + abs(t.a[m][m])):
            t.a[ns][m] = 0
            break
        ns -= 1
    return ns, n

def eig2(h00, h01, h10, h11):
    d = h00 - h11
    D4 = 4 * h10 * h01 + d * d  # raw^2 scale, = 4*discr * 2^64 ... careful
    if D4 < 0: return None
    s = isqrt(D4)  # = 2 sqrt(discr) raw
    tr = h00 + h11
    return (tr + s) >> 1, (tr - s) >> 1

def givens_new(c, s):
    d = norm([c, s])
    if d <= 0: return (ONE, 0), 0
    sign = -1 if c < 0 else 1
    nrm = sign * d
    return (div(abs(c), d), div(s, nrm)), nrm

def basis2(t, i):
    h00, h01, h10, h11 = t.a[i][i], t.a[i][i+1], t.a[i+1][i], t.a[i+1][i+1]
    if h10 == 0: return None
    dd = h00 - h11
    W = 4 * h10 * h01 + dd * dd
    if W < 0: return None
    sq = isqrt(W)
    x1 = ((dd + sq) * (1 << 31)) >> 32
    x2 = ((dd - sq) * (1 << 31)) >> 32
    x = x1 if abs(x1) > abs(x2) else x2
    (c, s), _ = givens_new(x, h10)
    d = norm([c, s])
    return (div(c, d), div(s, d)), x

def rot_left(t, c, s, r0, cols):  # rotate: rows r0,r0+1
    for j in cols:
        a, b = t.a[r0][j], t.a[r0+1][j]
        t.a[r0][j] = fused([(a, c), (-s, b)])
        t.a[r0+1][j] = fused([(s, a), (b, c)])
def rot_right(t, c, s, c0, rows):
    for j in rows:
        a, b = t.a[j][c0], t.a[j][c0+1]
        t.a[j][c0] = fused([(a, c), (s, b)])
        t.a[j][c0+1] = fused([(-s, a), (b, c)])

STUCK_K = 0
EXC = False
EXC_EVERY = 8
EXC_AT = 5
DEFL = []
THR_CAP = 1 << 16
def schur(m, eps=EPS, max_niter=0, thr_abs=0, max_guard=2000, compute_q=True):
    n = m.n
    amax = max(abs(x) for r in m.a for x in r)
    t = m.copy()
    if amax:
        t.a = [[div(x, amax) for x in r] for r in t.a]
    q, t = hessenberg(t)
    niter = 0
    base_thr = thr_abs
    stuck = 0
    start, end = delimit(t, eps, n - 1, thr_abs)
    while end != start:
        old_top = (start, end)
        subdim = end - start + 1
        if subdim > 2:
            mm, nn = end - 1, end
            h11, h12, h21, h22, h32 = t.a[start][start], t.a[start][start+1], t.a[start+1][start], t.a[start+1][start+1], t.a[start+2][start+1]
            hnn, hmm, hnm, hmn = t.a[nn][nn], t.a[mm][mm], t.a[nn][mm], t.a[mm][nn]
            if EXC and stuck == EXC_AT:
                sx = abs(t.a[nn][mm]) + abs(t.a[mm][mm-1])
                hnn = mul(sx, 3221225472) + t.a[nn][nn]   # 0.75 s + hnn
                hmm = hnn
                hmn = -mul(sx, 1879048192)                # -0.4375 s
                hnm = sx
            tra = hnn + hmm
            # det = hnn*hmm - hnm*hmn; axis.x = h11^2 + h12 h21 - tra h11 + det  (one fused)
            if SCALED:
                sc = abs(h21) + abs(h11 - hmm) + abs(h11 - hnn) + abs(hnm)
                p, r, g = div(h11 - hmm, sc), div(hnm, sc), div(h21, sc)
                ax = fused([(p, h11 - hnn), (-hmn, r), (h12, g)])
                ay = mul(g, h11 + h22 - tra)
                az = mul(g, h32)
            else:
                ax = fused([(h11, h11), (h12, h21), (-tra, h11), (hnn, hmm), (-hnm, hmn)])
                ay = mul(h21, h11 + h22 - tra)
                az = mul(h21, h32)
            axis = [ax, ay, az]
            for k in range(start, nn - 1):
                nrm, nz, u = refl_axis(axis)
                if nz:
                    if k > start:
                        t.a[k][k-1] = nrm; t.a[k+1][k-1] = 0; t.a[k+2][k-1] = 0
                    rows = [k, k+1, k+2]
                    reflect_left(t, u, rows, range(k, n))
                    krows = min(k + 4, end + 1)
                    reflect_right(t, u, rows, range(0, krows))
                    if compute_q: reflect_right(q, u, rows, range(n))
                axis = [t.a[k+1][k], t.a[k+2][k], t.a[k+3][k] if k < nn - 2 else axis[2]]
                if not (k < nn - 2):
                    axis = [t.a[k+1][k], t.a[k+2][k], axis[2]]
            nrm, nz, u = refl_axis(axis[:2])
            if nz:
                t.a[mm][mm-1] = nrm; t.a[nn][mm-1] = 0
                rows = [mm, nn]
                reflect_left(t, u, rows, range(mm, n))
                reflect_right(t, u, rows, range(0, end + 1))
                if compute_q: reflect_right(q, u, rows, range(n))
        else:
            rot = basis2(t, start)
            if rot is not None:
                (c, s), x = rot
                h00, h11 = t.a[start][start], t.a[end][end]
                rot_left(t, c, -s, start, range(start, n))  # inverse rotation
                rot_right(t, c, s, start, range(0, end + 1))
                t.a[end][start] = 0
                t.a[start][start] = h11 + x; t.a[end][end] = h00 - x
                if compute_q: rot_right(q, c, s, start, range(n))
            if end > 2: end -= 2
            else: break
        old = old_top
        start, end = delimit(t, eps, end, thr_abs)
        if STUCK_K:
            if (start, end) == old:
                stuck += 1
                if stuck == STUCK_K:
                    stuck = 0
                    if thr_abs < THR_CAP: thr_abs *= 2
            else:
                if thr_abs > base_thr: DEFL.append(thr_abs)
                stuck = 0
                thr_abs = base_thr
        niter += 1
        if niter == max_niter: return ("STATE", q, t)
        if niter >= max_guard: return ("DIVERGED", niter)
    t.a = [[mul(x, amax) for x in r] for r in t.a]
    return q, t, niter
