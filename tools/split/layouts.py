"""Candidate layouts for `plan.py` (WP 9-NS1). Each layout: `CRATES` (lowest first), `home(g, n)`."""
import types

from plan import dims_in_name, shape_of


def band(dims, cuts):
    """Index of the band of a shape (max dimension) given the band upper bounds `cuts`."""
    m = max(dims)
    for i, c in enumerate(cuts):
        if m <= c:
            return i
    return len(cuts)


def make_naive():
    """Every file stays whole; base by dimension band, then geometry, linalg, the rest."""
    crates = [
        "core", "dim1_3", "dim4", "dim5", "dim6", "base_families", "geometry", "linalg", "dynamic",
        "sparse", "top",
    ]

    def home(g, n):
        p = g.file(n)
        s = shape_of(p)
        if s:
            return ["dim1_3", "dim4", "dim5", "dim6"][band(s, [3, 4, 5])]
        if p.startswith("base/dynamic"):
            return "dynamic"
        if p.startswith("base/"):
            if any(k in p for k in ("statistics", "blas", "solve", "cg")):
                return "base_families"
            return "core"
        if p.startswith("geometry"):
            return "geometry"
        if p.startswith("linalg"):
            return "linalg"
        if p.startswith("sparse") or p.startswith("io"):
            return "sparse"
        return "top"

    return types.SimpleNamespace(CRATES=crates, home=home)


naive = make_naive()


def follow_struct(g, n, home_of_struct):
    """Home of a core-trait impl of one nalgebra type: its struct's home (same module)."""
    if g.kind(n) == "typebound" and g.info[n]["args"]:
        st = g.struct_node.get(g.info[n]["args"][0])
        if st is not None:
            return home_of_struct(g, st)
    return None


def make_z(split_geometry=True):
    """Zero-break layout: per dimension band, the TYPES (struct + core-trait impls + the generic
    impls anchored on it) below the INHERENT traits (`Matrix3Trait`...), because an inherent trait
    of dimension k builds shapes of dimension k + 1 (`insert_column`, `push`, `to_homogeneous`)."""
    crates = [
        "core", "t1_3", "t4", "i1_3", "t5", "i4", "geometry", "t6", "i5", "i6", "kron",
        "base_families", "linalg", "dynamic", "sparse", "top",
    ]

    def home(g, n):
        f = follow_struct(g, n, home0)
        return f if f is not None else home0(g, n)

    def home0(g, n):
        p = g.file(n)
        it = g.info[n]
        name = it["gen"] or it["name"]
        if name == "ApproxEqTrait":
            return "core"
        if name == "MatrixKronecker" or it["of"] == "MatrixKronecker":
            return "kron"
        s = shape_of(p)
        if s:
            b = band(s, [3, 4, 5])
            if g.kind(n) == "inherent":
                return ["i1_3", "i4", "i5", "i6"][b]
            return ["t1_3", "t4", "t5", "t6"][b]
        if p.startswith("base/dynamic"):
            return "dynamic"
        if p.startswith("base/"):
            if any(k in p for k in ("statistics", "blas", "solve", "cg")):
                return "base_families"
            return "core"
        if p.startswith("geometry"):
            if g.kind(n) == "struct":
                return "t1_3"
            return "geometry"
        if p.startswith("linalg"):
            return "linalg"
        if p.startswith("sparse") or p.startswith("io"):
            return "sparse"
        return "top"

    return types.SimpleNamespace(CRATES=crates, home=home)


z = make_z()


GEOMETRY_CORE = {
    "abstract_rotation", "point", "quaternion", "unit_quaternion", "unit_complex", "rotation2",
    "rotation3", "translation2", "translation3", "isometry2", "isometry3", "isometry_matrix2",
    "isometry_matrix3", "similarity2", "similarity3", "similarity_matrix2", "similarity_matrix3",
    "dual_quaternion", "unit_dual_quaternion",
}
GEOMETRY_ND = {"point5", "point6", "translation5", "translation6", "scale5", "scale6",
               "reflection5", "reflection6"}
LINALG_FAMILY = {
    # file stem (without the shape suffix) -> family key
    "lu": "lu", "lu_steps": "lu", "inverse": "lu", "permutation_sequence": "perm",
    "cholesky": "chol", "cholesky_update": "chol", "ldlt": "chol", "udu": "chol",
    "qr": "qr", "col_piv_qr": "cpqr", "full_piv_lu": "fplu", "lblt": "lblt",
    "svd": "svd", "svd2": "svd", "svd3": "svd", "symmetric_eigen": "eig",
    "symmetric_tridiagonal": "tri", "hessenberg": "hess", "householder": "hh",
    "householder_kernels": "hh", "householder_steps": "hh", "givens": "hh", "balancing": "hess",
    "schur": "schur", "eigen": "schur", "bidiagonal": "bidiag", "exp": "exp", "pow": "exp",
}


def linalg_key(p):
    """(family, band) of a linalg file; band 's' (dimensions <= 4), '5', '6' or 'x' (shared)."""
    parts = p[: -len(".cairo")].split("/")
    stem = parts[1]
    import re as _re

    fam_stem = _re.sub(r"\d(x\d)?$", "", stem)
    fam = LINALG_FAMILY.get(fam_stem, fam_stem)
    d = dims_in_name(p) if (len(parts) > 2 or _re.search(r"\d(x\d)?$", stem)) else None
    if parts[-1] in ("kernels", "perm1_5") or d is None:
        return fam, "x"
    m = max(d)
    return fam, ("s" if m <= 4 else str(m))


def make_v1(linalg_crates=None):
    """Zero-break layout, first cut (see docs/SPLIT.md)."""
    la = linalg_crates or [
        "la_solve", "la_perm", "la_hh", "la_lu_s", "la_chol", "la_qr_s", "la_eig_s", "la_svd_s",
        "la_small_rest", "la_lu_5", "la_lu_6", "la_qr_5", "la_qr_6", "la_eig_5", "la_eig_6",
        "la_svd_5", "la_svd_6", "la_rest_5", "la_rest_6",
    ]
    crates = [
        "core", "base3", "t5", "base4", "geometry", "t6", "base5", "base6", "geometry_nd", "kron",
        "statistics", "blas",
    ] + la + ["dynamic", "sparse", "top"]

    def home(g, n):
        f = follow_struct(g, n, home0)
        return f if f is not None else home0(g, n)

    def home0(g, n):
        p = g.file(n)
        it = g.info[n]
        name = it["gen"] or it["name"]
        if name == "ApproxEqTrait":
            return "core"
        if name == "MatrixKronecker" or it["of"] == "MatrixKronecker":
            return "kron"
        s = shape_of(p)
        if s:
            b = band(s, [3, 4, 5])
            if g.kind(n) == "inherent":
                return ["base3", "base4", "base5", "base6"][b]
            return ["core", "core", "t5", "t6"][b]
        stem = p[: -len(".cairo")].split("/")[-1]
        if p.startswith("base/dynamic"):
            return "dynamic"
        if p.startswith("base/"):
            if "statistics" in p:
                return "statistics"
            if "blas" in p:
                return "blas"
            if "solve" in p:
                return "la_solve"
            if "cg" in p:
                return "geometry"
            return "core"
        if p.startswith("geometry"):
            if stem in GEOMETRY_CORE:
                return "base3"
            if stem in GEOMETRY_ND:
                return "geometry_nd"
            return "geometry"
        if p.startswith("linalg/"):
            fam, b = linalg_key(p)
            if fam == "perm":
                return "la_perm"
            if fam == "hh":
                return "la_hh"
            if fam == "chol":
                return "la_chol"
            if b in ("s", "x"):
                if fam in ("lu", "qr", "eig", "svd"):
                    return f"la_{fam}_s"
                return "la_small_rest"
            if fam in ("lu", "qr", "eig", "svd"):
                return f"la_{fam}_{b}"
            return f"la_rest_{b}"
        if p.startswith("sparse") or p.startswith("io"):
            return "sparse"
        return "top"

    def pull(g):
        # `Matrix6Trait::is_invertible` runs `Lu6`: the 6x6 LU sits with the dimension-6 methods
        return {n: "base6" for n in g.nodes if g.label(n).split(":")[0] in ("Matrix6LuTrait",)}

    return types.SimpleNamespace(CRATES=crates, home=home, pull=pull)


v1 = make_v1()


VIEW_TRAITS = {"FixedView", "FixedRows", "FixedColumns", "FixedResize", "PadTo6", "CropFrom6",
               "ShapeDims", "ColumnPart", "RowPart"}
PRODUCT_TRAITS = {"MatrixMul", "MatrixTrMul"}
NORM_MARKERS = {"EuclideanNorm", "LpNorm", "OneNorm", "UniformNorm"}


def make_v2(la=None, base6_split=True):
    """Zero-break layout, second cut: the 36 structs and their core-trait impls at the bottom; the
    generic families (views, products, Kronecker, norms) in crates of their own, their impls in the
    module of the trait (or of the norm marker) instead of the shape's; then the inherent traits
    per dimension band (see docs/SPLIT.md)."""
    la = la or [
        "la_solve", "la_perm", "la_hh", "la_lu_s", "la_chol", "la_qr_s", "la_eig_s", "la_svd_s",
        "la_small_rest", "la_lu_5", "la_lu_6", "la_qr_5", "la_qr_6", "la_eig_5", "la_eig_6",
        "la_svd_5", "la_svd_6", "la_rest_5", "la_rest_6",
    ]
    crates = [
        "core", "products", "base3", "base4", "geometry", "base5", "base6", "views", "kron",
        "norms", "geometry_nd", "statistics", "blas",
    ] + la + ["dynamic", "sparse", "top"]

    def home(g, n):
        f = follow_struct(g, n, home0)
        return f if f is not None else home0(g, n)

    def home0(g, n):
        p = g.file(n)
        it = g.info[n]
        name = it["gen"] or it["name"]
        of = it["of"] or ""
        if name == "ApproxEqTrait":
            return "core"
        if name == "MatrixKronecker" or of == "MatrixKronecker":
            return "kron"
        if name in VIEW_TRAITS or of in VIEW_TRAITS:
            return "views"
        if name in PRODUCT_TRAITS or of in PRODUCT_TRAITS:
            return "products"
        if name in NORM_MARKERS or (of == "Norm"):
            return "norms"
        if of.endswith("MatrixInfSup") or name == "MatrixInfSup":
            return "top"
        s = shape_of(p)
        if s:
            b = band(s, [3, 4, 5])
            if g.kind(n) == "inherent" and not name.endswith("EditTrait"):
                return ["base3", "base4", "base5", "base6"][b]
            return "core"
        stem = p[: -len(".cairo")].split("/")[-1]
        if p.startswith("base/dynamic"):
            return "dynamic"
        if p.startswith("base/"):
            if "statistics" in p:
                return "statistics"
            if "blas" in p:
                return "blas"
            if "solve" in p:
                return "la_solve"
            if "cg" in p:
                return "geometry"
            return "core"
        if p.startswith("geometry"):
            if stem in GEOMETRY_CORE:
                return "base3"
            if stem in GEOMETRY_ND:
                return "geometry_nd"
            return "geometry"
        if p.startswith("linalg/"):
            fam, b = linalg_key(p)
            if fam == "perm":
                return "la_perm"
            if fam == "hh":
                return "la_hh"
            if fam == "chol":
                return "la_chol"
            if b in ("s", "x"):
                if fam in ("lu", "qr", "eig", "svd"):
                    return f"la_{fam}_s"
                return "la_small_rest"
            if fam in ("lu", "qr", "eig", "svd"):
                return f"la_{fam}_{b}"
            return f"la_rest_{b}"
        if p.startswith("sparse") or p.startswith("io"):
            return "sparse"
        return "top"

    def pull(g):
        return {n: "base6" for n in g.nodes if g.label(n).split(":")[0] in ("Matrix6LuTrait",)}

    return types.SimpleNamespace(CRATES=crates, home=home, pull=pull)


v2 = make_v2()


BAND_SPLIT_TRAITS = {"SvdRightTrait"}
# impls of `Normed` that call the inherent `norm`: the moves give them the type-level kernel the
# other shapes' impls already use (an `#[inline(always)]` wrapper: no step change)
KERNEL_WRAPPERS = {"Matrix1Normed", "Vector5Normed", "Vector6Normed"}
MATRIX6_CLUSTER = {"Matrix6Trait", "Matrix6AngleTrait", "Matrix5x6Trait", "Matrix5x6AngleTrait",
                   "Matrix4x6Trait", "Matrix4x6AngleTrait", "Matrix3x6Trait", "Matrix3x6AngleTrait",
                   "Matrix2x6Trait", "Matrix2x6AngleTrait", "RowVector6Trait", "RowVector6AngleTrait"}


def band_of_label(label):
    import re as _re

    m = _re.findall(r"(\d)(?:x(\d))?", label)
    if not m:
        return None
    return max(max(int(a), int(b or a)) for a, b in m)


def make_v3():
    base = make_v2()
    la = [
        "la_solve", "la_perm", "la_hh", "la_lu_s", "la_chol", "la_qr_s", "la_eig_s", "la_svd_s",
        "la_small_rest", "la_lu_5", "la_lu_6", "la_qr_5", "la_qr_6", "la_eig_5", "la_eig_6",
        "la_svd_5", "la_svd_6", "la_rest_5", "la_rest_6",
    ]
    crates = [
        "core", "products", "base3", "base4", "geometry", "base5", "base6a", "solve", "perm",
        "base6", "views_b", "views_a", "kron", "norms", "geometry_nd", "statistics", "blas",
    ] + la + ["dynamic", "sparse", "top"]

    def home(g, n):
        h = base.home(g, n)
        it = g.info[n]
        name = it["gen"] or it["name"]
        of = it["of"] or ""
        p = g.file(n)
        if h == "base6" and name not in MATRIX6_CLUSTER:
            h = "base6a"
        if h == "views":
            h = "views_a" if (name == "FixedView" or of == "FixedView") else "views_b"
        if h == "la_solve":
            h = "solve"
        if h == "la_perm" or (p == "linalg/lu.cairo" and name.startswith("Perm")):
            h = "perm"
        if p.startswith("linalg/lu/lu6"):
            h = "base6"
        if p == "linalg/lu_steps.cairo":
            h = "perm"
        if name in KERNEL_WRAPPERS:
            h = "core"
        return h

    def ignore_edge(g, n, m):
        # crate-internal kernel traits holding every dimension (`SvdRightTrait::right2..right6`):
        # the moves split them per band, so a small SVD does not reach `Sym6`
        lm = g.label(m).split(":")[0]
        ln = g.label(n).split(":")[0]
        if ln in BAND_SPLIT_TRAITS:
            return True
        if ln in KERNEL_WRAPPERS and g.kind(m) == "inherent":
            return True
        return False

    return types.SimpleNamespace(CRATES=crates, home=home, ignore_edge=ignore_edge)


v3 = make_v3()


def make_v4():
    """Zero-break layout, fourth cut (measured, docs/SPLIT.md): TYPES per dimension band (structs,
    core-trait impls, products / transposed products / indexing anchored on the larger operand),
    inherent traits above them per band, the view / Kronecker / norm families in crates of their
    own (impls in the module of the trait or of the norm marker)."""
    v3 = make_v3()
    la = [c for c in v3.CRATES if c.startswith("la_")]
    crates = [
        "core", "base3", "t5", "base4", "geometry", "t6", "base5", "base6a", "solve", "perm",
        "base6", "views_b", "views_a", "kron", "norms", "geometry_nd", "statistics", "blas",
    ] + la + ["dynamic", "sparse", "top"]

    def arg_band(g, n):
        """Band of the largest nalgebra shape among an impl's arguments (None: no shape)."""
        best = None
        for a in g.info[n]["args"]:
            st = g.struct_node.get(a)
            if st is None:
                continue
            s = shape_of(g.file(st))
            if s:
                best = max(best or 0, max(s))
        return best

    def home(g, n):
        h = v3.home(g, n)
        it = g.info[n]
        of = it["of"] or ""
        name = it["gen"] or it["name"]
        p = g.file(n)
        s = shape_of(p)
        if h in ("core", "products"):
            b = None
            if of in PRODUCT_TRAITS or of in ("MatrixIndex",) or (g.kind(n) == "typebound" and s):
                b = arg_band(g, n)
            elif s and (it["kind"] == "struct" or name.endswith("EditTrait")):
                b = max(s)
            if b is not None:
                return "core" if b <= 4 else ("t5" if b == 5 else "t6")
            if h == "products":
                return "core"
        return h

    return types.SimpleNamespace(CRATES=crates, home=home, ignore_edge=v3.ignore_edge)


v4 = make_v4()


SOLVE_PERM_TRAITS = {"SolveKernel", "MatrixSolve", "PermuteRows", "PermuteColumns"}


def make_v5(la_group=None):
    """Zero-break layout, fifth cut: v4 plus the solve / permutation families spread over the type
    bands (declarations in `core`, impls in the module of their largest shape) and the linalg
    items of shared files banded by the size in their name."""
    v4 = make_v4()
    la_group = la_group or (lambda fam, b: f"la_{fam if fam in ('lu', 'chol', 'qr', 'eig', 'svd') else 'rest'}_{b}")
    la = []
    for b in ("s", "5", "6"):
        for fam in ("lu", "chol", "qr", "eig", "svd", "rest"):
            c = la_group(fam, b)
            if c not in la:
                la.append(c)
    crates = [c for c in v4.CRATES if not c.startswith("la_") and c not in ("solve", "perm")]
    i = crates.index("dynamic")
    crates = crates[:i] + ["la_hh"] + la + crates[i:]

    def shape_band(g, n):
        best = None
        for a in g.info[n]["args"]:
            st = g.struct_node.get(a)
            if st is not None and shape_of(g.file(st)):
                best = max(best or 0, max(shape_of(g.file(st))))
        return best

    def home(g, n):
        it = g.info[n]
        of = it["of"] or ""
        name = it["gen"] or it["name"]
        p = g.file(n)
        if name in SOLVE_PERM_TRAITS or (
            p in ("linalg/lu.cairo", "linalg/lu/perm1_5.cairo") and name.startswith("Perm")
        ):
            b = band_of_label(name)
            return "core" if b is None or b <= 4 else ("t5" if b == 5 else "t6")
        if of in SOLVE_PERM_TRAITS:
            b = shape_band(g, n) or 1
            return "core" if b <= 4 else ("t5" if b == 5 else "t6")
        if p == "linalg/lu_steps.cairo":
            return "core"
        if p == "linalg/givens.cairo":
            # public `GivensRotation` (upstream linalg::givens): declarations low, impls anchored
            # on their shape's module
            b = shape_band(g, n) if it["kind"] == "impl" else None
            return "core" if b is None or b <= 4 else ("t5" if b == 5 else "t6")
        h = v4.home(g, n)
        if h == "la_hh" and it["kind"] == "impl" and it["args"]:
            # impls of the crate-private Householder / Givens traits: with the linalg of their band
            b = shape_band(g, n)
            if b is not None and b > 4:
                return la_group("rest", str(b))
        if p.startswith("linalg/") and h.startswith("la_") and h not in ("la_hh",):
            fam, b = linalg_key(p)
            if b == "x":
                lb = band_of_label(name)
                b = "s" if lb is None or lb <= 4 else str(lb)
            key = {"lu": "lu", "chol": "chol", "qr": "qr", "eig": "eig", "svd": "svd"}.get(fam, "rest")
            return la_group(key, b)
        return h

    def ignore_edge(g, n, m):
        if v4.ignore_edge(g, n, m):
            return True
        # shared linalg files (`exp.cairo`...) import every size's traits: a method call of a
        # sized item resolved to a LARGER size is the over-approximation of plan.py (the prototype
        # build proves the cut)
        if g.file(n).startswith("linalg/"):
            bn = band_of_label(g.label(n).split(":")[0])
            bm = band_of_label(g.label(m).split(":")[0])
            if bn is not None and bm is not None and bm > max(bn, 4):
                return True
        return False

    return types.SimpleNamespace(CRATES=crates, home=home, ignore_edge=ignore_edge)


v5 = make_v5()


def make_v6():
    """v5, with `t6` under the line cap: the crate-private edit kernels (`MatrixRxCEditTrait`) with
    the row / column views that use them, the crate-private `LuSteps` impls with the LU of their
    size, `Perm6` and the permutation impls it anchors in a crate of their own above `t6`."""
    v5 = make_v5()
    crates = list(v5.CRATES)
    crates.insert(crates.index("base6"), "perm6")

    def home(g, n):
        h = v5.home(g, n)
        it = g.info[n]
        name = it["gen"] or it["name"]
        of = it["of"] or ""
        s = shape_of(g.file(n))
        if s and name.endswith("EditTrait"):
            return "views_b"
        if of == "LuSteps" and h in ("core", "t5", "t6"):
            return {"core": "la_lu_s", "t5": "la_lu_5", "t6": "base6"}[h]
        if name == "Perm6" or (p_is_perm(of) and "Perm6" in it["args"]) or name in ("Perm6Trait",):
            return "perm6"
        return h

    return types.SimpleNamespace(CRATES=crates, home=home, ignore_edge=v5.ignore_edge)


def p_is_perm(of):
    return of in ("PermuteRows", "PermuteColumns")


v6 = make_v6()


# the r x 5 family (their methods use `Matrix5Trait`): the crate above the 5-row family
DIM5_A = {"Matrix2x5", "Matrix3x5", "Matrix4x5", "RowVector5"}


# the r x 6 family (their methods use `Matrix6Trait`): the crate above `Matrix6` and its LU
DIM6_B = {"Matrix2x6", "Matrix3x6", "Matrix4x6", "Matrix5x6", "RowVector6"}


def make_final():
    """The recommended cut (docs/SPLIT.md §3): v6 with the crate names of the plan, the edit
    kernels with the methods of their band, `base5` in two, the dimension-6 linalg in three and
    the small linalg crates merged."""
    v5 = make_v5()
    crates = [
        "core", "dim3v", "dim3", "types5", "dim4", "geometry", "types6", "dim5", "dim5a", "dim6a",
        "dim6", "dim6b", "edition", "views", "kronecker", "norm", "geometry_nd", "statistics", "blas",
        "linalg", "linalg_svd", "linalg_pivot", "linalg_spectral", "linalg5", "linalg5_pivot",
        "linalg5_spectral", "linalg6", "linalg6_pivot", "linalg6_spectral", "dynamic", "sparse", "facade",
    ]
    rename = {
        "core": "core", "base3": "dim3", "t5": "types5", "base4": "dim4", "geometry": "geometry",
        "t6": "types6", "base5": "dim5", "base6a": "dim6a", "base6": "dim6", "perm6": "dim6",
        "views_b": "edition", "views_a": "views", "kron": "kronecker", "norms": "norm",
        "geometry_nd": "geometry_nd", "statistics": "statistics", "blas": "blas",
        "dynamic": "dynamic", "sparse": "sparse", "top": "facade", "la_hh": "linalg",
    }
    la_map = {
        "s": {"lu": "linalg", "chol": "linalg", "qr": "linalg", "eig": "linalg_svd",
              "svd": "linalg_svd", "rest": None},
        "5": {"lu": "linalg5", "chol": "linalg5", "qr": "linalg5", "eig": "linalg5",
              "svd": "linalg5", "rest": None},
        "6": {"lu": "linalg6", "chol": "linalg6", "qr": "linalg6", "eig": "linalg6",
              "svd": "linalg6", "rest": None},
    }
    PIVOT6 = ("col_piv_qr", "full_piv_lu", "lblt", "lu")

    def home(g, n):
        if g.kind(n) == "typebound" and g.info[n]["args"]:
            st = g.struct_node.get(g.info[n]["args"][0])
            if st is not None and st != n:
                return home_(g, st)
        return home_(g, n)

    def home_(g, n):
        h = v5.home(g, n)
        it = g.info[n]
        name = it["gen"] or it["name"]
        of = it["of"] or ""
        p = g.file(n)
        s = shape_of(p)
        if s and name.endswith("EditTrait"):
            b = max(s)
            return {5: "types5", 6: "dim6a"}.get(b, "core")
        if p in ("geometry/point4.cairo", "geometry/translation4.cairo", "geometry/point1.cairo",
                 "geometry/translation1.cairo"):
            # the 1- and 4-dimensional point / translation TYPES with the other types of dimension
            # <= 4 (their methods stay in `geometry`): `nalgebra_glam` converts them
            return "core"
        if of == "LuSteps":
            b = band_of_label(name) or 1
            return "linalg" if b <= 4 else ("linalg5" if b == 5 else "dim6")
        if name in ("Perm6", "Perm6Trait") or (of in ("PermuteRows", "PermuteColumns") and "Perm6" in it["args"]):
            return "dim6"
        if h == "base3" and s and p.split("/")[-1] in (
            "vector2.cairo", "vector3.cairo", "matrix1.cairo"
        ):
            # the vector methods of dimensions 1..3 (the rotations use them): below the rotation
            # knot of `dim3`
            return "dim3v"
        if h == "base6" and s and max(s) == 6:
            shape = p.split("/")[-1][:-6]
            tname = "".join(w.capitalize() for w in shape.split("_"))
            if tname in DIM6_B:
                return "dim6b"
        if h == "base5" and s:
            shape = p.split("/")[-1][:-6]
            tname = "".join(w.capitalize() for w in shape.split("_"))
            return "dim5a" if tname in DIM5_A else "dim5"
        if h.startswith("la_") and h != "la_hh":
            fam, b = h[3:].rsplit("_", 1)
            c = la_map[b][fam]
            if c is None:
                stem = p.split("/")[1].split(".")[0]
                pre = {"s": "linalg", "5": "linalg5", "6": "linalg6"}[b]
                c = f"{pre}_pivot" if stem in PIVOT6 else f"{pre}_spectral"
            return c
        return rename[h]

    return types.SimpleNamespace(CRATES=crates, home=home, ignore_edge=v5.ignore_edge)


final = make_final()
