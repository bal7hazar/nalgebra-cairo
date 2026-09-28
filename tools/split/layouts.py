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
