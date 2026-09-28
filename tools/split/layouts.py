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
