"""The shape model shared by the library templates, the prototype and the generated tests.

A static shape is `(rows, columns)` with 1 <= rows, columns <= 6: the 36 distinct shapes behind
upstream's 48 matrix aliases (`src/base/alias.rs`). ONE Cairo struct per shape (the canonical
name), the other upstream names being Cairo `type` aliases (`DESIGN.md` §1.1).
"""

import re
from dataclasses import dataclass

COORDS = "xyzwab"


@dataclass(frozen=True, order=True)
class Shape:
    r: int
    c: int

    @property
    def n(self) -> int:
        return self.r * self.c

    @property
    def is_vector(self) -> bool:
        """One column or one row: components named `x, y, z, w, a, b` like upstream's Deref."""
        return self.r == 1 or self.c == 1

    @property
    def is_column(self) -> bool:
        """A column vector (upstream `Vector`): `Matrix1` included."""
        return self.c == 1

    @property
    def is_row(self) -> bool:
        """A row vector (upstream `RowSVector`): `Matrix1` included."""
        return self.r == 1

    @property
    def is_square(self) -> bool:
        return self.r == self.c

    @property
    def name(self) -> str:
        """The canonical struct: upstream's aliases of one shape are ONE Cairo type."""
        if self.r == 1 and self.c == 1:
            return "Matrix1"
        if self.c == 1:
            return f"Vector{self.r}"
        if self.r == 1:
            return f"RowVector{self.c}"
        if self.is_square:
            return f"Matrix{self.r}"
        return f"Matrix{self.r}x{self.c}"

    @property
    def aliases(self) -> list[str]:
        """The other upstream names of the shape, as Cairo type aliases (upstream has no
        `Matrix1x1`)."""
        if self.r == 1 and self.c == 1:
            return ["Vector1", "RowVector1"]
        if self.c == 1:
            return [f"Matrix{self.r}x1"]
        if self.r == 1:
            return [f"Matrix1x{self.c}"]
        return []

    @property
    def unit_alias(self) -> str | None:
        """`UnitVectorN = Unit<VectorN>` for the column vectors (upstream `alias.rs`)."""
        return f"UnitVector{self.r}" if self.c == 1 else None

    @property
    def module(self) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.name).lower()

    @property
    def kind(self) -> str:
        if self.r == 1 and self.c == 1:
            return "1x1 matrix"
        if self.c == 1:
            return f"{self.r}-dimensional column vector"
        if self.r == 1:
            return f"{self.c}-dimensional row vector"
        return f"{self.r}x{self.c} matrix"

    def f(self, i: int, j: int) -> str:
        """Field of the component at row `i`, column `j` (0-based)."""
        if self.c == 1:
            return COORDS[i]
        if self.r == 1:
            return COORDS[j]
        return f"m{i + 1}{j + 1}"

    @property
    def fields(self) -> list[str]:
        """Column-major: the declaration order, hence upstream's `Serde` order."""
        return [self.f(i, j) for j in range(self.c) for i in range(self.r)]

    @property
    def row_major(self) -> list[str]:
        return [self.f(i, j) for i in range(self.r) for j in range(self.c)]

    def transposed(self) -> "Shape":
        return Shape(self.c, self.r)

    def lit(self, values: dict[str, str]) -> str:
        """Struct literal in declaration order."""
        body = ", ".join(f"{k}: {values[k]}" for k in self.fields)
        return f"{self.name} {{ {body} }}"


ALL_SHAPES = [Shape(r, c) for r in range(1, 7) for c in range(1, 7)]
