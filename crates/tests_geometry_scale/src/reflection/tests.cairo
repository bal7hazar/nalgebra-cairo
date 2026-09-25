//! Tests of `Reflection1` .. `Reflection6` (WP 8.4-P10) through the public API: exact cases, the
//! identities the in-place kernels must satisfy on EVERY shape (columns of each shape reflected
//! like the same vectors, involution, the reflection of the dot product with the axis, the
//! `sign = 1` / `sign = -1` forms, the scratch vector of `reflect_rows`) and the oracle vectors of
//! `tools/oracle` (suite `scale_reflection`, upstream nalgebra 0.35 on the same raw inputs,
//! tolerance in ulp).

use fixed::Fixed;
use nalgebra::base::matrix1::{Matrix1, Matrix1Trait};
use nalgebra::base::row_vector2::RowVector2;
use nalgebra::base::row_vector3::RowVector3;
use nalgebra::base::row_vector4::RowVector4;
use nalgebra::base::row_vector5::RowVector5;
use nalgebra::base::row_vector6::RowVector6;
use nalgebra::base::unit::{Unit, UnitTrait};
use nalgebra::base::vector2::{Vector2, Vector2Trait};
use nalgebra::base::vector3::{Vector3, Vector3Trait};
use nalgebra::base::vector4::{Vector4, Vector4Trait};
use nalgebra::base::vector5::{Vector5, Vector5Trait};
use nalgebra::base::vector6::{Vector6, Vector6Trait};
use nalgebra::geometry::reflection1::{
    Reflection1, Reflection1Columns, Reflection1Rows, Reflection1Trait,
};
use nalgebra::geometry::reflection2::{
    Reflection2, Reflection2Columns, Reflection2Rows, Reflection2Trait,
};
use nalgebra::geometry::reflection3::{
    Reflection3, Reflection3Columns, Reflection3Rows, Reflection3Trait,
};
use nalgebra::geometry::reflection4::{
    Reflection4, Reflection4Columns, Reflection4Rows, Reflection4Trait,
};
use nalgebra::geometry::reflection5::{
    Reflection5, Reflection5Columns, Reflection5Rows, Reflection5Trait,
};
use nalgebra::geometry::reflection6::{
    Reflection6, Reflection6Columns, Reflection6Rows, Reflection6Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{abs_raw, fx, int, ulp_diff};
use crate::common::{
    err_mat1x1, err_mat1x2, err_mat1x3, err_mat1x4, err_mat1x5, err_mat1x6, err_mat2x1, err_mat2x2,
    err_mat2x3, err_mat2x4, err_mat2x5, err_mat2x6, err_mat3x1, err_mat3x2, err_mat3x3, err_mat3x4,
    err_mat3x5, err_mat3x6, err_mat4x1, err_mat4x2, err_mat4x3, err_mat4x4, err_mat4x5, err_mat4x6,
    err_mat5x1, err_mat5x2, err_mat5x3, err_mat5x4, err_mat5x5, err_mat5x6, err_mat6x1, err_mat6x2,
    err_mat6x3, err_mat6x4, err_mat6x5, err_mat6x6, mat1x1, mat1x2, mat1x3, mat1x4, mat1x5, mat1x6,
    mat2x1, mat2x2, mat2x3, mat2x4, mat2x5, mat2x6, mat3x1, mat3x2, mat3x3, mat3x4, mat3x5, mat3x6,
    mat4x1, mat4x2, mat4x3, mat4x4, mat4x5, mat4x6, mat5x1, mat5x2, mat5x3, mat5x4, mat5x5, mat5x6,
    mat6x1, mat6x2, mat6x3, mat6x4, mat6x5, mat6x6, pt1, pt2, pt3, pt4, pt5, pt6, vec1, vec2, vec3,
    vec4, vec5, vec6,
};
use crate::oracle;

/// The excess of `err` over `tol`, printed when positive.
fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = nalgebra_tests_utils::excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}

// --- Reflection1

fn axis1() -> Unit<Matrix1<Fixed>> {
    UnitTrait::new_normalize(vec1((4294967296,)))
}

fn refl1() -> Reflection1<Fixed> {
    Reflection1Trait::new(axis1(), fx(0x12345678))
}

#[test]
fn test_reflection1_api() {
    let axis = Unit { value: vec1((0x100000000,)) };
    let r = Reflection1Trait::new(axis, int(3));
    assert!(r.axis() == axis.value && r.bias() == int(3));
    let through = Reflection1Trait::new_containing_point(axis, pt1((21474836480,)));
    assert!(through.bias() == int(5) && through.axis() == axis.value);
    // The plane x = 3: x -> 3 - (x - 3), the other coordinates unchanged.
    let mut v = vec1((30064771072,));
    r.reflect(ref v);
    assert!(v == vec1((-4294967296,)));
    let mut s = vec1((30064771072,));
    r.reflect_with_sign(ref s, int(-1));
    assert!(s == vec1((4294967296,)));
    let mut row = Matrix1 { x: int(7) };
    let mut work = Matrix1 { x: int(0) };
    r.reflect_rows(ref row, ref work);
    assert!(work.x == int(4));
    assert!(row == Matrix1 { x: int(-1) });
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_reflection1_reflect_overflow_panics() {
    let r = Reflection1Trait::new(Unit { value: vec1((0x100000000,)) }, int(-2000000000));
    let mut v = black_box(vec1((2000000000 * 0x100000000,)));
    r.reflect(ref v);
}

#[test]
fn test_reflection1_columns_of_matrix1() {
    let r = refl1();
    let m = mat1x1([[1365889077]]);
    let mut got = m;
    r.reflect(ref got);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat1x1(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat1x1(minus, -got) <= 3);
}

#[test]
fn test_reflection1_columns_of_rowvector2() {
    let r = refl1();
    let m = mat1x2([[4438431132, -1649386749]]);
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec1((m.x.raw,));
    r.reflect(ref col1);
    assert!(got.x == col1.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec1((m.y.raw,));
    r.reflect(ref col2);
    assert!(got.y == col2.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.y.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.y.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat1x2(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat1x2(minus, -got) <= 3);
}

#[test]
fn test_reflection1_columns_of_rowvector3() {
    let r = refl1();
    let m = mat1x3([[-2521801981, 1423155306, -4664662575]]);
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec1((m.x.raw,));
    r.reflect(ref col1);
    assert!(got.x == col1.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec1((m.y.raw,));
    r.reflect(ref col2);
    assert!(got.y == col2.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.y.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.y.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec1((m.z.raw,));
    r.reflect(ref col3);
    assert!(got.z == col3.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.z.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.z.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat1x3(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat1x3(minus, -got) <= 3);
}

#[test]
fn test_reflection1_columns_of_rowvector4() {
    let r = refl1();
    let m = mat1x4([[550740074, 4495697361, -1592120520, 2352836767]]);
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec1((m.x.raw,));
    r.reflect(ref col1);
    assert!(got.x == col1.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec1((m.y.raw,));
    r.reflect(ref col2);
    assert!(got.y == col2.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.y.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.y.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec1((m.z.raw,));
    r.reflect(ref col3);
    assert!(got.z == col3.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.z.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.z.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec1((m.w.raw,));
    r.reflect(ref col4);
    assert!(got.w == col4.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.w.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.w.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat1x4(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat1x4(minus, -got) <= 3);
}

#[test]
fn test_reflection1_columns_of_rowvector5() {
    let r = refl1();
    let m = mat1x5([[3623282129, -2464535752, 1480421535, -4607396346, -662439059]]);
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec1((m.x.raw,));
    r.reflect(ref col1);
    assert!(got.x == col1.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec1((m.y.raw,));
    r.reflect(ref col2);
    assert!(got.y == col2.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.y.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.y.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec1((m.z.raw,));
    r.reflect(ref col3);
    assert!(got.z == col3.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.z.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.z.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec1((m.w.raw,));
    r.reflect(ref col4);
    assert!(got.w == col4.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.w.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.w.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec1((m.a.raw,));
    r.reflect(ref col5);
    assert!(got.a == col5.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.a.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.a.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat1x5(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat1x5(minus, -got) <= 3);
}

#[test]
fn test_reflection1_columns_of_rowvector6() {
    let r = refl1();
    let m = mat1x6([[-3336950984, 608006303, 4552963590, -1534854291, 2410102996, -3677714885]]);
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec1((m.x.raw,));
    r.reflect(ref col1);
    assert!(got.x == col1.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec1((m.y.raw,));
    r.reflect(ref col2);
    assert!(got.y == col2.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.y.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.y.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec1((m.z.raw,));
    r.reflect(ref col3);
    assert!(got.z == col3.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.z.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.z.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec1((m.w.raw,));
    r.reflect(ref col4);
    assert!(got.w == col4.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.w.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.w.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec1((m.a.raw,));
    r.reflect(ref col5);
    assert!(got.a == col5.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.a.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.a.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col6 = vec1((m.b.raw,));
    r.reflect(ref col6);
    assert!(got.b == col6.x);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.b.raw,))) - r.bias();
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.b.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat1x6(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat1x6(minus, -got) <= 3);
}

#[test]
fn test_reflection1_rows_of_matrix1() {
    let r = refl1();
    let m = mat1x1([[4438431132]]);
    let mut got = m;
    let mut work = vec1((0,));
    r.reflect_rows(ref got, ref work);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    assert!(work.x == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat1x1(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat1x1(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection1_rows_of_vector2() {
    let r = refl1();
    let m = mat2x1([[-2521801981], [3604193386]]);
    let mut got = m;
    let mut work = vec2((0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    assert!(work.x == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.y.raw,))) - r.bias();
    assert!(work.y == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.y.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat2x1(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat2x1(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection1_rows_of_vector3() {
    let r = refl1();
    let m = mat3x1([[550740074], [-3356039727], [2769955640]]);
    let mut got = m;
    let mut work = vec3((0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    assert!(work.x == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.y.raw,))) - r.bias();
    assert!(work.y == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.y.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.z.raw,))) - r.bias();
    assert!(work.z == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.z.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat3x1(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat3x1(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection1_rows_of_vector4() {
    let r = refl1();
    let m = mat4x1([[3623282129], [-283497672], [-4190277473], [1935717894]]);
    let mut got = m;
    let mut work = vec4((0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    assert!(work.x == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.y.raw,))) - r.bias();
    assert!(work.y == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.y.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.z.raw,))) - r.bias();
    assert!(work.z == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.z.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.w.raw,))) - r.bias();
    assert!(work.w == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.w.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat4x1(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat4x1(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection1_rows_of_vector5() {
    let r = refl1();
    let m = mat5x1([[-3336950984], [2789044383], [-1117735418], [5008259949], [1101480148]]);
    let mut got = m;
    let mut work = vec5((0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    assert!(work.x == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.y.raw,))) - r.bias();
    assert!(work.y == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.y.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.z.raw,))) - r.bias();
    assert!(work.z == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.z.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.w.raw,))) - r.bias();
    assert!(work.w == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.w.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.a.raw,))) - r.bias();
    assert!(work.a == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.a.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat5x1(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat5x1(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection1_rows_of_vector6() {
    let r = refl1();
    let m = mat6x1(
        [[-264408929], [-4171188730], [1954806637], [-1951973164], [4174022203], [267242402]],
    );
    let mut got = m;
    let mut work = vec6((0, 0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.x.raw,))) - r.bias();
    assert!(work.x == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.x.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.y.raw,))) - r.bias();
    assert!(work.y == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.y.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.z.raw,))) - r.bias();
    assert!(work.z == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.z.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.w.raw,))) - r.bias();
    assert!(work.w == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.w.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.a.raw,))) - r.bias();
    assert!(work.a == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.a.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Matrix1Trait::dot(r.axis(), vec1((m.b.raw,))) - r.bias();
    assert!(work.b == d0);
    let d1 = Matrix1Trait::dot(r.axis(), vec1((got.b.raw,))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat6x1(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat6x1(minus, -got) <= 3 && work4 == work);
}

// --- Reflection2

fn axis2() -> Unit<Vector2<Fixed>> {
    UnitTrait::new_normalize(vec2((4294967296, -4294967296)))
}

fn refl2() -> Reflection2<Fixed> {
    Reflection2Trait::new(axis2(), fx(0x12345678))
}

#[test]
fn test_reflection2_api() {
    let axis = Unit { value: vec2((0x100000000, 0)) };
    let r = Reflection2Trait::new(axis, int(3));
    assert!(r.axis() == axis.value && r.bias() == int(3));
    let through = Reflection2Trait::new_containing_point(axis, pt2((21474836480, 4294967296)));
    assert!(through.bias() == int(5) && through.axis() == axis.value);
    // The plane x = 3: x -> 3 - (x - 3), the other coordinates unchanged.
    let mut v = vec2((30064771072, 4294967296));
    r.reflect(ref v);
    assert!(v == vec2((-4294967296, 4294967296)));
    let mut s = vec2((30064771072, 4294967296));
    r.reflect_with_sign(ref s, int(-1));
    assert!(s == vec2((4294967296, -4294967296)));
    let mut row = RowVector2 { x: int(7), y: int(1) };
    let mut work = Matrix1 { x: int(0) };
    r.reflect_rows(ref row, ref work);
    assert!(work.x == int(4));
    assert!(row == RowVector2 { x: int(-1), y: int(1) });
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_reflection2_reflect_overflow_panics() {
    let r = Reflection2Trait::new(Unit { value: vec2((0x100000000, 0)) }, int(-2000000000));
    let mut v = black_box(vec2((2000000000 * 0x100000000, 0)));
    r.reflect(ref v);
}

#[test]
fn test_reflection2_columns_of_vector2() {
    let r = refl2();
    let m = mat2x1([[4438431132], [531651331]]);
    let mut got = m;
    r.reflect(ref got);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.x.raw, m.y.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.x.raw, got.y.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat2x1(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat2x1(minus, -got) <= 3);
}

#[test]
fn test_reflection2_columns_of_matrix2() {
    let r = refl2();
    let m = mat2x2([[-2521801981, 1423155306], [3604193386, -2464535752]]);
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec2((m.m11.raw, m.m21.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m11.raw, m.m21.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m11.raw, got.m21.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec2((m.m12.raw, m.m22.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m12.raw, m.m22.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m12.raw, got.m22.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat2x2(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat2x2(minus, -got) <= 3);
}

#[test]
fn test_reflection2_columns_of_matrix2x3() {
    let r = refl2();
    let m = mat2x3([[550740074, 4495697361, -1592120520], [-3356039727, 608006303, 4572052333]]);
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec2((m.m11.raw, m.m21.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m11.raw, m.m21.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m11.raw, got.m21.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec2((m.m12.raw, m.m22.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m12.raw, m.m22.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m12.raw, got.m22.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec2((m.m13.raw, m.m23.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m13.raw, m.m23.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m13.raw, got.m23.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat2x3(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat2x3(minus, -got) <= 3);
}

#[test]
fn test_reflection2_columns_of_matrix2x4() {
    let r = refl2();
    let m = mat2x4(
        [
            [3623282129, -2464535752, 1480421535, -4607396346],
            [-283497672, 3680548358, -2388180780, 1575865250],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec2((m.m11.raw, m.m21.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m11.raw, m.m21.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m11.raw, got.m21.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec2((m.m12.raw, m.m22.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m12.raw, m.m22.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m12.raw, got.m22.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec2((m.m13.raw, m.m23.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m13.raw, m.m23.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m13.raw, got.m23.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec2((m.m14.raw, m.m24.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m14.raw, m.m24.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m14.raw, got.m24.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat2x4(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat2x4(minus, -got) <= 3);
}

#[test]
fn test_reflection2_columns_of_matrix2x5() {
    let r = refl2();
    let m = mat2x5(
        [
            [-3336950984, 608006303, 4552963590, -1534854291, 2410102996],
            [2789044383, -3279684755, 684361275, 4648407305, -1420321833],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec2((m.m11.raw, m.m21.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m11.raw, m.m21.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m11.raw, got.m21.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec2((m.m12.raw, m.m22.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m12.raw, m.m22.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m12.raw, got.m22.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec2((m.m13.raw, m.m23.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m13.raw, m.m23.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m13.raw, got.m23.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec2((m.m14.raw, m.m24.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m14.raw, m.m24.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m14.raw, got.m24.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec2((m.m15.raw, m.m25.raw));
    r.reflect(ref col5);
    assert!(got.m15 == col5.x);
    assert!(got.m25 == col5.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m15.raw, m.m25.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m15.raw, got.m25.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat2x5(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat2x5(minus, -got) <= 3);
}

#[test]
fn test_reflection2_columns_of_matrix2x6() {
    let r = refl2();
    let m = mat2x6(
        [
            [-264408929, 3680548358, -2407269523, 1537687764, -4550130117, -605172830],
            [-4171188730, -207142700, 3756903330, -2311825808, 1652220222, -4416508916],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec2((m.m11.raw, m.m21.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m11.raw, m.m21.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m11.raw, got.m21.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec2((m.m12.raw, m.m22.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m12.raw, m.m22.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m12.raw, got.m22.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec2((m.m13.raw, m.m23.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m13.raw, m.m23.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m13.raw, got.m23.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec2((m.m14.raw, m.m24.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m14.raw, m.m24.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m14.raw, got.m24.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec2((m.m15.raw, m.m25.raw));
    r.reflect(ref col5);
    assert!(got.m15 == col5.x);
    assert!(got.m25 == col5.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m15.raw, m.m25.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m15.raw, got.m25.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col6 = vec2((m.m16.raw, m.m26.raw));
    r.reflect(ref col6);
    assert!(got.m16 == col6.x);
    assert!(got.m26 == col6.y);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m16.raw, m.m26.raw))) - r.bias();
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m16.raw, got.m26.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat2x6(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat2x6(minus, -got) <= 3);
}

#[test]
fn test_reflection2_rows_of_rowvector2() {
    let r = refl2();
    let m = mat1x2([[550740074, 4495697361]]);
    let mut got = m;
    let mut work = vec1((0,));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.x.raw, m.y.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.x.raw, got.y.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat1x2(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat1x2(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection2_rows_of_matrix2() {
    let r = refl2();
    let m = mat2x2([[3623282129, -2464535752], [-283497672, 3680548358]]);
    let mut got = m;
    let mut work = vec2((0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m11.raw, m.m12.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m11.raw, got.m12.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m21.raw, m.m22.raw))) - r.bias();
    assert!(work.y == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m21.raw, got.m22.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat2x2(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat2x2(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection2_rows_of_matrix3x2() {
    let r = refl2();
    let m = mat3x2(
        [[-3336950984, 608006303], [2789044383, -3279684755], [-1117735418, 2865399355]],
    );
    let mut got = m;
    let mut work = vec3((0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m11.raw, m.m12.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m11.raw, got.m12.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m21.raw, m.m22.raw))) - r.bias();
    assert!(work.y == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m21.raw, got.m22.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m31.raw, m.m32.raw))) - r.bias();
    assert!(work.z == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m31.raw, got.m32.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat3x2(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat3x2(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection2_rows_of_matrix4x2() {
    let r = refl2();
    let m = mat4x2(
        [
            [-264408929, 3680548358], [-4171188730, -207142700], [1954806637, -4094833758],
            [-1951973164, 2050250352],
        ],
    );
    let mut got = m;
    let mut work = vec4((0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m11.raw, m.m12.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m11.raw, got.m12.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m21.raw, m.m22.raw))) - r.bias();
    assert!(work.y == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m21.raw, got.m22.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m31.raw, m.m32.raw))) - r.bias();
    assert!(work.z == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m31.raw, got.m32.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m41.raw, m.m42.raw))) - r.bias();
    assert!(work.w == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m41.raw, got.m42.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat4x2(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat4x2(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection2_rows_of_matrix5x2() {
    let r = refl2();
    let m = mat5x2(
        [
            [2808133126, -3279684755], [-1098646675, 2865399355], [5027348692, -1022291703],
            [1120568891, 5122792407], [-2786210910, 1235101349],
        ],
    );
    let mut got = m;
    let mut work = vec5((0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m11.raw, m.m12.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m11.raw, got.m12.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m21.raw, m.m22.raw))) - r.bias();
    assert!(work.y == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m21.raw, got.m22.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m31.raw, m.m32.raw))) - r.bias();
    assert!(work.z == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m31.raw, got.m32.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m41.raw, m.m42.raw))) - r.bias();
    assert!(work.w == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m41.raw, got.m42.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m51.raw, m.m52.raw))) - r.bias();
    assert!(work.a == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m51.raw, got.m52.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat5x2(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat5x2(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection2_rows_of_matrix6x2() {
    let r = refl2();
    let m = mat6x2(
        [
            [-4152099987, -207142700], [1973895380, -4094833758], [-1932884421, 2050250352],
            [4193110946, -1837440706], [286331145, 4307643404], [-3620448656, 419952346],
        ],
    );
    let mut got = m;
    let mut work = vec6((0, 0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m11.raw, m.m12.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m11.raw, got.m12.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m21.raw, m.m22.raw))) - r.bias();
    assert!(work.y == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m21.raw, got.m22.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m31.raw, m.m32.raw))) - r.bias();
    assert!(work.z == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m31.raw, got.m32.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m41.raw, m.m42.raw))) - r.bias();
    assert!(work.w == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m41.raw, got.m42.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m51.raw, m.m52.raw))) - r.bias();
    assert!(work.a == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m51.raw, got.m52.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector2Trait::dot(r.axis(), vec2((m.m61.raw, m.m62.raw))) - r.bias();
    assert!(work.b == d0);
    let d1 = Vector2Trait::dot(r.axis(), vec2((got.m61.raw, got.m62.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat6x2(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat6x2(minus, -got) <= 3 && work4 == work);
}

// --- Reflection3

fn axis3() -> Unit<Vector3<Fixed>> {
    UnitTrait::new_normalize(vec3((4294967296, -4294967296, 12884901888)))
}

fn refl3() -> Reflection3<Fixed> {
    Reflection3Trait::new(axis3(), fx(0x12345678))
}

#[test]
fn test_reflection3_api() {
    let axis = Unit { value: vec3((0x100000000, 0, 0)) };
    let r = Reflection3Trait::new(axis, int(3));
    assert!(r.axis() == axis.value && r.bias() == int(3));
    let through = Reflection3Trait::new_containing_point(
        axis, pt3((21474836480, 4294967296, 8589934592)),
    );
    assert!(through.bias() == int(5) && through.axis() == axis.value);
    // The plane x = 3: x -> 3 - (x - 3), the other coordinates unchanged.
    let mut v = vec3((30064771072, 4294967296, 4294967296));
    r.reflect(ref v);
    assert!(v == vec3((-4294967296, 4294967296, 4294967296)));
    let mut s = vec3((30064771072, 4294967296, 4294967296));
    r.reflect_with_sign(ref s, int(-1));
    assert!(s == vec3((4294967296, -4294967296, -4294967296)));
    let mut row = RowVector3 { x: int(7), y: int(1), z: int(1) };
    let mut work = Matrix1 { x: int(0) };
    r.reflect_rows(ref row, ref work);
    assert!(work.x == int(4));
    assert!(row == RowVector3 { x: int(-1), y: int(1), z: int(1) });
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_reflection3_reflect_overflow_panics() {
    let r = Reflection3Trait::new(Unit { value: vec3((0x100000000, 0, 0)) }, int(-2000000000));
    let mut v = black_box(vec3((2000000000 * 0x100000000, 0, 0)));
    r.reflect(ref v);
}

#[test]
fn test_reflection3_columns_of_vector3() {
    let r = refl3();
    let m = mat3x1([[-2521801981], [3604193386], [-302586415]]);
    let mut got = m;
    r.reflect(ref got);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.x.raw, m.y.raw, m.z.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.x.raw, got.y.raw, got.z.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat3x1(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat3x1(minus, -got) <= 3);
}

#[test]
fn test_reflection3_columns_of_matrix3x2() {
    let r = refl3();
    let m = mat3x2([[550740074, 4495697361], [-3356039727, 608006303], [2769955640, -3279684755]]);
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec3((m.m11.raw, m.m21.raw, m.m31.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m11.raw, m.m21.raw, m.m31.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m11.raw, got.m21.raw, got.m31.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec3((m.m12.raw, m.m22.raw, m.m32.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m12.raw, m.m22.raw, m.m32.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m12.raw, got.m22.raw, got.m32.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat3x2(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat3x2(minus, -got) <= 3);
}

#[test]
fn test_reflection3_columns_of_matrix3() {
    let r = refl3();
    let m = mat3x3(
        [
            [3623282129, -2464535752, 1480421535], [-283497672, 3680548358, -2388180780],
            [-4190277473, -207142700, 3775992073],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec3((m.m11.raw, m.m21.raw, m.m31.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m11.raw, m.m21.raw, m.m31.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m11.raw, got.m21.raw, got.m31.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec3((m.m12.raw, m.m22.raw, m.m32.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m12.raw, m.m22.raw, m.m32.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m12.raw, got.m22.raw, got.m32.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec3((m.m13.raw, m.m23.raw, m.m33.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m13.raw, m.m23.raw, m.m33.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m13.raw, got.m23.raw, got.m33.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat3x3(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat3x3(minus, -got) <= 3);
}

#[test]
fn test_reflection3_columns_of_matrix3x4() {
    let r = refl3();
    let m = mat3x4(
        [
            [-3336950984, 608006303, 4552963590, -1534854291],
            [2789044383, -3279684755, 684361275, 4648407305],
            [-1117735418, 2865399355, -3184241040, 798893733],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec3((m.m11.raw, m.m21.raw, m.m31.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m11.raw, m.m21.raw, m.m31.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m11.raw, got.m21.raw, got.m31.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec3((m.m12.raw, m.m22.raw, m.m32.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m12.raw, m.m22.raw, m.m32.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m12.raw, got.m22.raw, got.m32.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec3((m.m13.raw, m.m23.raw, m.m33.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m13.raw, m.m23.raw, m.m33.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m13.raw, got.m23.raw, got.m33.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec3((m.m14.raw, m.m24.raw, m.m34.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m14.raw, m.m24.raw, m.m34.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m14.raw, got.m24.raw, got.m34.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat3x4(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat3x4(minus, -got) <= 3);
}

#[test]
fn test_reflection3_columns_of_matrix3x5() {
    let r = refl3();
    let m = mat3x5(
        [
            [-264408929, 3680548358, -2407269523, 1537687764, -4550130117],
            [-4171188730, -207142700, 3756903330, -2311825808, 1652220222],
            [1954806637, -4094833758, -111698985, 3871435788, -2178204607],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec3((m.m11.raw, m.m21.raw, m.m31.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m11.raw, m.m21.raw, m.m31.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m11.raw, got.m21.raw, got.m31.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec3((m.m12.raw, m.m22.raw, m.m32.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m12.raw, m.m22.raw, m.m32.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m12.raw, got.m22.raw, got.m32.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec3((m.m13.raw, m.m23.raw, m.m33.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m13.raw, m.m23.raw, m.m33.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m13.raw, got.m23.raw, got.m33.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec3((m.m14.raw, m.m24.raw, m.m34.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m14.raw, m.m24.raw, m.m34.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m14.raw, got.m24.raw, got.m34.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec3((m.m15.raw, m.m25.raw, m.m35.raw));
    r.reflect(ref col5);
    assert!(got.m15 == col5.x);
    assert!(got.m25 == col5.y);
    assert!(got.m35 == col5.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m15.raw, m.m25.raw, m.m35.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m15.raw, got.m25.raw, got.m35.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat3x5(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat3x5(minus, -got) <= 3);
}

#[test]
fn test_reflection3_columns_of_matrix3x6() {
    let r = refl3();
    let m = mat3x6(
        [
            [2808133126, -3279684755, 665272532, 4610229819, -1477588062, 2467369225],
            [-1098646675, 2865399355, -3203329783, 760716247, 4724762277, -1343966861],
            [5027348692, -1022291703, 2960843070, -3088797325, 894337448, 4877472221],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec3((m.m11.raw, m.m21.raw, m.m31.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m11.raw, m.m21.raw, m.m31.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m11.raw, got.m21.raw, got.m31.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec3((m.m12.raw, m.m22.raw, m.m32.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m12.raw, m.m22.raw, m.m32.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m12.raw, got.m22.raw, got.m32.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec3((m.m13.raw, m.m23.raw, m.m33.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m13.raw, m.m23.raw, m.m33.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m13.raw, got.m23.raw, got.m33.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec3((m.m14.raw, m.m24.raw, m.m34.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m14.raw, m.m24.raw, m.m34.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m14.raw, got.m24.raw, got.m34.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec3((m.m15.raw, m.m25.raw, m.m35.raw));
    r.reflect(ref col5);
    assert!(got.m15 == col5.x);
    assert!(got.m25 == col5.y);
    assert!(got.m35 == col5.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m15.raw, m.m25.raw, m.m35.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m15.raw, got.m25.raw, got.m35.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col6 = vec3((m.m16.raw, m.m26.raw, m.m36.raw));
    r.reflect(ref col6);
    assert!(got.m16 == col6.x);
    assert!(got.m26 == col6.y);
    assert!(got.m36 == col6.z);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m16.raw, m.m26.raw, m.m36.raw))) - r.bias();
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m16.raw, got.m26.raw, got.m36.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat3x6(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat3x6(minus, -got) <= 3);
}

#[test]
fn test_reflection3_rows_of_rowvector3() {
    let r = refl3();
    let m = mat1x3([[-3336950984, 608006303, 4552963590]]);
    let mut got = m;
    let mut work = vec1((0,));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.x.raw, m.y.raw, m.z.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.x.raw, got.y.raw, got.z.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat1x3(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat1x3(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection3_rows_of_matrix2x3() {
    let r = refl3();
    let m = mat2x3([[-264408929, 3680548358, -2407269523], [-4171188730, -207142700, 3756903330]]);
    let mut got = m;
    let mut work = vec2((0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m11.raw, m.m12.raw, m.m13.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m11.raw, got.m12.raw, got.m13.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m21.raw, m.m22.raw, m.m23.raw))) - r.bias();
    assert!(work.y == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m21.raw, got.m22.raw, got.m23.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat2x3(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat2x3(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection3_rows_of_matrix3() {
    let r = refl3();
    let m = mat3x3(
        [
            [2808133126, -3279684755, 665272532], [-1098646675, 2865399355, -3203329783],
            [5027348692, -1022291703, 2960843070],
        ],
    );
    let mut got = m;
    let mut work = vec3((0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m11.raw, m.m12.raw, m.m13.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m11.raw, got.m12.raw, got.m13.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m21.raw, m.m22.raw, m.m23.raw))) - r.bias();
    assert!(work.y == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m21.raw, got.m22.raw, got.m23.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m31.raw, m.m32.raw, m.m33.raw))) - r.bias();
    assert!(work.z == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m31.raw, got.m32.raw, got.m33.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat3x3(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat3x3(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection3_rows_of_matrix4x3() {
    let r = refl3();
    let m = mat4x3(
        [
            [-4152099987, -207142700, 3737814587], [1973895380, -4094833758, -130787728],
            [-1932884421, 2050250352, -3999390043], [4193110946, -1837440706, 2164782810],
        ],
    );
    let mut got = m;
    let mut work = vec4((0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m11.raw, m.m12.raw, m.m13.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m11.raw, got.m12.raw, got.m13.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m21.raw, m.m22.raw, m.m23.raw))) - r.bias();
    assert!(work.y == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m21.raw, got.m22.raw, got.m23.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m31.raw, m.m32.raw, m.m33.raw))) - r.bias();
    assert!(work.z == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m31.raw, got.m32.raw, got.m33.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m41.raw, m.m42.raw, m.m43.raw))) - r.bias();
    assert!(work.w == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m41.raw, got.m42.raw, got.m43.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat4x3(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat4x3(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection3_rows_of_matrix5x3() {
    let r = refl3();
    let m = mat5x3(
        [
            [-1079557932, 2865399355, -3222418526], [5046437435, -1022291703, 2941754327],
            [1139657634, 5122792407, -926847988], [-2767122167, 1235101349, 5237324865],
            [3358873200, -2652589709, 1368722550],
        ],
    );
    let mut got = m;
    let mut work = vec5((0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m11.raw, m.m12.raw, m.m13.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m11.raw, got.m12.raw, got.m13.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m21.raw, m.m22.raw, m.m23.raw))) - r.bias();
    assert!(work.y == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m21.raw, got.m22.raw, got.m23.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m31.raw, m.m32.raw, m.m33.raw))) - r.bias();
    assert!(work.z == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m31.raw, got.m32.raw, got.m33.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m41.raw, m.m42.raw, m.m43.raw))) - r.bias();
    assert!(work.w == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m41.raw, got.m42.raw, got.m43.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m51.raw, m.m52.raw, m.m53.raw))) - r.bias();
    assert!(work.a == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m51.raw, got.m52.raw, got.m53.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat5x3(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat5x3(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection3_rows_of_matrix6x3() {
    let r = refl3();
    let m = mat6x3(
        [
            [1992984123, -4094833758, -149876471], [-1913795678, 2050250352, -4018478786],
            [4212199689, -1837440706, 2145694067], [305419888, 4307643404, -1722908248],
            [-3601359913, 419952346, 4441264605], [2524635454, -3467738712, 572662290],
        ],
    );
    let mut got = m;
    let mut work = vec6((0, 0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m11.raw, m.m12.raw, m.m13.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m11.raw, got.m12.raw, got.m13.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m21.raw, m.m22.raw, m.m23.raw))) - r.bias();
    assert!(work.y == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m21.raw, got.m22.raw, got.m23.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m31.raw, m.m32.raw, m.m33.raw))) - r.bias();
    assert!(work.z == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m31.raw, got.m32.raw, got.m33.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m41.raw, m.m42.raw, m.m43.raw))) - r.bias();
    assert!(work.w == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m41.raw, got.m42.raw, got.m43.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m51.raw, m.m52.raw, m.m53.raw))) - r.bias();
    assert!(work.a == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m51.raw, got.m52.raw, got.m53.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector3Trait::dot(r.axis(), vec3((m.m61.raw, m.m62.raw, m.m63.raw))) - r.bias();
    assert!(work.b == d0);
    let d1 = Vector3Trait::dot(r.axis(), vec3((got.m61.raw, got.m62.raw, got.m63.raw))) - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat6x3(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat6x3(minus, -got) <= 3 && work4 == work);
}

// --- Reflection4

fn axis4() -> Unit<Vector4<Fixed>> {
    UnitTrait::new_normalize(vec4((4294967296, -4294967296, 12884901888, 4294967296)))
}

fn refl4() -> Reflection4<Fixed> {
    Reflection4Trait::new(axis4(), fx(0x12345678))
}

#[test]
fn test_reflection4_api() {
    let axis = Unit { value: vec4((0x100000000, 0, 0, 0)) };
    let r = Reflection4Trait::new(axis, int(3));
    assert!(r.axis() == axis.value && r.bias() == int(3));
    let through = Reflection4Trait::new_containing_point(
        axis, pt4((21474836480, 4294967296, 8589934592, 12884901888)),
    );
    assert!(through.bias() == int(5) && through.axis() == axis.value);
    // The plane x = 3: x -> 3 - (x - 3), the other coordinates unchanged.
    let mut v = vec4((30064771072, 4294967296, 4294967296, 4294967296));
    r.reflect(ref v);
    assert!(v == vec4((-4294967296, 4294967296, 4294967296, 4294967296)));
    let mut s = vec4((30064771072, 4294967296, 4294967296, 4294967296));
    r.reflect_with_sign(ref s, int(-1));
    assert!(s == vec4((4294967296, -4294967296, -4294967296, -4294967296)));
    let mut row = RowVector4 { x: int(7), y: int(1), z: int(1), w: int(1) };
    let mut work = Matrix1 { x: int(0) };
    r.reflect_rows(ref row, ref work);
    assert!(work.x == int(4));
    assert!(row == RowVector4 { x: int(-1), y: int(1), z: int(1), w: int(1) });
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_reflection4_reflect_overflow_panics() {
    let r = Reflection4Trait::new(Unit { value: vec4((0x100000000, 0, 0, 0)) }, int(-2000000000));
    let mut v = black_box(vec4((2000000000 * 0x100000000, 0, 0, 0)));
    r.reflect(ref v);
}

#[test]
fn test_reflection4_columns_of_vector4() {
    let r = refl4();
    let m = mat4x1([[550740074], [-3356039727], [2769955640], [-1136824161]]);
    let mut got = m;
    r.reflect(ref got);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.x.raw, m.y.raw, m.z.raw, m.w.raw))) - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.x.raw, got.y.raw, got.z.raw, got.w.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat4x1(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat4x1(minus, -got) <= 3);
}

#[test]
fn test_reflection4_columns_of_matrix4x2() {
    let r = refl4();
    let m = mat4x2(
        [
            [3623282129, -2464535752], [-283497672, 3680548358], [-4190277473, -207142700],
            [1935717894, -4094833758],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec4((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec4((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat4x2(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat4x2(minus, -got) <= 3);
}

#[test]
fn test_reflection4_columns_of_matrix4x3() {
    let r = refl4();
    let m = mat4x3(
        [
            [-3336950984, 608006303, 4552963590], [2789044383, -3279684755, 684361275],
            [-1117735418, 2865399355, -3184241040], [5008259949, -1022291703, 2979931813],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec4((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec4((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec4((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat4x3(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat4x3(minus, -got) <= 3);
}

#[test]
fn test_reflection4_columns_of_matrix4() {
    let r = refl4();
    let m = mat4x4(
        [
            [-264408929, 3680548358, -2407269523, 1537687764],
            [-4171188730, -207142700, 3756903330, -2311825808],
            [1954806637, -4094833758, -111698985, 3871435788],
            [-1951973164, 2050250352, -3980301300, 21922216],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec4((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec4((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec4((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec4((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    assert!(got.m44 == col4.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m14.raw, got.m24.raw, got.m34.raw, got.m44.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat4x4(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat4x4(minus, -got) <= 3);
}

#[test]
fn test_reflection4_columns_of_matrix4x5() {
    let r = refl4();
    let m = mat4x5(
        [
            [2808133126, -3279684755, 665272532, 4610229819, -1477588062],
            [-1098646675, 2865399355, -3203329783, 760716247, 4724762277],
            [5027348692, -1022291703, 2960843070, -3088797325, 894337448],
            [1120568891, 5122792407, -907759245, 3094464271, -2936087381],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec4((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec4((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec4((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec4((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    assert!(got.m44 == col4.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m14.raw, got.m24.raw, got.m34.raw, got.m44.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec4((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw));
    r.reflect(ref col5);
    assert!(got.m15 == col5.x);
    assert!(got.m25 == col5.y);
    assert!(got.m35 == col5.z);
    assert!(got.m45 == col5.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m15.raw, got.m25.raw, got.m35.raw, got.m45.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat4x5(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat4x5(minus, -got) <= 3);
}

#[test]
fn test_reflection4_columns_of_matrix4x6() {
    let r = refl4();
    let m = mat4x6(
        [
            [-4152099987, -207142700, 3737814587, -2350003294, 1594953993, -4492863888],
            [1973895380, -4094833758, -130787728, 3833258302, -2235470836, 1728575194],
            [-1932884421, 2050250352, -3999390043, -16255270, 3966879503, -2082760892],
            [4193110946, -1837440706, 2164782810, -3865768842, 136454674, 4138678190],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec4((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec4((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec4((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec4((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    assert!(got.m44 == col4.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m14.raw, got.m24.raw, got.m34.raw, got.m44.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec4((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw));
    r.reflect(ref col5);
    assert!(got.m15 == col5.x);
    assert!(got.m25 == col5.y);
    assert!(got.m35 == col5.z);
    assert!(got.m45 == col5.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m15.raw, got.m25.raw, got.m35.raw, got.m45.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col6 = vec4((m.m16.raw, m.m26.raw, m.m36.raw, m.m46.raw));
    r.reflect(ref col6);
    assert!(got.m16 == col6.x);
    assert!(got.m26 == col6.y);
    assert!(got.m36 == col6.z);
    assert!(got.m46 == col6.w);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m16.raw, m.m26.raw, m.m36.raw, m.m46.raw)))
        - r.bias();
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m16.raw, got.m26.raw, got.m36.raw, got.m46.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat4x6(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat4x6(minus, -got) <= 3);
}

#[test]
fn test_reflection4_rows_of_rowvector4() {
    let r = refl4();
    let m = mat1x4([[2808133126, -3279684755, 665272532, 4610229819]]);
    let mut got = m;
    let mut work = vec1((0,));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.x.raw, m.y.raw, m.z.raw, m.w.raw))) - r.bias();
    assert!(work.x == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.x.raw, got.y.raw, got.z.raw, got.w.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat1x4(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat1x4(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection4_rows_of_matrix2x4() {
    let r = refl4();
    let m = mat2x4(
        [
            [-4152099987, -207142700, 3737814587, -2350003294],
            [1973895380, -4094833758, -130787728, 3833258302],
        ],
    );
    let mut got = m;
    let mut work = vec2((0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw)))
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw)))
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat2x4(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat2x4(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection4_rows_of_matrix3x4() {
    let r = refl4();
    let m = mat3x4(
        [
            [-1079557932, 2865399355, -3222418526, 722538761],
            [5046437435, -1022291703, 2941754327, -3126974811],
            [1139657634, 5122792407, -926847988, 3056286785],
        ],
    );
    let mut got = m;
    let mut work = vec3((0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw)))
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw)))
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw)))
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat3x4(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat3x4(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection4_rows_of_matrix4() {
    let r = refl4();
    let m = mat4x4(
        [
            [1992984123, -4094833758, -149876471, 3795080816],
            [-1913795678, 2050250352, -4018478786, -54432756],
            [4212199689, -1837440706, 2145694067, -3903946328],
            [305419888, 4307643404, -1722908248, 2279315268],
        ],
    );
    let mut got = m;
    let mut work = vec4((0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw)))
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw)))
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw)))
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m41.raw, m.m42.raw, m.m43.raw, m.m44.raw)))
        - r.bias();
    assert!(work.w == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m41.raw, got.m42.raw, got.m43.raw, got.m44.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat4x4(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat4x4(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection4_rows_of_matrix5x4() {
    let r = refl4();
    let m = mat5x4(
        [
            [5065526178, -1022291703, 2922665584, -3165152297],
            [1158746377, 5122792407, -945936731, 3018109299],
            [-2748033424, 1235101349, 5218236122, -831404273],
            [3377961943, -2652589709, 1349633807, 5351857323],
            [-528817858, 3492494401, -2518968508, 1502343751],
        ],
    );
    let mut got = m;
    let mut work = vec5((0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw)))
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw)))
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw)))
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m41.raw, m.m42.raw, m.m43.raw, m.m44.raw)))
        - r.bias();
    assert!(work.w == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m41.raw, got.m42.raw, got.m43.raw, got.m44.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m51.raw, m.m52.raw, m.m53.raw, m.m54.raw)))
        - r.bias();
    assert!(work.a == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m51.raw, got.m52.raw, got.m53.raw, got.m54.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat5x4(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat5x4(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection4_rows_of_matrix6x4() {
    let r = refl4();
    let m = mat6x4(
        [
            [-1894706935, 2050250352, -4037567529, -92610242],
            [4231288432, -1837440706, 2126605324, -3942123814],
            [324508631, 4307643404, -1741996991, 2241137782],
            [-3582271170, 419952346, 4422175862, -1608375790],
            [2543724197, -3467738712, 553573547, 4574885806],
            [-1363055604, 2677345398, -3315028768, 725372234],
        ],
    );
    let mut got = m;
    let mut work = vec6((0, 0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw)))
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw)))
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw)))
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m41.raw, m.m42.raw, m.m43.raw, m.m44.raw)))
        - r.bias();
    assert!(work.w == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m41.raw, got.m42.raw, got.m43.raw, got.m44.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m51.raw, m.m52.raw, m.m53.raw, m.m54.raw)))
        - r.bias();
    assert!(work.a == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m51.raw, got.m52.raw, got.m53.raw, got.m54.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector4Trait::dot(r.axis(), vec4((m.m61.raw, m.m62.raw, m.m63.raw, m.m64.raw)))
        - r.bias();
    assert!(work.b == d0);
    let d1 = Vector4Trait::dot(r.axis(), vec4((got.m61.raw, got.m62.raw, got.m63.raw, got.m64.raw)))
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat6x4(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat6x4(minus, -got) <= 3 && work4 == work);
}

// --- Reflection5

fn axis5() -> Unit<Vector5<Fixed>> {
    UnitTrait::new_normalize(vec5((4294967296, -4294967296, 12884901888, 4294967296, 21474836480)))
}

fn refl5() -> Reflection5<Fixed> {
    Reflection5Trait::new(axis5(), fx(0x12345678))
}

#[test]
fn test_reflection5_api() {
    let axis = Unit { value: vec5((0x100000000, 0, 0, 0, 0)) };
    let r = Reflection5Trait::new(axis, int(3));
    assert!(r.axis() == axis.value && r.bias() == int(3));
    let through = Reflection5Trait::new_containing_point(
        axis, pt5((21474836480, 4294967296, 8589934592, 12884901888, 17179869184)),
    );
    assert!(through.bias() == int(5) && through.axis() == axis.value);
    // The plane x = 3: x -> 3 - (x - 3), the other coordinates unchanged.
    let mut v = vec5((30064771072, 4294967296, 4294967296, 4294967296, 4294967296));
    r.reflect(ref v);
    assert!(v == vec5((-4294967296, 4294967296, 4294967296, 4294967296, 4294967296)));
    let mut s = vec5((30064771072, 4294967296, 4294967296, 4294967296, 4294967296));
    r.reflect_with_sign(ref s, int(-1));
    assert!(s == vec5((4294967296, -4294967296, -4294967296, -4294967296, -4294967296)));
    let mut row = RowVector5 { x: int(7), y: int(1), z: int(1), w: int(1), a: int(1) };
    let mut work = Matrix1 { x: int(0) };
    r.reflect_rows(ref row, ref work);
    assert!(work.x == int(4));
    assert!(row == RowVector5 { x: int(-1), y: int(1), z: int(1), w: int(1), a: int(1) });
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_reflection5_reflect_overflow_panics() {
    let r = Reflection5Trait::new(
        Unit { value: vec5((0x100000000, 0, 0, 0, 0)) }, int(-2000000000),
    );
    let mut v = black_box(vec5((2000000000 * 0x100000000, 0, 0, 0, 0)));
    r.reflect(ref v);
}

#[test]
fn test_reflection5_columns_of_vector5() {
    let r = refl5();
    let m = mat5x1([[3623282129], [-283497672], [-4190277473], [1935717894], [-1971061907]]);
    let mut got = m;
    r.reflect(ref got);
    let d0 = Vector5Trait::dot(r.axis(), vec5((m.x.raw, m.y.raw, m.z.raw, m.w.raw, m.a.raw)))
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.x.raw, got.y.raw, got.z.raw, got.w.raw, got.a.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat5x1(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat5x1(minus, -got) <= 3);
}

#[test]
fn test_reflection5_columns_of_matrix5x2() {
    let r = refl5();
    let m = mat5x2(
        [
            [-3336950984, 608006303], [2789044383, -3279684755], [-1117735418, 2865399355],
            [5008259949, -1022291703], [1101480148, 5122792407],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec5((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    assert!(got.m51 == col1.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw, got.m51.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec5((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    assert!(got.m52 == col2.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw, got.m52.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat5x2(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat5x2(minus, -got) <= 3);
}

#[test]
fn test_reflection5_columns_of_matrix5x3() {
    let r = refl5();
    let m = mat5x3(
        [
            [-264408929, 3680548358, -2407269523], [-4171188730, -207142700, 3756903330],
            [1954806637, -4094833758, -111698985], [-1951973164, 2050250352, -3980301300],
            [4174022203, -1837440706, 2183871553],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec5((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    assert!(got.m51 == col1.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw, got.m51.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec5((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    assert!(got.m52 == col2.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw, got.m52.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec5((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    assert!(got.m53 == col3.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw, got.m53.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat5x3(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat5x3(minus, -got) <= 3);
}

#[test]
fn test_reflection5_columns_of_matrix5x4() {
    let r = refl5();
    let m = mat5x4(
        [
            [2808133126, -3279684755, 665272532, 4610229819],
            [-1098646675, 2865399355, -3203329783, 760716247],
            [5027348692, -1022291703, 2960843070, -3088797325],
            [1120568891, 5122792407, -907759245, 3094464271],
            [-2786210910, 1235101349, 5256413608, -755049301],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec5((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    assert!(got.m51 == col1.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw, got.m51.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec5((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    assert!(got.m52 == col2.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw, got.m52.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec5((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    assert!(got.m53 == col3.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw, got.m53.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec5((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    assert!(got.m44 == col4.w);
    assert!(got.m54 == col4.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m14.raw, got.m24.raw, got.m34.raw, got.m44.raw, got.m54.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat5x4(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat5x4(minus, -got) <= 3);
}

#[test]
fn test_reflection5_columns_of_matrix5() {
    let r = refl5();
    let m = mat5x5(
        [
            [-4152099987, -207142700, 3737814587, -2350003294, 1594953993],
            [1973895380, -4094833758, -130787728, 3833258302, -2235470836],
            [-1932884421, 2050250352, -3999390043, -16255270, 3966879503],
            [4193110946, -1837440706, 2164782810, -3865768842, 136454674],
            [286331145, 4307643404, -1703819505, 2317492754, -3693970155],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec5((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    assert!(got.m51 == col1.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw, got.m51.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec5((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    assert!(got.m52 == col2.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw, got.m52.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec5((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    assert!(got.m53 == col3.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw, got.m53.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec5((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    assert!(got.m44 == col4.w);
    assert!(got.m54 == col4.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m14.raw, got.m24.raw, got.m34.raw, got.m44.raw, got.m54.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec5((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw, m.m55.raw));
    r.reflect(ref col5);
    assert!(got.m15 == col5.x);
    assert!(got.m25 == col5.y);
    assert!(got.m35 == col5.z);
    assert!(got.m45 == col5.w);
    assert!(got.m55 == col5.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw, m.m55.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m15.raw, got.m25.raw, got.m35.raw, got.m45.raw, got.m55.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat5x5(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat5x5(minus, -got) <= 3);
}

#[test]
fn test_reflection5_columns_of_matrix5x6() {
    let r = refl5();
    let m = mat5x6(
        [
            [-1079557932, 2865399355, -3222418526, 722538761, 4667496048, -1420321833],
            [5046437435, -1022291703, 2941754327, -3126974811, 837071219, 4801117249],
            [1139657634, 5122792407, -926847988, 3056286785, -2993353610, 989781163],
            [-2767122167, 1235101349, 5237324865, -793226787, 3208996729, -2821554923],
            [3358873200, -2652589709, 1368722550, 5390034809, -621428100, 3399884159],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec5((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    assert!(got.m51 == col1.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw, got.m51.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec5((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    assert!(got.m52 == col2.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw, got.m52.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec5((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    assert!(got.m53 == col3.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw, got.m53.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec5((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    assert!(got.m44 == col4.w);
    assert!(got.m54 == col4.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m14.raw, got.m24.raw, got.m34.raw, got.m44.raw, got.m54.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec5((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw, m.m55.raw));
    r.reflect(ref col5);
    assert!(got.m15 == col5.x);
    assert!(got.m25 == col5.y);
    assert!(got.m35 == col5.z);
    assert!(got.m45 == col5.w);
    assert!(got.m55 == col5.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw, m.m55.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m15.raw, got.m25.raw, got.m35.raw, got.m45.raw, got.m55.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col6 = vec5((m.m16.raw, m.m26.raw, m.m36.raw, m.m46.raw, m.m56.raw));
    r.reflect(ref col6);
    assert!(got.m16 == col6.x);
    assert!(got.m26 == col6.y);
    assert!(got.m36 == col6.z);
    assert!(got.m46 == col6.w);
    assert!(got.m56 == col6.a);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m16.raw, m.m26.raw, m.m36.raw, m.m46.raw, m.m56.raw)),
    )
        - r.bias();
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m16.raw, got.m26.raw, got.m36.raw, got.m46.raw, got.m56.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat5x6(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat5x6(minus, -got) <= 3);
}

#[test]
fn test_reflection5_rows_of_rowvector5() {
    let r = refl5();
    let m = mat1x5([[-1079557932, 2865399355, -3222418526, 722538761, 4667496048]]);
    let mut got = m;
    let mut work = vec1((0,));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector5Trait::dot(r.axis(), vec5((m.x.raw, m.y.raw, m.z.raw, m.w.raw, m.a.raw)))
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.x.raw, got.y.raw, got.z.raw, got.w.raw, got.a.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat1x5(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat1x5(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection5_rows_of_matrix2x5() {
    let r = refl5();
    let m = mat2x5(
        [
            [1992984123, -4094833758, -149876471, 3795080816, -2292737065],
            [-1913795678, 2050250352, -4018478786, -54432756, 3909613274],
        ],
    );
    let mut got = m;
    let mut work = vec2((0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw, m.m15.raw)),
    )
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw, got.m15.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw, m.m25.raw)),
    )
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw, got.m25.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat2x5(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat2x5(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection5_rows_of_matrix3x5() {
    let r = refl5();
    let m = mat3x5(
        [
            [5065526178, -1022291703, 2922665584, -3165152297, 779804990],
            [1158746377, 5122792407, -945936731, 3018109299, -3050619839],
            [-2748033424, 1235101349, 5218236122, -831404273, 3151730500],
        ],
    );
    let mut got = m;
    let mut work = vec3((0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw, m.m15.raw)),
    )
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw, got.m15.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw, m.m25.raw)),
    )
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw, got.m25.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw, m.m35.raw)),
    )
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw, got.m35.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat3x5(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat3x5(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection5_rows_of_matrix4x5() {
    let r = refl5();
    let m = mat4x5(
        [
            [-1894706935, 2050250352, -4037567529, -92610242, 3852347045],
            [4231288432, -1837440706, 2126605324, -3942123814, 21922216],
            [324508631, 4307643404, -1741996991, 2241137782, -3808502613],
            [-3582271170, 419952346, 4422175862, -1608375790, 2393847726],
        ],
    );
    let mut got = m;
    let mut work = vec4((0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw, m.m15.raw)),
    )
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw, got.m15.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw, m.m25.raw)),
    )
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw, got.m25.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw, m.m35.raw)),
    )
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw, got.m35.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m41.raw, m.m42.raw, m.m43.raw, m.m44.raw, m.m45.raw)),
    )
        - r.bias();
    assert!(work.w == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m41.raw, got.m42.raw, got.m43.raw, got.m44.raw, got.m45.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat4x5(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat4x5(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection5_rows_of_matrix5() {
    let r = refl5();
    let m = mat5x5(
        [
            [1177835120, 5122792407, -965025474, 2979931813, -3107886068],
            [-2728944681, 1235101349, 5199147379, -869581759, 3094464271],
            [3397050686, -2652589709, 1330545064, 5313679837, -735960558],
            [-509729115, 3492494401, -2538057251, 1464166265, 5466389781],
            [-4416508916, -395196657, 3626115602, -2385347307, 1635964952],
        ],
    );
    let mut got = m;
    let mut work = vec5((0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw, m.m15.raw)),
    )
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw, got.m15.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw, m.m25.raw)),
    )
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw, got.m25.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw, m.m35.raw)),
    )
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw, got.m35.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m41.raw, m.m42.raw, m.m43.raw, m.m44.raw, m.m45.raw)),
    )
        - r.bias();
    assert!(work.w == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m41.raw, got.m42.raw, got.m43.raw, got.m44.raw, got.m45.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m51.raw, m.m52.raw, m.m53.raw, m.m54.raw, m.m55.raw)),
    )
        - r.bias();
    assert!(work.a == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m51.raw, got.m52.raw, got.m53.raw, got.m54.raw, got.m55.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat5x5(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat5x5(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection5_rows_of_matrix6x5() {
    let r = refl5();
    let m = mat6x5(
        [
            [4250377175, -1837440706, 2107516581, -3980301300, -35344013],
            [343597374, 4307643404, -1761085734, 2202960296, -3865768842],
            [-3563182427, 419952346, 4403087119, -1646553276, 2336581497],
            [2562812940, -3467738712, 534484804, 4536708320, -1493843332],
            [-1343966861, 2677345398, -3334117511, 687194748, 4708507007],
            [4782028506, -1210345660, 2830055342, -3162318824, 878082178],
        ],
    );
    let mut got = m;
    let mut work = vec6((0, 0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw, m.m15.raw)),
    )
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw, got.m15.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw, m.m25.raw)),
    )
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw, got.m25.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw, m.m35.raw)),
    )
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw, got.m35.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m41.raw, m.m42.raw, m.m43.raw, m.m44.raw, m.m45.raw)),
    )
        - r.bias();
    assert!(work.w == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m41.raw, got.m42.raw, got.m43.raw, got.m44.raw, got.m45.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m51.raw, m.m52.raw, m.m53.raw, m.m54.raw, m.m55.raw)),
    )
        - r.bias();
    assert!(work.a == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m51.raw, got.m52.raw, got.m53.raw, got.m54.raw, got.m55.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector5Trait::dot(
        r.axis(), vec5((m.m61.raw, m.m62.raw, m.m63.raw, m.m64.raw, m.m65.raw)),
    )
        - r.bias();
    assert!(work.b == d0);
    let d1 = Vector5Trait::dot(
        r.axis(), vec5((got.m61.raw, got.m62.raw, got.m63.raw, got.m64.raw, got.m65.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat6x5(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat6x5(minus, -got) <= 3 && work4 == work);
}

// --- Reflection6

fn axis6() -> Unit<Vector6<Fixed>> {
    UnitTrait::new_normalize(
        vec6((4294967296, -4294967296, 12884901888, 4294967296, 21474836480, 12884901888)),
    )
}

fn refl6() -> Reflection6<Fixed> {
    Reflection6Trait::new(axis6(), fx(0x12345678))
}

#[test]
fn test_reflection6_api() {
    let axis = Unit { value: vec6((0x100000000, 0, 0, 0, 0, 0)) };
    let r = Reflection6Trait::new(axis, int(3));
    assert!(r.axis() == axis.value && r.bias() == int(3));
    let through = Reflection6Trait::new_containing_point(
        axis, pt6((21474836480, 4294967296, 8589934592, 12884901888, 17179869184, 21474836480)),
    );
    assert!(through.bias() == int(5) && through.axis() == axis.value);
    // The plane x = 3: x -> 3 - (x - 3), the other coordinates unchanged.
    let mut v = vec6((30064771072, 4294967296, 4294967296, 4294967296, 4294967296, 4294967296));
    r.reflect(ref v);
    assert!(v == vec6((-4294967296, 4294967296, 4294967296, 4294967296, 4294967296, 4294967296)));
    let mut s = vec6((30064771072, 4294967296, 4294967296, 4294967296, 4294967296, 4294967296));
    r.reflect_with_sign(ref s, int(-1));
    assert!(
        s == vec6((4294967296, -4294967296, -4294967296, -4294967296, -4294967296, -4294967296)),
    );
    let mut row = RowVector6 { x: int(7), y: int(1), z: int(1), w: int(1), a: int(1), b: int(1) };
    let mut work = Matrix1 { x: int(0) };
    r.reflect_rows(ref row, ref work);
    assert!(work.x == int(4));
    assert!(
        row == RowVector6 { x: int(-1), y: int(1), z: int(1), w: int(1), a: int(1), b: int(1) },
    );
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_reflection6_reflect_overflow_panics() {
    let r = Reflection6Trait::new(
        Unit { value: vec6((0x100000000, 0, 0, 0, 0, 0)) }, int(-2000000000),
    );
    let mut v = black_box(vec6((2000000000 * 0x100000000, 0, 0, 0, 0, 0)));
    r.reflect(ref v);
}

#[test]
fn test_reflection6_columns_of_vector6() {
    let r = refl6();
    let m = mat6x1(
        [[-3336950984], [2789044383], [-1117735418], [5008259949], [1101480148], [-2805299653]],
    );
    let mut got = m;
    r.reflect(ref got);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.x.raw, m.y.raw, m.z.raw, m.w.raw, m.a.raw, m.b.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(), vec6((got.x.raw, got.y.raw, got.z.raw, got.w.raw, got.a.raw, got.b.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat6x1(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat6x1(minus, -got) <= 3);
}

#[test]
fn test_reflection6_columns_of_matrix6x2() {
    let r = refl6();
    let m = mat6x2(
        [
            [-264408929, 3680548358], [-4171188730, -207142700], [1954806637, -4094833758],
            [-1951973164, 2050250352], [4174022203, -1837440706], [267242402, 4307643404],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec6((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw, m.m61.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    assert!(got.m51 == col1.a);
    assert!(got.m61 == col1.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw, m.m61.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw, got.m51.raw, got.m61.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec6((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw, m.m62.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    assert!(got.m52 == col2.a);
    assert!(got.m62 == col2.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw, m.m62.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw, got.m52.raw, got.m62.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat6x2(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat6x2(minus, -got) <= 3);
}

#[test]
fn test_reflection6_columns_of_matrix6x3() {
    let r = refl6();
    let m = mat6x3(
        [
            [2808133126, -3279684755, 665272532], [-1098646675, 2865399355, -3203329783],
            [5027348692, -1022291703, 2960843070], [1120568891, 5122792407, -907759245],
            [-2786210910, 1235101349, 5256413608], [3339784457, -2652589709, 1387811293],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec6((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw, m.m61.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    assert!(got.m51 == col1.a);
    assert!(got.m61 == col1.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw, m.m61.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw, got.m51.raw, got.m61.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec6((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw, m.m62.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    assert!(got.m52 == col2.a);
    assert!(got.m62 == col2.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw, m.m62.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw, got.m52.raw, got.m62.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec6((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw, m.m63.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    assert!(got.m53 == col3.a);
    assert!(got.m63 == col3.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw, m.m63.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw, got.m53.raw, got.m63.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat6x3(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat6x3(minus, -got) <= 3);
}

#[test]
fn test_reflection6_columns_of_matrix6x4() {
    let r = refl6();
    let m = mat6x4(
        [
            [-4152099987, -207142700, 3737814587, -2350003294],
            [1973895380, -4094833758, -130787728, 3833258302],
            [-1932884421, 2050250352, -3999390043, -16255270],
            [4193110946, -1837440706, 2164782810, -3865768842],
            [286331145, 4307643404, -1703819505, 2317492754],
            [-3620448656, 419952346, 4460353348, -1532020818],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec6((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw, m.m61.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    assert!(got.m51 == col1.a);
    assert!(got.m61 == col1.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw, m.m61.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw, got.m51.raw, got.m61.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec6((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw, m.m62.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    assert!(got.m52 == col2.a);
    assert!(got.m62 == col2.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw, m.m62.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw, got.m52.raw, got.m62.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec6((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw, m.m63.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    assert!(got.m53 == col3.a);
    assert!(got.m63 == col3.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw, m.m63.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw, got.m53.raw, got.m63.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec6((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw, m.m64.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    assert!(got.m44 == col4.w);
    assert!(got.m54 == col4.a);
    assert!(got.m64 == col4.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw, m.m64.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m14.raw, got.m24.raw, got.m34.raw, got.m44.raw, got.m54.raw, got.m64.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat6x4(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat6x4(minus, -got) <= 3);
}

#[test]
fn test_reflection6_columns_of_matrix6x5() {
    let r = refl6();
    let m = mat6x5(
        [
            [-1079557932, 2865399355, -3222418526, 722538761, 4667496048],
            [5046437435, -1022291703, 2941754327, -3126974811, 837071219],
            [1139657634, 5122792407, -926847988, 3056286785, -2993353610],
            [-2767122167, 1235101349, 5237324865, -793226787, 3208996729],
            [3358873200, -2652589709, 1368722550, 5390034809, -621428100],
            [-547906601, 3492494401, -2499879765, 1540521237, 5580922239],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec6((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw, m.m61.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    assert!(got.m51 == col1.a);
    assert!(got.m61 == col1.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw, m.m61.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw, got.m51.raw, got.m61.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec6((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw, m.m62.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    assert!(got.m52 == col2.a);
    assert!(got.m62 == col2.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw, m.m62.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw, got.m52.raw, got.m62.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec6((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw, m.m63.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    assert!(got.m53 == col3.a);
    assert!(got.m63 == col3.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw, m.m63.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw, got.m53.raw, got.m63.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec6((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw, m.m64.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    assert!(got.m44 == col4.w);
    assert!(got.m54 == col4.a);
    assert!(got.m64 == col4.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw, m.m64.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m14.raw, got.m24.raw, got.m34.raw, got.m44.raw, got.m54.raw, got.m64.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec6((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw, m.m55.raw, m.m65.raw));
    r.reflect(ref col5);
    assert!(got.m15 == col5.x);
    assert!(got.m25 == col5.y);
    assert!(got.m35 == col5.z);
    assert!(got.m45 == col5.w);
    assert!(got.m55 == col5.a);
    assert!(got.m65 == col5.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw, m.m55.raw, m.m65.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m15.raw, got.m25.raw, got.m35.raw, got.m45.raw, got.m55.raw, got.m65.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat6x5(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat6x5(minus, -got) <= 3);
}

#[test]
fn test_reflection6_columns_of_matrix6() {
    let r = refl6();
    let m = mat6x6(
        [
            [1992984123, -4094833758, -149876471, 3795080816, -2292737065, 1652220222],
            [-1913795678, 2050250352, -4018478786, -54432756, 3909613274, -2159115864],
            [4212199689, -1837440706, 2145694067, -3903946328, 79188445, 4062323218],
            [305419888, 4307643404, -1722908248, 2279315268, -3751236384, 250987132],
            [-3601359913, 419952346, 4441264605, -1570198304, 2451113955, -3560348954],
            [2524635454, -3467738712, 572662290, 4613063292, -1379310874, 2661090128],
        ],
    );
    let mut got = m;
    r.reflect(ref got);
    let mut col1 = vec6((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw, m.m61.raw));
    r.reflect(ref col1);
    assert!(got.m11 == col1.x);
    assert!(got.m21 == col1.y);
    assert!(got.m31 == col1.z);
    assert!(got.m41 == col1.w);
    assert!(got.m51 == col1.a);
    assert!(got.m61 == col1.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m11.raw, m.m21.raw, m.m31.raw, m.m41.raw, m.m51.raw, m.m61.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m11.raw, got.m21.raw, got.m31.raw, got.m41.raw, got.m51.raw, got.m61.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col2 = vec6((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw, m.m62.raw));
    r.reflect(ref col2);
    assert!(got.m12 == col2.x);
    assert!(got.m22 == col2.y);
    assert!(got.m32 == col2.z);
    assert!(got.m42 == col2.w);
    assert!(got.m52 == col2.a);
    assert!(got.m62 == col2.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m12.raw, m.m22.raw, m.m32.raw, m.m42.raw, m.m52.raw, m.m62.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m12.raw, got.m22.raw, got.m32.raw, got.m42.raw, got.m52.raw, got.m62.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col3 = vec6((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw, m.m63.raw));
    r.reflect(ref col3);
    assert!(got.m13 == col3.x);
    assert!(got.m23 == col3.y);
    assert!(got.m33 == col3.z);
    assert!(got.m43 == col3.w);
    assert!(got.m53 == col3.a);
    assert!(got.m63 == col3.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m13.raw, m.m23.raw, m.m33.raw, m.m43.raw, m.m53.raw, m.m63.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m13.raw, got.m23.raw, got.m33.raw, got.m43.raw, got.m53.raw, got.m63.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col4 = vec6((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw, m.m64.raw));
    r.reflect(ref col4);
    assert!(got.m14 == col4.x);
    assert!(got.m24 == col4.y);
    assert!(got.m34 == col4.z);
    assert!(got.m44 == col4.w);
    assert!(got.m54 == col4.a);
    assert!(got.m64 == col4.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m14.raw, m.m24.raw, m.m34.raw, m.m44.raw, m.m54.raw, m.m64.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m14.raw, got.m24.raw, got.m34.raw, got.m44.raw, got.m54.raw, got.m64.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col5 = vec6((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw, m.m55.raw, m.m65.raw));
    r.reflect(ref col5);
    assert!(got.m15 == col5.x);
    assert!(got.m25 == col5.y);
    assert!(got.m35 == col5.z);
    assert!(got.m45 == col5.w);
    assert!(got.m55 == col5.a);
    assert!(got.m65 == col5.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m15.raw, m.m25.raw, m.m35.raw, m.m45.raw, m.m55.raw, m.m65.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m15.raw, got.m25.raw, got.m35.raw, got.m45.raw, got.m55.raw, got.m65.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let mut col6 = vec6((m.m16.raw, m.m26.raw, m.m36.raw, m.m46.raw, m.m56.raw, m.m66.raw));
    r.reflect(ref col6);
    assert!(got.m16 == col6.x);
    assert!(got.m26 == col6.y);
    assert!(got.m36 == col6.z);
    assert!(got.m46 == col6.w);
    assert!(got.m56 == col6.a);
    assert!(got.m66 == col6.b);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m16.raw, m.m26.raw, m.m36.raw, m.m46.raw, m.m56.raw, m.m66.raw)),
    )
        - r.bias();
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m16.raw, got.m26.raw, got.m36.raw, got.m46.raw, got.m56.raw, got.m66.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    r.reflect(ref back);
    assert!(err_mat6x6(back, m) <= 64);
    // sign = 1 is `reflect` bit for bit; sign = -1 is its opposite
    let mut plus = m;
    r.reflect_with_sign(ref plus, int(1));
    assert!(plus == got);
    let mut minus = m;
    r.reflect_with_sign(ref minus, int(-1));
    assert!(err_mat6x6(minus, -got) <= 3);
}

#[test]
fn test_reflection6_rows_of_rowvector6() {
    let r = refl6();
    let m = mat1x6([[5065526178, -1022291703, 2922665584, -3165152297, 779804990, 4724762277]]);
    let mut got = m;
    let mut work = vec1((0,));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.x.raw, m.y.raw, m.z.raw, m.w.raw, m.a.raw, m.b.raw)),
    )
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector6Trait::dot(
        r.axis(), vec6((got.x.raw, got.y.raw, got.z.raw, got.w.raw, got.a.raw, got.b.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat1x6(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat1x6(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection6_rows_of_matrix2x6() {
    let r = refl6();
    let m = mat2x6(
        [
            [-1894706935, 2050250352, -4037567529, -92610242, 3852347045, -2235470836],
            [4231288432, -1837440706, 2126605324, -3942123814, 21922216, 3985968246],
        ],
    );
    let mut got = m;
    let mut work = vec2((0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw, m.m15.raw, m.m16.raw)),
    )
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw, got.m15.raw, got.m16.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw, m.m25.raw, m.m26.raw)),
    )
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw, got.m25.raw, got.m26.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat2x6(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat2x6(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection6_rows_of_matrix3x6() {
    let r = refl6();
    let m = mat3x6(
        [
            [1177835120, 5122792407, -965025474, 2979931813, -3107886068, 837071219],
            [-2728944681, 1235101349, 5199147379, -869581759, 3094464271, -2974264867],
            [3397050686, -2652589709, 1330545064, 5313679837, -735960558, 3247174215],
        ],
    );
    let mut got = m;
    let mut work = vec3((0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw, m.m15.raw, m.m16.raw)),
    )
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw, got.m15.raw, got.m16.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw, m.m25.raw, m.m26.raw)),
    )
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw, got.m25.raw, got.m26.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw, m.m35.raw, m.m36.raw)),
    )
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw, got.m35.raw, got.m36.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat3x6(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat3x6(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection6_rows_of_matrix4x6() {
    let r = refl6();
    let m = mat4x6(
        [
            [4250377175, -1837440706, 2107516581, -3980301300, -35344013, 3909613274],
            [343597374, 4307643404, -1761085734, 2202960296, -3865768842, 98277188],
            [-3563182427, 419952346, 4403087119, -1646553276, 2336581497, -3713058898],
            [2562812940, -3467738712, 534484804, 4536708320, -1493843332, 2508380184],
        ],
    );
    let mut got = m;
    let mut work = vec4((0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw, m.m15.raw, m.m16.raw)),
    )
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw, got.m15.raw, got.m16.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw, m.m25.raw, m.m26.raw)),
    )
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw, got.m25.raw, got.m26.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw, m.m35.raw, m.m36.raw)),
    )
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw, got.m35.raw, got.m36.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m41.raw, m.m42.raw, m.m43.raw, m.m44.raw, m.m45.raw, m.m46.raw)),
    )
        - r.bias();
    assert!(work.w == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m41.raw, got.m42.raw, got.m43.raw, got.m44.raw, got.m45.raw, got.m46.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat4x6(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat4x6(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection6_rows_of_matrix5x6() {
    let r = refl6();
    let m = mat5x6(
        [
            [-2709855938, 1235101349, 5180058636, -907759245, 3037198042, -3050619839],
            [3416139429, -2652589709, 1311456321, 5275502351, -793226787, 3170819243],
            [-490640372, 3492494401, -2557145994, 1425988779, 5409123552, -640516843],
            [-4397420173, -395196657, 3607026859, -2423524793, 1578698723, 5580922239],
            [1728575194, -4282887715, -261575456, 3759736803, -2251726106, 1769586153],
        ],
    );
    let mut got = m;
    let mut work = vec5((0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw, m.m15.raw, m.m16.raw)),
    )
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw, got.m15.raw, got.m16.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw, m.m25.raw, m.m26.raw)),
    )
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw, got.m25.raw, got.m26.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw, m.m35.raw, m.m36.raw)),
    )
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw, got.m35.raw, got.m36.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m41.raw, m.m42.raw, m.m43.raw, m.m44.raw, m.m45.raw, m.m46.raw)),
    )
        - r.bias();
    assert!(work.w == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m41.raw, got.m42.raw, got.m43.raw, got.m44.raw, got.m45.raw, got.m46.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m51.raw, m.m52.raw, m.m53.raw, m.m54.raw, m.m55.raw, m.m56.raw)),
    )
        - r.bias();
    assert!(work.a == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m51.raw, got.m52.raw, got.m53.raw, got.m54.raw, got.m55.raw, got.m56.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat5x6(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat5x6(minus, -got) <= 3 && work4 == work);
}

#[test]
fn test_reflection6_rows_of_matrix6() {
    let r = refl6();
    let m = mat6x6(
        [
            [362686117, 4307643404, -1780174477, 2164782810, -3923035071, 21922216],
            [-3544093684, 419952346, 4383998376, -1684730762, 2279315268, -3789413870],
            [2581901683, -3467738712, 515396061, 4498530834, -1551109561, 2432025212],
            [-1324878118, 2677345398, -3353206254, 649017262, 4651240778, -1379310874],
            [4801117249, -1210345660, 2810966599, -3200496310, 820815949, 4842128208],
            [894337448, 4934738450, -1057635716, 2982765286, -3009608880, 1030792122],
        ],
    );
    let mut got = m;
    let mut work = vec6((0, 0, 0, 0, 0, 0));
    r.reflect_rows(ref got, ref work);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m11.raw, m.m12.raw, m.m13.raw, m.m14.raw, m.m15.raw, m.m16.raw)),
    )
        - r.bias();
    assert!(work.x == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m11.raw, got.m12.raw, got.m13.raw, got.m14.raw, got.m15.raw, got.m16.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m21.raw, m.m22.raw, m.m23.raw, m.m24.raw, m.m25.raw, m.m26.raw)),
    )
        - r.bias();
    assert!(work.y == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m21.raw, got.m22.raw, got.m23.raw, got.m24.raw, got.m25.raw, got.m26.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m31.raw, m.m32.raw, m.m33.raw, m.m34.raw, m.m35.raw, m.m36.raw)),
    )
        - r.bias();
    assert!(work.z == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m31.raw, got.m32.raw, got.m33.raw, got.m34.raw, got.m35.raw, got.m36.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m41.raw, m.m42.raw, m.m43.raw, m.m44.raw, m.m45.raw, m.m46.raw)),
    )
        - r.bias();
    assert!(work.w == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m41.raw, got.m42.raw, got.m43.raw, got.m44.raw, got.m45.raw, got.m46.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m51.raw, m.m52.raw, m.m53.raw, m.m54.raw, m.m55.raw, m.m56.raw)),
    )
        - r.bias();
    assert!(work.a == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m51.raw, got.m52.raw, got.m53.raw, got.m54.raw, got.m55.raw, got.m56.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    let d0 = Vector6Trait::dot(
        r.axis(), vec6((m.m61.raw, m.m62.raw, m.m63.raw, m.m64.raw, m.m65.raw, m.m66.raw)),
    )
        - r.bias();
    assert!(work.b == d0);
    let d1 = Vector6Trait::dot(
        r.axis(),
        vec6((got.m61.raw, got.m62.raw, got.m63.raw, got.m64.raw, got.m65.raw, got.m66.raw)),
    )
        - r.bias();
    assert!(ulp_diff(d1, -d0) <= 4 + abs_raw(d0) / 0x20000000);
    // involution
    let mut back = got;
    let mut work2 = work;
    r.reflect_rows(ref back, ref work2);
    assert!(err_mat6x6(back, m) <= 64);
    // sign = 1 is `reflect_rows` bit for bit
    let mut plus = m;
    let mut work3 = work;
    r.reflect_rows_with_sign(ref plus, ref work3, int(1));
    assert!(plus == got && work3 == work);
    let mut minus = m;
    let mut work4 = work;
    r.reflect_rows_with_sign(ref minus, ref work4, int(-1));
    assert!(err_mat6x6(minus, -got) <= 3 && work4 == work);
}

// --- oracle vectors

#[test]
fn test_reflection1_new_containing_point_oracle() {
    let mut cases = oracle::reflection1_new_containing_point_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, pt, e, tol) = *case;
        let got = Reflection1Trait::new_containing_point(Unit { value: vec1(axis) }, pt1(pt));
        worst =
            core::cmp::max(
                worst,
                report("reflection1_new_containing_point", n, ulp_diff(got.bias(), fx(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection2_new_containing_point_oracle() {
    let mut cases = oracle::reflection2_new_containing_point_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, pt, e, tol) = *case;
        let got = Reflection2Trait::new_containing_point(Unit { value: vec2(axis) }, pt2(pt));
        worst =
            core::cmp::max(
                worst,
                report("reflection2_new_containing_point", n, ulp_diff(got.bias(), fx(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection3_new_containing_point_oracle() {
    let mut cases = oracle::reflection3_new_containing_point_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, pt, e, tol) = *case;
        let got = Reflection3Trait::new_containing_point(Unit { value: vec3(axis) }, pt3(pt));
        worst =
            core::cmp::max(
                worst,
                report("reflection3_new_containing_point", n, ulp_diff(got.bias(), fx(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection5_new_containing_point_oracle() {
    let mut cases = oracle::reflection5_new_containing_point_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, pt, e, tol) = *case;
        let got = Reflection5Trait::new_containing_point(Unit { value: vec5(axis) }, pt5(pt));
        worst =
            core::cmp::max(
                worst,
                report("reflection5_new_containing_point", n, ulp_diff(got.bias(), fx(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection6_new_containing_point_oracle() {
    let mut cases = oracle::reflection6_new_containing_point_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, pt, e, tol) = *case;
        let got = Reflection6Trait::new_containing_point(Unit { value: vec6(axis) }, pt6(pt));
        worst =
            core::cmp::max(
                worst,
                report("reflection6_new_containing_point", n, ulp_diff(got.bias(), fx(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection1_reflect_cols2_oracle() {
    let mut cases = oracle::reflection1_reflect_cols2_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, tol) = *case;
        let r = Reflection1Trait::new(Unit { value: vec1(axis) }, fx(bias));
        let mut got = mat1x2(m);
        r.reflect(ref got);
        worst =
            core::cmp::max(
                worst, report("reflection1_reflect_cols2", n, err_mat1x2(got, mat1x2(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection2_reflect_cols3_oracle() {
    let mut cases = oracle::reflection2_reflect_cols3_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, tol) = *case;
        let r = Reflection2Trait::new(Unit { value: vec2(axis) }, fx(bias));
        let mut got = mat2x3(m);
        r.reflect(ref got);
        worst =
            core::cmp::max(
                worst, report("reflection2_reflect_cols3", n, err_mat2x3(got, mat2x3(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection3_reflect_cols1_oracle() {
    let mut cases = oracle::reflection3_reflect_cols1_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, tol) = *case;
        let r = Reflection3Trait::new(Unit { value: vec3(axis) }, fx(bias));
        let mut got = mat3x1(m);
        r.reflect(ref got);
        worst =
            core::cmp::max(
                worst, report("reflection3_reflect_cols1", n, err_mat3x1(got, mat3x1(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection3_reflect_cols2_oracle() {
    let mut cases = oracle::reflection3_reflect_cols2_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, tol) = *case;
        let r = Reflection3Trait::new(Unit { value: vec3(axis) }, fx(bias));
        let mut got = mat3x2(m);
        r.reflect(ref got);
        worst =
            core::cmp::max(
                worst, report("reflection3_reflect_cols2", n, err_mat3x2(got, mat3x2(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection4_reflect_cols2_oracle() {
    let mut cases = oracle::reflection4_reflect_cols2_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, tol) = *case;
        let r = Reflection4Trait::new(Unit { value: vec4(axis) }, fx(bias));
        let mut got = mat4x2(m);
        r.reflect(ref got);
        worst =
            core::cmp::max(
                worst, report("reflection4_reflect_cols2", n, err_mat4x2(got, mat4x2(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection5_reflect_cols1_oracle() {
    let mut cases = oracle::reflection5_reflect_cols1_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, tol) = *case;
        let r = Reflection5Trait::new(Unit { value: vec5(axis) }, fx(bias));
        let mut got = mat5x1(m);
        r.reflect(ref got);
        worst =
            core::cmp::max(
                worst, report("reflection5_reflect_cols1", n, err_mat5x1(got, mat5x1(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection6_reflect_cols1_oracle() {
    let mut cases = oracle::reflection6_reflect_cols1_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, tol) = *case;
        let r = Reflection6Trait::new(Unit { value: vec6(axis) }, fx(bias));
        let mut got = mat6x1(m);
        r.reflect(ref got);
        worst =
            core::cmp::max(
                worst, report("reflection6_reflect_cols1", n, err_mat6x1(got, mat6x1(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection2_reflect_with_sign_cols2_oracle() {
    let mut cases = oracle::reflection2_reflect_with_sign_cols2_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, sign, e, tol) = *case;
        let r = Reflection2Trait::new(Unit { value: vec2(axis) }, fx(bias));
        let mut got = mat2x2(m);
        r.reflect_with_sign(ref got, fx(sign));
        worst =
            core::cmp::max(
                worst,
                report("reflection2_reflect_with_sign_cols2", n, err_mat2x2(got, mat2x2(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection3_reflect_with_sign_cols2_oracle() {
    let mut cases = oracle::reflection3_reflect_with_sign_cols2_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, sign, e, tol) = *case;
        let r = Reflection3Trait::new(Unit { value: vec3(axis) }, fx(bias));
        let mut got = mat3x2(m);
        r.reflect_with_sign(ref got, fx(sign));
        worst =
            core::cmp::max(
                worst,
                report("reflection3_reflect_with_sign_cols2", n, err_mat3x2(got, mat3x2(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection5_reflect_with_sign_cols2_oracle() {
    let mut cases = oracle::reflection5_reflect_with_sign_cols2_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, sign, e, tol) = *case;
        let r = Reflection5Trait::new(Unit { value: vec5(axis) }, fx(bias));
        let mut got = mat5x2(m);
        r.reflect_with_sign(ref got, fx(sign));
        worst =
            core::cmp::max(
                worst,
                report("reflection5_reflect_with_sign_cols2", n, err_mat5x2(got, mat5x2(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection1_reflect_rows3_oracle() {
    let mut cases = oracle::reflection1_reflect_rows3_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, we, tol) = *case;
        let r = Reflection1Trait::new(Unit { value: vec1(axis) }, fx(bias));
        let mut got = mat3x1(m);
        let mut work = vec3((0, 0, 0));
        r.reflect_rows(ref got, ref work);
        let err = core::cmp::max(err_mat3x1(got, mat3x1(e)), err_mat3x1(work, vec3(we)));
        worst = core::cmp::max(worst, report("reflection1_reflect_rows3", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection2_reflect_rows3_oracle() {
    let mut cases = oracle::reflection2_reflect_rows3_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, we, tol) = *case;
        let r = Reflection2Trait::new(Unit { value: vec2(axis) }, fx(bias));
        let mut got = mat3x2(m);
        let mut work = vec3((0, 0, 0));
        r.reflect_rows(ref got, ref work);
        let err = core::cmp::max(err_mat3x2(got, mat3x2(e)), err_mat3x1(work, vec3(we)));
        worst = core::cmp::max(worst, report("reflection2_reflect_rows3", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection3_reflect_rows2_oracle() {
    let mut cases = oracle::reflection3_reflect_rows2_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, we, tol) = *case;
        let r = Reflection3Trait::new(Unit { value: vec3(axis) }, fx(bias));
        let mut got = mat2x3(m);
        let mut work = vec2((0, 0));
        r.reflect_rows(ref got, ref work);
        let err = core::cmp::max(err_mat2x3(got, mat2x3(e)), err_mat2x1(work, vec2(we)));
        worst = core::cmp::max(worst, report("reflection3_reflect_rows2", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection4_reflect_rows1_oracle() {
    let mut cases = oracle::reflection4_reflect_rows1_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, we, tol) = *case;
        let r = Reflection4Trait::new(Unit { value: vec4(axis) }, fx(bias));
        let mut got = mat1x4(m);
        let mut work = vec1((0,));
        r.reflect_rows(ref got, ref work);
        let err = core::cmp::max(err_mat1x4(got, mat1x4(e)), err_mat1x1(work, vec1(we)));
        worst = core::cmp::max(worst, report("reflection4_reflect_rows1", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection5_reflect_rows2_oracle() {
    let mut cases = oracle::reflection5_reflect_rows2_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, we, tol) = *case;
        let r = Reflection5Trait::new(Unit { value: vec5(axis) }, fx(bias));
        let mut got = mat2x5(m);
        let mut work = vec2((0, 0));
        r.reflect_rows(ref got, ref work);
        let err = core::cmp::max(err_mat2x5(got, mat2x5(e)), err_mat2x1(work, vec2(we)));
        worst = core::cmp::max(worst, report("reflection5_reflect_rows2", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection6_reflect_rows2_oracle() {
    let mut cases = oracle::reflection6_reflect_rows2_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, e, we, tol) = *case;
        let r = Reflection6Trait::new(Unit { value: vec6(axis) }, fx(bias));
        let mut got = mat2x6(m);
        let mut work = vec2((0, 0));
        r.reflect_rows(ref got, ref work);
        let err = core::cmp::max(err_mat2x6(got, mat2x6(e)), err_mat2x1(work, vec2(we)));
        worst = core::cmp::max(worst, report("reflection6_reflect_rows2", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection2_reflect_rows_with_sign2_oracle() {
    let mut cases = oracle::reflection2_reflect_rows_with_sign2_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, sign, e, we, tol) = *case;
        let r = Reflection2Trait::new(Unit { value: vec2(axis) }, fx(bias));
        let mut got = mat2x2(m);
        let mut work = vec2((0, 0));
        r.reflect_rows_with_sign(ref got, ref work, fx(sign));
        let err = core::cmp::max(err_mat2x2(got, mat2x2(e)), err_mat2x1(work, vec2(we)));
        worst = core::cmp::max(worst, report("reflection2_reflect_rows_with_sign2", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection3_reflect_rows_with_sign3_oracle() {
    let mut cases = oracle::reflection3_reflect_rows_with_sign3_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, sign, e, we, tol) = *case;
        let r = Reflection3Trait::new(Unit { value: vec3(axis) }, fx(bias));
        let mut got = mat3x3(m);
        let mut work = vec3((0, 0, 0));
        r.reflect_rows_with_sign(ref got, ref work, fx(sign));
        let err = core::cmp::max(err_mat3x3(got, mat3x3(e)), err_mat3x1(work, vec3(we)));
        worst = core::cmp::max(worst, report("reflection3_reflect_rows_with_sign3", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_reflection5_reflect_rows_with_sign1_oracle() {
    let mut cases = oracle::reflection5_reflect_rows_with_sign1_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (axis, bias, m, sign, e, we, tol) = *case;
        let r = Reflection5Trait::new(Unit { value: vec5(axis) }, fx(bias));
        let mut got = mat1x5(m);
        let mut work = vec1((0,));
        r.reflect_rows_with_sign(ref got, ref work, fx(sign));
        let err = core::cmp::max(err_mat1x5(got, mat1x5(e)), err_mat1x1(work, vec1(we)));
        worst = core::cmp::max(worst, report("reflection5_reflect_rows_with_sign1", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}
