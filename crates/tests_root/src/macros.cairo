//! The construction macros against the constructors they expand to (upstream
//! `tests/macros/matrix.rs`, `tests/macros/stack.rs`): every static shape of `matrix!`, every size
//! of `vector!` / `point!`, `dmatrix!` / `dvector!` (up to 4x4 like upstream, plus the largest
//! arm), `stack!` grids, and the trailing separators. Components through `black_box`.

use fixed::Fixed;
use nalgebra::base::point2::Point2Trait;
use nalgebra::base::point3::Point3Trait;
use nalgebra::{
    DMatrix, DMatrixTrait, DVector, DVectorTrait, Matrix1, Matrix1Trait, Matrix2, Matrix2Trait,
    Matrix2x3, Matrix2x3Trait, Matrix2x4, Matrix2x4Trait, Matrix2x5, Matrix2x5Trait, Matrix2x6,
    Matrix2x6Trait, Matrix3, Matrix3Trait, Matrix3x2, Matrix3x2Trait, Matrix3x4, Matrix3x4Trait,
    Matrix3x5, Matrix3x5Trait, Matrix3x6, Matrix3x6Trait, Matrix4, Matrix4Trait, Matrix4x2,
    Matrix4x2Trait, Matrix4x3, Matrix4x3Trait, Matrix4x5, Matrix4x5Trait, Matrix4x6, Matrix4x6Trait,
    Matrix5, Matrix5Trait, Matrix5x2, Matrix5x2Trait, Matrix5x3, Matrix5x3Trait, Matrix5x4,
    Matrix5x4Trait, Matrix5x6, Matrix5x6Trait, Matrix6, Matrix6Trait, Matrix6x2, Matrix6x2Trait,
    Matrix6x3, Matrix6x3Trait, Matrix6x4, Matrix6x4Trait, Matrix6x5, Matrix6x5Trait, Point1Trait,
    Point4Trait, Point5Trait, Point6Trait, RowVector2, RowVector2Trait, RowVector3, RowVector3Trait,
    RowVector4, RowVector4Trait, RowVector5, RowVector5Trait, RowVector6, RowVector6Trait, Vector2,
    Vector2Trait, Vector3, Vector3Trait, Vector4, Vector4Trait, Vector5, Vector5Trait, Vector6,
    Vector6Trait, dmatrix, dvector, matrix, point, stack, vector,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::int;

/// `n` distinct components `1, 2, ..`, through `black_box`.
fn vals(n: usize) -> Span<Fixed> {
    let mut out: Array<Fixed> = array![];
    let mut i: usize = 1;
    while i <= n {
        out.append(int(i.into()));
        i += 1;
    }
    black_box(out.span())
}

#[test]
fn test_matrix_macro_1_rows() {
    let v = vals(6);
    let m: Matrix1<Fixed> = matrix![*v[0]];
    assert_eq!(m, Matrix1Trait::new(*v[0]));
    let m: RowVector2<Fixed> = matrix![*v[0], *v[1]];
    assert_eq!(m, RowVector2Trait::new(*v[0], *v[1]));
    let m: RowVector3<Fixed> = matrix![*v[0], *v[1], *v[2]];
    assert_eq!(m, RowVector3Trait::new(*v[0], *v[1], *v[2]));
    let m: RowVector4<Fixed> = matrix![*v[0], *v[1], *v[2], *v[3]];
    assert_eq!(m, RowVector4Trait::new(*v[0], *v[1], *v[2], *v[3]));
    let m: RowVector5<Fixed> = matrix![*v[0], *v[1], *v[2], *v[3], *v[4]];
    assert_eq!(m, RowVector5Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4]));
    let m: RowVector6<Fixed> = matrix![*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]];
    assert_eq!(m, RowVector6Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]));
}

#[test]
fn test_matrix_macro_2_rows() {
    let v = vals(12);
    let m: Vector2<Fixed> = matrix![ * v[0]; * v[1]];
    assert_eq!(m, Vector2Trait::new(*v[0], *v[1]));
    let m: Matrix2<Fixed> = matrix![ * v[0], * v[1]; * v[2], * v[3]];
    assert_eq!(m, Matrix2Trait::new(*v[0], *v[1], *v[2], *v[3]));
    let m: Matrix2x3<Fixed> = matrix![ * v[0], * v[1], * v[2]; * v[3], * v[4], * v[5]];
    assert_eq!(m, Matrix2x3Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]));
    let m: Matrix2x4<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3];
        * v[4],
        * v[5],
        * v[6],
        * v[7]];
    assert_eq!(m, Matrix2x4Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7]));
    let m: Matrix2x5<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3],
        * v[4];
        * v[5],
        * v[6],
        * v[7],
        * v[8],
        * v[9]];
    assert_eq!(
        m,
        Matrix2x5Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7], *v[8], *v[9]),
    );
    let m: Matrix2x6<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8],
        * v[9],
        * v[10],
        * v[11]];
    assert_eq!(
        m,
        Matrix2x6Trait::new(
            *v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7], *v[8], *v[9], *v[10], *v[11],
        ),
    );
}

#[test]
fn test_matrix_macro_3_rows() {
    let v = vals(18);
    let m: Vector3<Fixed> = matrix![ * v[0]; * v[1]; * v[2]];
    assert_eq!(m, Vector3Trait::new(*v[0], *v[1], *v[2]));
    let m: Matrix3x2<Fixed> = matrix![ * v[0], * v[1]; * v[2], * v[3]; * v[4], * v[5]];
    assert_eq!(m, Matrix3x2Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]));
    let m: Matrix3<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2];
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8]];
    assert_eq!(m, Matrix3Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7], *v[8]));
    let m: Matrix3x4<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3];
        * v[4],
        * v[5],
        * v[6],
        * v[7];
        * v[8],
        * v[9],
        * v[10],
        * v[11]];
    assert_eq!(
        m,
        Matrix3x4Trait::new(
            *v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7], *v[8], *v[9], *v[10], *v[11],
        ),
    );
    let m: Matrix3x5<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3],
        * v[4];
        * v[5],
        * v[6],
        * v[7],
        * v[8],
        * v[9];
        * v[10],
        * v[11],
        * v[12],
        * v[13],
        * v[14]];
    assert_eq!(
        m,
        Matrix3x5Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
        ),
    );
    let m: Matrix3x6<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8],
        * v[9],
        * v[10],
        * v[11];
        * v[12],
        * v[13],
        * v[14],
        * v[15],
        * v[16],
        * v[17]];
    assert_eq!(
        m,
        Matrix3x6Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
            *v[16],
            *v[17],
        ),
    );
}

#[test]
fn test_matrix_macro_4_rows() {
    let v = vals(24);
    let m: Vector4<Fixed> = matrix![ * v[0]; * v[1]; * v[2]; * v[3]];
    assert_eq!(m, Vector4Trait::new(*v[0], *v[1], *v[2], *v[3]));
    let m: Matrix4x2<Fixed> = matrix![
        * v[0],
        * v[1];
        * v[2],
        * v[3];
        * v[4],
        * v[5];
        * v[6],
        * v[7]];
    assert_eq!(m, Matrix4x2Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7]));
    let m: Matrix4x3<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2];
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8];
        * v[9],
        * v[10],
        * v[11]];
    assert_eq!(
        m,
        Matrix4x3Trait::new(
            *v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7], *v[8], *v[9], *v[10], *v[11],
        ),
    );
    let m: Matrix4<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3];
        * v[4],
        * v[5],
        * v[6],
        * v[7];
        * v[8],
        * v[9],
        * v[10],
        * v[11];
        * v[12],
        * v[13],
        * v[14],
        * v[15]];
    assert_eq!(
        m,
        Matrix4Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
        ),
    );
    let m: Matrix4x5<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3],
        * v[4];
        * v[5],
        * v[6],
        * v[7],
        * v[8],
        * v[9];
        * v[10],
        * v[11],
        * v[12],
        * v[13],
        * v[14];
        * v[15],
        * v[16],
        * v[17],
        * v[18],
        * v[19]];
    assert_eq!(
        m,
        Matrix4x5Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
            *v[16],
            *v[17],
            *v[18],
            *v[19],
        ),
    );
    let m: Matrix4x6<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8],
        * v[9],
        * v[10],
        * v[11];
        * v[12],
        * v[13],
        * v[14],
        * v[15],
        * v[16],
        * v[17];
        * v[18],
        * v[19],
        * v[20],
        * v[21],
        * v[22],
        * v[23]];
    assert_eq!(
        m,
        Matrix4x6Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
            *v[16],
            *v[17],
            *v[18],
            *v[19],
            *v[20],
            *v[21],
            *v[22],
            *v[23],
        ),
    );
}

#[test]
fn test_matrix_macro_5_rows() {
    let v = vals(30);
    let m: Vector5<Fixed> = matrix![ * v[0]; * v[1]; * v[2]; * v[3]; * v[4]];
    assert_eq!(m, Vector5Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4]));
    let m: Matrix5x2<Fixed> = matrix![
        * v[0],
        * v[1];
        * v[2],
        * v[3];
        * v[4],
        * v[5];
        * v[6],
        * v[7];
        * v[8],
        * v[9]];
    assert_eq!(
        m,
        Matrix5x2Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7], *v[8], *v[9]),
    );
    let m: Matrix5x3<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2];
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8];
        * v[9],
        * v[10],
        * v[11];
        * v[12],
        * v[13],
        * v[14]];
    assert_eq!(
        m,
        Matrix5x3Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
        ),
    );
    let m: Matrix5x4<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3];
        * v[4],
        * v[5],
        * v[6],
        * v[7];
        * v[8],
        * v[9],
        * v[10],
        * v[11];
        * v[12],
        * v[13],
        * v[14],
        * v[15];
        * v[16],
        * v[17],
        * v[18],
        * v[19]];
    assert_eq!(
        m,
        Matrix5x4Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
            *v[16],
            *v[17],
            *v[18],
            *v[19],
        ),
    );
    let m: Matrix5<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3],
        * v[4];
        * v[5],
        * v[6],
        * v[7],
        * v[8],
        * v[9];
        * v[10],
        * v[11],
        * v[12],
        * v[13],
        * v[14];
        * v[15],
        * v[16],
        * v[17],
        * v[18],
        * v[19];
        * v[20],
        * v[21],
        * v[22],
        * v[23],
        * v[24]];
    assert_eq!(
        m,
        Matrix5Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
            *v[16],
            *v[17],
            *v[18],
            *v[19],
            *v[20],
            *v[21],
            *v[22],
            *v[23],
            *v[24],
        ),
    );
    let m: Matrix5x6<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8],
        * v[9],
        * v[10],
        * v[11];
        * v[12],
        * v[13],
        * v[14],
        * v[15],
        * v[16],
        * v[17];
        * v[18],
        * v[19],
        * v[20],
        * v[21],
        * v[22],
        * v[23];
        * v[24],
        * v[25],
        * v[26],
        * v[27],
        * v[28],
        * v[29]];
    assert_eq!(
        m,
        Matrix5x6Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
            *v[16],
            *v[17],
            *v[18],
            *v[19],
            *v[20],
            *v[21],
            *v[22],
            *v[23],
            *v[24],
            *v[25],
            *v[26],
            *v[27],
            *v[28],
            *v[29],
        ),
    );
}

#[test]
fn test_matrix_macro_6_rows() {
    let v = vals(36);
    let m: Vector6<Fixed> = matrix![ * v[0]; * v[1]; * v[2]; * v[3]; * v[4]; * v[5]];
    assert_eq!(m, Vector6Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]));
    let m: Matrix6x2<Fixed> = matrix![
        * v[0],
        * v[1];
        * v[2],
        * v[3];
        * v[4],
        * v[5];
        * v[6],
        * v[7];
        * v[8],
        * v[9];
        * v[10],
        * v[11]];
    assert_eq!(
        m,
        Matrix6x2Trait::new(
            *v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7], *v[8], *v[9], *v[10], *v[11],
        ),
    );
    let m: Matrix6x3<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2];
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8];
        * v[9],
        * v[10],
        * v[11];
        * v[12],
        * v[13],
        * v[14];
        * v[15],
        * v[16],
        * v[17]];
    assert_eq!(
        m,
        Matrix6x3Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
            *v[16],
            *v[17],
        ),
    );
    let m: Matrix6x4<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3];
        * v[4],
        * v[5],
        * v[6],
        * v[7];
        * v[8],
        * v[9],
        * v[10],
        * v[11];
        * v[12],
        * v[13],
        * v[14],
        * v[15];
        * v[16],
        * v[17],
        * v[18],
        * v[19];
        * v[20],
        * v[21],
        * v[22],
        * v[23]];
    assert_eq!(
        m,
        Matrix6x4Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
            *v[16],
            *v[17],
            *v[18],
            *v[19],
            *v[20],
            *v[21],
            *v[22],
            *v[23],
        ),
    );
    let m: Matrix6x5<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3],
        * v[4];
        * v[5],
        * v[6],
        * v[7],
        * v[8],
        * v[9];
        * v[10],
        * v[11],
        * v[12],
        * v[13],
        * v[14];
        * v[15],
        * v[16],
        * v[17],
        * v[18],
        * v[19];
        * v[20],
        * v[21],
        * v[22],
        * v[23],
        * v[24];
        * v[25],
        * v[26],
        * v[27],
        * v[28],
        * v[29]];
    assert_eq!(
        m,
        Matrix6x5Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
            *v[16],
            *v[17],
            *v[18],
            *v[19],
            *v[20],
            *v[21],
            *v[22],
            *v[23],
            *v[24],
            *v[25],
            *v[26],
            *v[27],
            *v[28],
            *v[29],
        ),
    );
    let m: Matrix6<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8],
        * v[9],
        * v[10],
        * v[11];
        * v[12],
        * v[13],
        * v[14],
        * v[15],
        * v[16],
        * v[17];
        * v[18],
        * v[19],
        * v[20],
        * v[21],
        * v[22],
        * v[23];
        * v[24],
        * v[25],
        * v[26],
        * v[27],
        * v[28],
        * v[29];
        * v[30],
        * v[31],
        * v[32],
        * v[33],
        * v[34],
        * v[35]];
    assert_eq!(
        m,
        Matrix6Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
            *v[16],
            *v[17],
            *v[18],
            *v[19],
            *v[20],
            *v[21],
            *v[22],
            *v[23],
            *v[24],
            *v[25],
            *v[26],
            *v[27],
            *v[28],
            *v[29],
            *v[30],
            *v[31],
            *v[32],
            *v[33],
            *v[34],
            *v[35],
        ),
    );
}

#[test]
fn test_matrix_macro_trailing_semicolon() {
    let v = vals(4);
    let m: Matrix2<Fixed> = matrix![ * v[0], * v[1]; * v[2], * v[3];];
    assert_eq!(m, Matrix2Trait::new(*v[0], *v[1], *v[2], *v[3]));
}

#[test]
fn test_vector_macro() {
    let v = vals(6);
    let x: Matrix1<Fixed> = vector![*v[0]];
    assert_eq!(x, Matrix1Trait::new(*v[0]));
    let x: Vector2<Fixed> = vector![*v[0], *v[1]];
    assert_eq!(x, Vector2Trait::new(*v[0], *v[1]));
    let x: Vector3<Fixed> = vector![*v[0], *v[1], *v[2]];
    assert_eq!(x, Vector3Trait::new(*v[0], *v[1], *v[2]));
    let x: Vector4<Fixed> = vector![*v[0], *v[1], *v[2], *v[3]];
    assert_eq!(x, Vector4Trait::new(*v[0], *v[1], *v[2], *v[3]));
    let x: Vector5<Fixed> = vector![*v[0], *v[1], *v[2], *v[3], *v[4]];
    assert_eq!(x, Vector5Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4]));
    let x: Vector6<Fixed> = vector![*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]];
    assert_eq!(x, Vector6Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]));
    // Trailing comma.
    assert_eq!(vector![*v[0], *v[1], *v[2]], Vector3Trait::new(*v[0], *v[1], *v[2]));
}

#[test]
fn test_point_macro() {
    let v = vals(6);
    assert_eq!(point![*v[0]], Point1Trait::new(*v[0]));
    assert_eq!(point![*v[0], *v[1]], Point2Trait::new(*v[0], *v[1]));
    assert_eq!(point![*v[0], *v[1], *v[2]], Point3Trait::new(*v[0], *v[1], *v[2]));
    assert_eq!(point![*v[0], *v[1], *v[2], *v[3]], Point4Trait::new(*v[0], *v[1], *v[2], *v[3]));
    assert_eq!(
        point![*v[0], *v[1], *v[2], *v[3], *v[4]],
        Point5Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4]),
    );
    assert_eq!(
        point![*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]],
        Point6Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]),
    );
    assert_eq!(point![*v[0], *v[1]], Point2Trait::new(*v[0], *v[1]));
}

#[test]
fn test_dmatrix_macro_small_dims_exhaustive() {
    let v = vals(16);
    assert_eq!(dmatrix![*v[0]], DMatrixTrait::from_row_slice(1, 1, v.slice(0, 1)));
    assert_eq!(dmatrix![*v[0], *v[1]], DMatrixTrait::from_row_slice(1, 2, v.slice(0, 2)));
    assert_eq!(dmatrix![*v[0], *v[1], *v[2]], DMatrixTrait::from_row_slice(1, 3, v.slice(0, 3)));
    assert_eq!(
        dmatrix![*v[0], *v[1], *v[2], *v[3]], DMatrixTrait::from_row_slice(1, 4, v.slice(0, 4)),
    );
    assert_eq!(dmatrix![ * v[0]; * v[1]], DMatrixTrait::from_row_slice(2, 1, v.slice(0, 2)));
    assert_eq!(
        dmatrix![ * v[0], * v[1]; * v[2], * v[3]],
        DMatrixTrait::from_row_slice(2, 2, v.slice(0, 4)),
    );
    assert_eq!(
        dmatrix![ * v[0], * v[1], * v[2]; * v[3], * v[4], * v[5]],
        DMatrixTrait::from_row_slice(2, 3, v.slice(0, 6)),
    );
    assert_eq!(
        dmatrix![ * v[0], * v[1], * v[2], * v[3]; * v[4], * v[5], * v[6], * v[7]],
        DMatrixTrait::from_row_slice(2, 4, v.slice(0, 8)),
    );
    assert_eq!(
        dmatrix![ * v[0]; * v[1]; * v[2]], DMatrixTrait::from_row_slice(3, 1, v.slice(0, 3)),
    );
    assert_eq!(
        dmatrix![ * v[0], * v[1]; * v[2], * v[3]; * v[4], * v[5]],
        DMatrixTrait::from_row_slice(3, 2, v.slice(0, 6)),
    );
    assert_eq!(
        dmatrix![ * v[0], * v[1], * v[2]; * v[3], * v[4], * v[5]; * v[6], * v[7], * v[8]],
        DMatrixTrait::from_row_slice(3, 3, v.slice(0, 9)),
    );
    assert_eq!(
        dmatrix![
            * v[0],
            * v[1],
            * v[2],
            * v[3];
            * v[4],
            * v[5],
            * v[6],
            * v[7];
            * v[8],
            * v[9],
            * v[10],
            * v[11]],
        DMatrixTrait::from_row_slice(3, 4, v.slice(0, 12)),
    );
    assert_eq!(
        dmatrix![ * v[0]; * v[1]; * v[2]; * v[3]],
        DMatrixTrait::from_row_slice(4, 1, v.slice(0, 4)),
    );
    assert_eq!(
        dmatrix![ * v[0], * v[1]; * v[2], * v[3]; * v[4], * v[5]; * v[6], * v[7]],
        DMatrixTrait::from_row_slice(4, 2, v.slice(0, 8)),
    );
    assert_eq!(
        dmatrix![
            * v[0],
            * v[1],
            * v[2];
            * v[3],
            * v[4],
            * v[5];
            * v[6],
            * v[7],
            * v[8];
            * v[9],
            * v[10],
            * v[11]],
        DMatrixTrait::from_row_slice(4, 3, v.slice(0, 12)),
    );
    assert_eq!(
        dmatrix![
            * v[0],
            * v[1],
            * v[2],
            * v[3];
            * v[4],
            * v[5],
            * v[6],
            * v[7];
            * v[8],
            * v[9],
            * v[10],
            * v[11];
            * v[12],
            * v[13],
            * v[14],
            * v[15]],
        DMatrixTrait::from_row_slice(4, 4, v.slice(0, 16)),
    );
}

#[test]
fn test_dmatrix_macro_16_rows() {
    let v = vals(32);
    let m: DMatrix<Fixed> = dmatrix![
        * v[0],
        * v[1];
        * v[2],
        * v[3];
        * v[4],
        * v[5];
        * v[6],
        * v[7];
        * v[8],
        * v[9];
        * v[10],
        * v[11];
        * v[12],
        * v[13];
        * v[14],
        * v[15];
        * v[16],
        * v[17];
        * v[18],
        * v[19];
        * v[20],
        * v[21];
        * v[22],
        * v[23];
        * v[24],
        * v[25];
        * v[26],
        * v[27];
        * v[28],
        * v[29];
        * v[30],
        * v[31]];
    assert_eq!(m, DMatrixTrait::from_row_slice(16, 2, v));
}

#[test]
fn test_dmatrix_macro_empty_and_trailing_semicolon() {
    let e: DMatrix<Fixed> = dmatrix![];
    assert_eq!(e, DMatrixTrait::zeros(0, 0));
    let v = vals(4);
    assert_eq!(dmatrix![ * v[0], * v[1]; * v[2], * v[3];], DMatrixTrait::from_row_slice(2, 2, v));
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_dmatrix_macro_ragged_rows() {
    let v = vals(5);
    let _m: DMatrix<Fixed> = dmatrix![ * v[0], * v[1], * v[2]; * v[3], * v[4]];
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_dmatrix_macro_ragged_rows_same_total() {
    // 3 + 1 components: 4 = 2 x 2, which `from_row_slice` alone would accept.
    let v = vals(4);
    let _m: DMatrix<Fixed> = dmatrix![ * v[0], * v[1], * v[2]; * v[3]];
}

#[test]
fn test_dvector_macro() {
    let v = vals(6);
    let e: DVector<Fixed> = dvector![];
    assert_eq!(e, DVectorTrait::zeros(0));
    assert_eq!(dvector![*v[0]], DVectorTrait::from_column_slice(v.slice(0, 1)));
    assert_eq!(dvector![*v[0], *v[1]], DVectorTrait::from_column_slice(v.slice(0, 2)));
    assert_eq!(dvector![*v[0], *v[1], *v[2]], DVectorTrait::from_column_slice(v.slice(0, 3)));
    assert_eq!(
        dvector![*v[0], *v[1], *v[2], *v[3]], DVectorTrait::from_column_slice(v.slice(0, 4)),
    );
    assert_eq!(
        dvector![*v[0], *v[1], *v[2], *v[3], *v[4]], DVectorTrait::from_column_slice(v.slice(0, 5)),
    );
    assert_eq!(
        dvector![*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]],
        DVectorTrait::from_column_slice(v.slice(0, 6)),
    );
    assert_eq!(dvector![*v[0], *v[1]], DVectorTrait::from_column_slice(v.slice(0, 2)));
}

#[test]
fn test_stack_upstream_example() {
    // Upstream's doc example, its zero block written out (`0` is not supported).
    let v = vals(10);
    let a: Matrix2<Fixed> = matrix![ * v[0], * v[1]; * v[2], * v[3]];
    let b: Matrix2<Fixed> = matrix![ * v[4], * v[5]; * v[6], * v[7]];
    let c: RowVector2<Fixed> = matrix![*v[8], *v[9]];
    let block: Matrix3x4<Fixed> = stack![a, b; c, RowVector2Trait::zeros ()];
    let z = int(0);
    let expected: Matrix3x4<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[4],
        * v[5];
        * v[2],
        * v[3],
        * v[6],
        * v[7];
        * v[8],
        * v[9], z, z];
    assert_eq!(block, expected);
}

#[test]
fn test_stack_block_diagonal() {
    // Upstream's `stack_simple`: the 4x4 identity from two 2x2 identities.
    let i: Matrix2<Fixed> = black_box(Matrix2Trait::identity());
    let z: Matrix2<Fixed> = black_box(Matrix2Trait::zeros());
    let m = stack![i, z; z, i;];
    assert_eq!(m, Matrix4Trait::identity());
}

#[test]
fn test_stack_one_block() {
    let v = vals(6);
    let a: Matrix2x3<Fixed> = matrix![ * v[0], * v[1], * v[2]; * v[3], * v[4], * v[5]];
    assert_eq!(stack![a], a);
}

#[test]
fn test_stack_rows_and_columns_of_scalars() {
    let v = vals(6);
    let s: Span<Matrix1<Fixed>> = array![
        Matrix1Trait::new(*v[0]), Matrix1Trait::new(*v[1]), Matrix1Trait::new(*v[2]),
        Matrix1Trait::new(*v[3]), Matrix1Trait::new(*v[4]), Matrix1Trait::new(*v[5]),
    ]
        .span();
    let row: RowVector6<Fixed> = stack![*s[0], *s[1], *s[2], *s[3], *s[4], *s[5]];
    assert_eq!(row, RowVector6Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]));
    let col: Vector6<Fixed> = stack![ * s[0]; * s[1]; * s[2]; * s[3]; * s[4]; * s[5]];
    assert_eq!(col, Vector6Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]));
}

#[test]
fn test_stack_vectors() {
    let v = vals(6);
    let a: Vector3<Fixed> = vector![*v[0], *v[1], *v[2]];
    let b: Vector3<Fixed> = vector![*v[3], *v[4], *v[5]];
    let col: Vector6<Fixed> = stack![a; b];
    assert_eq!(col, Vector6Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5]));
    let side = stack![a, b];
    assert_eq!(side, matrix![ * v[0], * v[3]; * v[1], * v[4]; * v[2], * v[5]]);
}

#[test]
fn test_stack_6x6_grid_of_scalars() {
    let v = vals(36);
    let mut blocks: Array<Matrix1<Fixed>> = array![];
    for x in v {
        blocks.append(Matrix1Trait::new(*x));
    }
    let b = blocks.span();
    let m: Matrix6<Fixed> = stack![
        * b[0],
        * b[1],
        * b[2],
        * b[3],
        * b[4],
        * b[5];
        * b[6],
        * b[7],
        * b[8],
        * b[9],
        * b[10],
        * b[11];
        * b[12],
        * b[13],
        * b[14],
        * b[15],
        * b[16],
        * b[17];
        * b[18],
        * b[19],
        * b[20],
        * b[21],
        * b[22],
        * b[23];
        * b[24],
        * b[25],
        * b[26],
        * b[27],
        * b[28],
        * b[29];
        * b[30],
        * b[31],
        * b[32],
        * b[33],
        * b[34],
        * b[35]];
    let expected: Matrix6<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2],
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8],
        * v[9],
        * v[10],
        * v[11];
        * v[12],
        * v[13],
        * v[14],
        * v[15],
        * v[16],
        * v[17];
        * v[18],
        * v[19],
        * v[20],
        * v[21],
        * v[22],
        * v[23];
        * v[24],
        * v[25],
        * v[26],
        * v[27],
        * v[28],
        * v[29];
        * v[30],
        * v[31],
        * v[32],
        * v[33],
        * v[34],
        * v[35]];
    assert_eq!(m, expected);
}

#[test]
fn test_stack_mixed_blocks() {
    // [Matrix3 | Vector3] over [RowVector3 | Matrix1]: a Matrix4.
    let v = vals(16);
    let a: Matrix3<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2];
        * v[4],
        * v[5],
        * v[6];
        * v[8],
        * v[9],
        * v[10]];
    let b: Vector3<Fixed> = vector![*v[3], *v[7], *v[11]];
    let c = matrix![*v[12], *v[13], *v[14]];
    let d = matrix![*v[15]];
    let m: Matrix4<Fixed> = stack![a, b; c, d];
    assert_eq!(
        m,
        Matrix4Trait::new(
            *v[0],
            *v[1],
            *v[2],
            *v[3],
            *v[4],
            *v[5],
            *v[6],
            *v[7],
            *v[8],
            *v[9],
            *v[10],
            *v[11],
            *v[12],
            *v[13],
            *v[14],
            *v[15],
        ),
    );
}
