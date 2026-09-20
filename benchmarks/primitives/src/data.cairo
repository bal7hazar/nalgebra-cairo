//! Data structures: arrays, spans, fixed-size arrays, tuples, structs (by value / snapshot /
//! boxed) and `Felt252Dict`.
//!
//! Groups: `array8` (read 8 elements), `build8` (create an 8-element container), `vec3`
//! (struct/tuple construction and destructuring), `pass_mat4` (passing a 16-field struct to a
//! non-inlined function 4 times), `dict8`, `box_u64`, `box_mat4`.

use core::dict::Felt252Dict;
use harness::black_box;

// ------------------------------------------------------------------------------------ reads

pub fn read8_array_at(a: @Array<u64>) -> u64 {
    *a.at(0) + *a.at(1) + *a.at(2) + *a.at(3) + *a.at(4) + *a.at(5) + *a.at(6) + *a.at(7)
}

pub fn read8_span_index(s: Span<u64>) -> u64 {
    *s[0] + *s[1] + *s[2] + *s[3] + *s[4] + *s[5] + *s[6] + *s[7]
}

pub fn read8_span_get(s: Span<u64>) -> u64 {
    *s.get(0).unwrap().unbox()
        + *s.get(1).unwrap().unbox()
        + *s.get(2).unwrap().unbox()
        + *s.get(3).unwrap().unbox()
        + *s.get(4).unwrap().unbox()
        + *s.get(5).unwrap().unbox()
        + *s.get(6).unwrap().unbox()
        + *s.get(7).unwrap().unbox()
}

pub fn read8_span_pop_front(s: Span<u64>) -> u64 {
    let mut s = s;
    *s.pop_front().unwrap()
        + *s.pop_front().unwrap()
        + *s.pop_front().unwrap()
        + *s.pop_front().unwrap()
        + *s.pop_front().unwrap()
        + *s.pop_front().unwrap()
        + *s.pop_front().unwrap()
        + *s.pop_front().unwrap()
}

pub fn read8_span_multi_pop(s: Span<u64>) -> u64 {
    let mut s = s;
    let [a, b, c, d, e, f, g, h] = (*s.multi_pop_front::<8>().unwrap()).unbox();
    a + b + c + d + e + f + g + h
}

pub fn read8_span_try_into_fixed(s: Span<u64>) -> u64 {
    let boxed: @Box<[u64; 8]> = s.try_into().unwrap();
    let [a, b, c, d, e, f, g, h] = boxed.as_snapshot().unbox();
    *a + *b + *c + *d + *e + *f + *g + *h
}

pub fn read8_fixed_destructure(v: [u64; 8]) -> u64 {
    let [a, b, c, d, e, f, g, h] = v;
    a + b + c + d + e + f + g + h
}

pub fn read8_fixed_span_index(v: [u64; 8]) -> u64 {
    read8_span_index(v.span())
}

pub fn read8_tuple(v: (u64, u64, u64, u64, u64, u64, u64, u64)) -> u64 {
    let (a, b, c, d, e, f, g, h) = v;
    a + b + c + d + e + f + g + h
}

/// One random access (index black-boxed).
pub fn read1_span_index(s: Span<u64>, k: u32) -> u64 {
    *s[k]
}

pub fn read1_fixed_span_index(v: @[u64; 8], k: u32) -> u64 {
    *v.span()[k]
}

// ------------------------------------------------------------------------------------ builds

pub fn build8_array_macro(x: u64) -> Array<u64> {
    array![x, x, x, x, x, x, x, x]
}

pub fn build8_array_append(x: u64) -> Array<u64> {
    let mut a = ArrayTrait::new();
    a.append(x);
    a.append(x);
    a.append(x);
    a.append(x);
    a.append(x);
    a.append(x);
    a.append(x);
    a.append(x);
    a
}

pub fn build8_array_loop(x: u64) -> Array<u64> {
    let mut a = ArrayTrait::new();
    for _ in 0..8_u32 {
        a.append(x);
    }
    a
}

pub fn build8_fixed(x: u64) -> [u64; 8] {
    [x, x, x, x, x, x, x, x]
}

pub fn build8_tuple(x: u64) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
    (x, x, x, x, x, x, x, x)
}

// ------------------------------------------------------------------------------------ vec3

#[derive(Copy, Drop, PartialEq)]
pub struct Vec3 {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

pub fn vec3_struct_roundtrip(x: u64, y: u64, z: u64) -> u64 {
    let v = Vec3 { x, y, z };
    let Vec3 { x, y, z } = v;
    x + y + z
}

pub fn vec3_tuple_roundtrip(x: u64, y: u64, z: u64) -> u64 {
    let v = (x, y, z);
    let (x, y, z) = v;
    x + y + z
}

pub fn vec3_fixed_roundtrip(x: u64, y: u64, z: u64) -> u64 {
    let v = [x, y, z];
    let [x, y, z] = v;
    x + y + z
}

pub fn vec3_plain(x: u64, y: u64, z: u64) -> u64 {
    x + y + z
}

/// Struct built and consumed across a non-inlined call boundary.
#[inline(never)]
pub fn vec3_make(x: u64, y: u64, z: u64) -> Vec3 {
    Vec3 { x, y, z }
}

#[inline(never)]
pub fn vec3_sum(v: Vec3) -> u64 {
    v.x + v.y + v.z
}

#[inline(never)]
pub fn vec3_sum_snap(v: @Vec3) -> u64 {
    *v.x + *v.y + *v.z
}

#[inline(never)]
pub fn vec3_sum_fields(x: u64, y: u64, z: u64) -> u64 {
    x + y + z
}

// ------------------------------------------------------------------------------------ mat4

#[derive(Copy, Drop)]
pub struct Mat4 {
    pub m00: u64,
    pub m01: u64,
    pub m02: u64,
    pub m03: u64,
    pub m10: u64,
    pub m11: u64,
    pub m12: u64,
    pub m13: u64,
    pub m20: u64,
    pub m21: u64,
    pub m22: u64,
    pub m23: u64,
    pub m30: u64,
    pub m31: u64,
    pub m32: u64,
    pub m33: u64,
}

pub fn mat4(x: u64) -> Mat4 {
    Mat4 {
        m00: x,
        m01: x,
        m02: x,
        m03: x,
        m10: x,
        m11: x,
        m12: x,
        m13: x,
        m20: x,
        m21: x,
        m22: x,
        m23: x,
        m30: x,
        m31: x,
        m32: x,
        m33: x,
    }
}

#[inline(never)]
pub fn trace_value(m: Mat4) -> u64 {
    m.m00 + m.m11 + m.m22 + m.m33
}

#[inline(never)]
pub fn trace_snap(m: @Mat4) -> u64 {
    *m.m00 + *m.m11 + *m.m22 + *m.m33
}

#[inline(never)]
pub fn trace_box(m: Box<Mat4>) -> u64 {
    let m = m.unbox();
    m.m00 + m.m11 + m.m22 + m.m33
}

#[inline(never)]
pub fn trace_span(m: Span<u64>) -> u64 {
    *m[0] + *m[5] + *m[10] + *m[15]
}

#[inline(never)]
pub fn trace_fixed(m: [u64; 16]) -> u64 {
    let [m00, _, _, _, _, m11, _, _, _, _, m22, _, _, _, _, m33] = m;
    m00 + m11 + m22 + m33
}

#[inline(never)]
pub fn trace_inputs(a: u64, b: u64, c: u64, d: u64) -> u64 {
    a + b + c + d
}

// ------------------------------------------------------------------------------------ dict

pub fn dict8_insert_get(x: u64) -> u64 {
    let mut d: Felt252Dict<u64> = Default::default();
    d.insert(0, x);
    d.insert(1, x);
    d.insert(2, x);
    d.insert(3, x);
    d.insert(4, x);
    d.insert(5, x);
    d.insert(6, x);
    d.insert(7, x);
    d.get(0) + d.get(1) + d.get(2) + d.get(3) + d.get(4) + d.get(5) + d.get(6) + d.get(7)
}

pub fn dict8_insert_only(x: u64) -> u64 {
    let mut d: Felt252Dict<u64> = Default::default();
    d.insert(0, x);
    d.insert(1, x);
    d.insert(2, x);
    d.insert(3, x);
    d.insert(4, x);
    d.insert(5, x);
    d.insert(6, x);
    d.insert(7, x);
    x * 8
}

pub fn dict1_insert_get(x: u64) -> u64 {
    let mut d: Felt252Dict<u64> = Default::default();
    d.insert(0, x);
    d.get(0) * 8
}

pub fn array8_build_read(x: u64) -> u64 {
    read8_span_index(build8_array_macro(x).span())
}

// ------------------------------------------------------------------------------------ box

/// The box goes through `black_box`, otherwise the compiler cancels `into_box`/`unbox`.
pub fn box_roundtrip(x: u64) -> u64 {
    black_box(BoxTrait::new(x)).unbox()
}

/// 16-field struct moved through a non-inlined call inside a box (1 pointer)...
pub fn box_roundtrip_mat4(x: u64) -> u64 {
    let m = black_box(BoxTrait::new(mat4(x))).unbox();
    m.m33
}

/// ... or by value (16 cells).
pub fn nobox_mat4(x: u64) -> u64 {
    let m = black_box(mat4(x));
    m.m33
}

#[cfg(test)]
mod tests {
    use harness::black_box;
    use super::*;

    #[test]
    #[inline(never)]
    fn bench_array8__baseline() {
        let a: Array<u64> = black_box(array![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(a.len() == 8);
    }

    #[test]
    #[inline(never)]
    fn bench_array8__array_at() {
        let a: Array<u64> = black_box(array![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(a.len() == 8);
        assert!(read8_array_at(@a) == 36);
    }

    #[test]
    #[inline(never)]
    fn bench_array8__span_index() {
        let a: Array<u64> = black_box(array![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(a.len() == 8);
        assert!(read8_span_index(a.span()) == 36);
    }

    #[test]
    #[inline(never)]
    fn bench_array8__span_get_unwrap() {
        let a: Array<u64> = black_box(array![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(a.len() == 8);
        assert!(read8_span_get(a.span()) == 36);
    }

    #[test]
    #[inline(never)]
    fn bench_array8__span_pop_front() {
        let a: Array<u64> = black_box(array![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(a.len() == 8);
        assert!(read8_span_pop_front(a.span()) == 36);
    }

    #[test]
    #[inline(never)]
    fn bench_array8__span_multi_pop_front() {
        let a: Array<u64> = black_box(array![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(a.len() == 8);
        assert!(read8_span_multi_pop(a.span()) == 36);
    }

    #[test]
    #[inline(never)]
    fn bench_array8__span_try_into_fixed() {
        let a: Array<u64> = black_box(array![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(a.len() == 8);
        assert!(read8_span_try_into_fixed(a.span()) == 36);
    }

    #[test]
    #[inline(never)]
    fn bench_array8__sum_of_8_locals_reference() {
        let a: Array<u64> = black_box(array![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(a.len() == 8);
        let x: u64 = black_box(4);
        assert!(x + x + x + x + x + x + x + x == 32);
    }

    #[test]
    #[inline(never)]
    fn bench_fixed8__baseline() {
        let v: [u64; 8] = black_box([1, 2, 3, 4, 5, 6, 7, 8]);
        let _ = v;
        assert!(black_box(36_u64) == 36);
    }

    #[test]
    #[inline(never)]
    fn bench_fixed8__destructure() {
        let v: [u64; 8] = black_box([1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(read8_fixed_destructure(v) == 36);
    }

    #[test]
    #[inline(never)]
    fn bench_fixed8__span_index() {
        let v: [u64; 8] = black_box([1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(read8_fixed_span_index(v) == 36);
    }

    #[test]
    #[inline(never)]
    fn bench_fixed8__tuple_destructure() {
        let v: (u64, u64, u64, u64, u64, u64, u64, u64) = black_box((1, 2, 3, 4, 5, 6, 7, 8));
        assert!(read8_tuple(v) == 36);
    }

    #[test]
    #[inline(never)]
    fn bench_index1__baseline() {
        let a: Array<u64> = black_box(array![1, 2, 3, 4, 5, 6, 7, 8]);
        let k: u32 = black_box(5);
        assert!(a.len() == 8);
        assert!(k == 5);
    }

    #[test]
    #[inline(never)]
    fn bench_index1__span_index() {
        let a: Array<u64> = black_box(array![1, 2, 3, 4, 5, 6, 7, 8]);
        let k: u32 = black_box(5);
        assert!(a.len() == 8);
        assert!(read1_span_index(a.span(), k) == 6);
    }

    #[test]
    #[inline(never)]
    fn bench_index1__fixed_array_span_index() {
        let a: Array<u64> = black_box(array![1, 2, 3, 4, 5, 6, 7, 8]);
        let v: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let k: u32 = black_box(5);
        assert!(a.len() == 8);
        assert!(read1_fixed_span_index(@v, k) == 6);
    }

    #[test]
    #[inline(never)]
    fn bench_build8__baseline() {
        let x: u64 = black_box(9);
        assert!(x == 9);
    }

    #[test]
    #[inline(never)]
    fn bench_build8__array_append_through_black_box() {
        let x: u64 = black_box(9);
        assert!(x == 9);
        assert!(black_box(build8_array_append(x)).len() == 8);
    }

    #[test]
    #[inline(never)]
    fn bench_build8__array_append_loop_through_black_box() {
        let x: u64 = black_box(9);
        assert!(x == 9);
        assert!(black_box(build8_array_loop(x)).len() == 8);
    }

    #[test]
    #[inline(never)]
    fn bench_build8__fixed_array_through_black_box() {
        let x: u64 = black_box(9);
        assert!(x == 9);
        let [_, _, _, _, _, _, _, h] = black_box(build8_fixed(x));
        assert!(h == 9);
    }

    #[test]
    #[inline(never)]
    fn bench_build8__tuple_through_black_box() {
        let x: u64 = black_box(9);
        assert!(x == 9);
        let (_, _, _, _, _, _, _, h) = black_box(build8_tuple(x));
        assert!(h == 9);
    }

    #[test]
    #[inline(never)]
    fn bench_build8__array_macro_through_black_box() {
        let x: u64 = black_box(9);
        assert!(x == 9);
        assert!(black_box(build8_array_macro(x)).len() == 8);
    }

    #[test]
    #[inline(never)]
    fn bench_vec3__baseline() {
        let (x, y, z): (u64, u64, u64) = (black_box(1), black_box(2), black_box(3));
        let _ = (y, z);
        assert!(x == 1);
    }

    #[test]
    #[inline(never)]
    fn bench_vec3__plain_locals() {
        let (x, y, z): (u64, u64, u64) = (black_box(1), black_box(2), black_box(3));
        assert!(vec3_plain(x, y, z) == 6);
    }

    #[test]
    #[inline(never)]
    fn bench_vec3__struct_roundtrip_inlined() {
        let (x, y, z): (u64, u64, u64) = (black_box(1), black_box(2), black_box(3));
        assert!(vec3_struct_roundtrip(x, y, z) == 6);
    }

    #[test]
    #[inline(never)]
    fn bench_vec3__tuple_roundtrip_inlined() {
        let (x, y, z): (u64, u64, u64) = (black_box(1), black_box(2), black_box(3));
        assert!(vec3_tuple_roundtrip(x, y, z) == 6);
    }

    #[test]
    #[inline(never)]
    fn bench_vec3__fixed_array_roundtrip_inlined() {
        let (x, y, z): (u64, u64, u64) = (black_box(1), black_box(2), black_box(3));
        assert!(vec3_fixed_roundtrip(x, y, z) == 6);
    }

    #[test]
    #[inline(never)]
    fn bench_vec3__call_with_fields() {
        let (x, y, z): (u64, u64, u64) = (black_box(1), black_box(2), black_box(3));
        assert!(vec3_sum_fields(x, y, z) == 6);
    }

    #[test]
    #[inline(never)]
    fn bench_vec3__call_with_struct_value() {
        let (x, y, z): (u64, u64, u64) = (black_box(1), black_box(2), black_box(3));
        assert!(vec3_sum(Vec3 { x, y, z }) == 6);
    }

    #[test]
    #[inline(never)]
    fn bench_vec3__call_with_struct_snapshot() {
        let (x, y, z): (u64, u64, u64) = (black_box(1), black_box(2), black_box(3));
        assert!(vec3_sum_snap(@Vec3 { x, y, z }) == 6);
    }

    #[test]
    #[inline(never)]
    fn bench_vec3__call_make_then_sum() {
        let (x, y, z): (u64, u64, u64) = (black_box(1), black_box(2), black_box(3));
        assert!(vec3_sum(vec3_make(x, y, z)) == 6);
    }

    #[test]
    #[inline(never)]
    fn bench_pass_mat4__baseline() {
        let x: u64 = black_box(5);
        assert!(x * 4 * 4 == 80);
    }

    #[test]
    #[inline(never)]
    fn bench_pass_mat4__four_scalars_x4() {
        let x: u64 = black_box(5);
        assert!(x * 4 * 4 == 80);
        let r = trace_inputs(x, x, x, x)
            + trace_inputs(x, x, x, x)
            + trace_inputs(x, x, x, x)
            + trace_inputs(x, x, x, x);
        assert!(r == 80);
    }

    #[test]
    #[inline(never)]
    fn bench_pass_mat4__by_value_x4() {
        let x: u64 = black_box(5);
        assert!(x * 4 * 4 == 80);
        let m = mat4(x);
        assert!(trace_value(m) + trace_value(m) + trace_value(m) + trace_value(m) == 80);
    }

    #[test]
    #[inline(never)]
    fn bench_pass_mat4__by_snapshot_x4() {
        let x: u64 = black_box(5);
        assert!(x * 4 * 4 == 80);
        let m = mat4(x);
        assert!(trace_snap(@m) + trace_snap(@m) + trace_snap(@m) + trace_snap(@m) == 80);
    }

    #[test]
    #[inline(never)]
    fn bench_pass_mat4__boxed_x4() {
        let x: u64 = black_box(5);
        assert!(x * 4 * 4 == 80);
        let m = BoxTrait::new(mat4(x));
        assert!(trace_box(m) + trace_box(m) + trace_box(m) + trace_box(m) == 80);
    }

    #[test]
    #[inline(never)]
    fn bench_pass_mat4__span_x4() {
        let x: u64 = black_box(5);
        assert!(x * 4 * 4 == 80);
        let m = array![x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x].span();
        assert!(trace_span(m) + trace_span(m) + trace_span(m) + trace_span(m) == 80);
    }

    #[test]
    #[inline(never)]
    fn bench_pass_mat4__fixed_array_x4() {
        let x: u64 = black_box(5);
        assert!(x * 4 * 4 == 80);
        let m = [x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x];
        assert!(trace_fixed(m) + trace_fixed(m) + trace_fixed(m) + trace_fixed(m) == 80);
    }

    #[test]
    #[inline(never)]
    fn bench_dict8__baseline() {
        let x: u64 = black_box(5);
        assert!(x * 8 == 40);
    }

    #[test]
    #[inline(never)]
    fn bench_dict8__dict_1_insert_1_get() {
        let x: u64 = black_box(5);
        assert!(dict1_insert_get(x) == 40);
    }

    #[test]
    #[inline(never)]
    fn bench_dict8__dict_8_inserts() {
        let x: u64 = black_box(5);
        assert!(dict8_insert_only(x) == 40);
    }

    #[test]
    #[inline(never)]
    fn bench_dict8__dict_8_inserts_8_gets() {
        let x: u64 = black_box(5);
        assert!(dict8_insert_get(x) == 40);
    }

    #[test]
    #[inline(never)]
    fn bench_dict8__array_8_appends_8_reads() {
        let x: u64 = black_box(5);
        assert!(array8_build_read(x) == 40);
    }

    #[test]
    #[inline(never)]
    fn bench_box_u64__baseline() {
        let x: u64 = black_box(5);
        assert!(black_box(x) == 5);
    }

    #[test]
    #[inline(never)]
    fn bench_box_u64__box_black_box_unbox() {
        let x: u64 = black_box(5);
        assert!(box_roundtrip(x) == 5);
    }

    #[test]
    #[inline(never)]
    fn bench_box_mat4__baseline() {
        let x: u64 = black_box(5);
        assert!(nobox_mat4(x) == 5);
    }

    #[test]
    #[inline(never)]
    fn bench_box_mat4__box_black_box_unbox() {
        let x: u64 = black_box(5);
        assert!(box_roundtrip_mat4(x) == 5);
    }
}
