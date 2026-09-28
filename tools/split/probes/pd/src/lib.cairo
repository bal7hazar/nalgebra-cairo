mod q1_mul_other_module_same_crate {
    use pa::A;
    fn f() -> A { A { x: 2 } * A { x: 3 } }
}
mod q2_into_in_rhs_type_module {
    use pa::A;
    use pb::B;
    fn f() -> B { A { x: 1 }.into() }
}
mod q3_trait_via_facade_path_unchanged {
    use pa::ATrait;
    fn f() -> u32 { ATrait::new(1).get() }
}
