mod e1_mulm_via_facade {
    use pf::{A, B, MulM};
    fn f() -> B { A { x: 2 }.mul_mat(B { x: 3 }) }
}
mod e2_add_third_crate_named_type_only {
    use pf::A;
    fn f() -> A { A { x: 2 } + A { x: 2 } }
}
mod e2b_add_third_crate_facade_glob {
    use pf::*;
    fn f() -> A { A { x: 2 } + A { x: 2 } }
}
mod e3_base_path {
    fn f() -> pf::base::A { pf::base::ATrait::new(3) }
}
mod e4_submodule_path_through_glob {
    fn f() -> pf::shapes::A { pf::shapes::A { x: 1 } }
}
mod e5_macro_reexported {
    use pf::mk;
    fn f() -> pf::A { mk!(3) }
}
mod e6_facade_macro_defsite {
    use pf::mk2;
    fn f() -> pf::A { mk2!(3) }
}
mod e7_transitive_crate_name {
    fn f() -> pa::A { pa::A { x: 1 } }
}
mod e8_generate_trait_method_via_facade {
    use pf::{ATrait, A};
    fn f() -> u32 { ATrait::new(1).get() }
}
mod e9_method_without_trait_import {
    use pf::A;
    fn f(a: A) -> u32 { a.get() }
}
mod e10_glob_conflict_errors {
    fn f() -> felt252 { pf::base::errors::E }
}
mod e11_forwarded_feature {
    fn f() -> u32 { pf::extra() }
}
