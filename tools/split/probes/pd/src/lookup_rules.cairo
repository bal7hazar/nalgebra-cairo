// Each module probes one lookup situation; the build output lists which ones fail.
mod m1_impl_in_rhs_type_module {
    use pa::{A, MulM};
    use pb::B;
    fn f() -> B { let a = A { x: 2 }; let b = B { x: 3 }; a.mul_mat(b) }
}
mod m2_core_sub_in_third_module_no_import {
    use pa::A;
    fn f() -> A { let a = A { x: 2 }; a - a }
}
mod m2b_core_sub_imported {
    use pa::A;
    use pb::bmod::SubA;
    fn f() -> A { let a = A { x: 2 }; a - a }
}
mod m3_core_add_no_import {
    use pa::A;
    fn f() -> A { let a = A { x: 2 }; a + a }
}
mod m3b_core_add_glob {
    use pa::A;
    use pc::*;
    fn f() -> A { let a = A { x: 2 }; a + a }
}
mod m3c_core_add_named {
    use pa::A;
    use pc::AddA;
    fn f() -> A { let a = A { x: 2 }; a + a }
}
mod m4_foreign_trait_foreign_type_no_import {
    use pa::Tr;
    use pb::B;
    fn f() -> u32 { B { x: 1 }.tr() }
}
mod m4b_foreign_trait_foreign_type_import {
    use pa::Tr;
    use pb::B;
    use pc::TrB;
    fn f() -> u32 { B { x: 1 }.tr() }
}
mod m5_generate_trait_foreign_type {
    use pb::B;
    use pc::BExtTrait;
    fn f() -> u32 { B { x: 1 }.twice() }
}
mod m6_mulm_third_crate_no_import {
    use pa::{A, MulM};
    use pb::B;
    fn f() -> A { B { x: 1 }.mul_mat(A { x: 2 }) }
}
mod m6b_mulm_third_crate_import {
    use pa::{A, MulM};
    use pb::B;
    use pc::MulBA;
    fn f() -> A { B { x: 1 }.mul_mat(A { x: 2 }) }
}
mod m7_into_no_import {
    use pa::A;
    use pb::B;
    fn f() -> B { A { x: 1 }.into() }
}
mod m7b_into_import {
    use pa::A;
    use pb::B;
    use pc::IntoAB;
    fn f() -> B { A { x: 1 }.into() }
}
mod m8_macro {
    fn f() -> pa::A { pc::mk!(3) }
}
