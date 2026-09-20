// EXTRACTED from orion @ bac0b42 (src/utils.cairo): only the items needed by the benchmarks are kept;
// function bodies are verbatim unless marked PATCH.

fn u32_max(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}

// PATCH: verbatim copy of alexandria_data_structures::array_ext::SpanTraitExt::reverse @ 800f5ad
// (the alexandria revision pinned by orion, which no longer compiles), as a free function.
fn reverse<T, +Copy<T>, +Drop<T>>(mut self: Span<T>) -> Array<T> {
    let mut response = array![];
    loop {
        match self.pop_back() {
            Option::Some(v) => { response.append(*v); },
            Option::None => { break; },
        };
    };
    response
}
