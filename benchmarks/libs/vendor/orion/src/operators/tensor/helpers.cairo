// EXTRACTED from orion @ bac0b42 (src/operators/tensor/helpers.cairo): only the items needed by the benchmarks are kept;
// function bodies are verbatim unless marked PATCH.

use orion::utils::u32_max;
use orion::operators::tensor::core::{Tensor, TensorTrait, stride};

fn len_from_shape(mut shape: Span<usize>) -> usize {
    let mut result: usize = 1;

    loop {
        match shape.pop_front() {
            Option::Some(item) => { result *= *item; },
            Option::None => { break; }
        };
    };

    result
}

fn check_shape<T>(shape: Span<usize>, data: Span<T>) {
    assert(len_from_shape(shape) == data.len(), 'wrong tensor shape');
}

fn check_compatibility(mut shape_1: Span<usize>, mut shape_2: Span<usize>) {
    // Start from the last dimension by getting the length of each shape
    let mut iter_1 = shape_1.len();
    let mut iter_2 = shape_2.len();

    // Iterate while there are dimensions left in either shape
    while iter_1 > 0 || iter_2 > 0 {
        // Get the current dimension for each shape, defaulting to 1 if we've run out of dimensions
        let dim_1 = if iter_1 > 0 {
            *shape_1[iter_1 - 1]
        } else {
            1
        };
        let dim_2 = if iter_2 > 0 {
            *shape_2[iter_2 - 1]
        } else {
            1
        };

        // Check the broadcasting rule for the current dimension
        if dim_1 != dim_2 && dim_1 != 1 && dim_2 != 1 {
            panic(array!['tensors shape must match']);
        }

        // Move to the next dimension
        if iter_1 > 0 {
            iter_1 -= 1;
        }
        if iter_2 > 0 {
            iter_2 -= 1;
        }
    }
}

fn broadcast_index_mapping(mut shape: Span<usize>, mut indices: Span<usize>) -> usize {
    if shape.len() == indices.len() {
        broadcast_index_mapping_equal_shape(shape, indices)
    } else {
        broadcast_index_mapping_non_equal_shape(shape, indices)
    }
}

fn broadcast_index_mapping_equal_shape(mut shape: Span<usize>, mut indices: Span<usize>) -> usize {
    let mut result = 0_usize;
    let mut stride = stride(shape);

    loop {
        match shape.pop_front() {
            Option::Some(shape_val) => {
                let indices_val = *indices.pop_front().unwrap();
                let stride_val = *stride.pop_front().unwrap();

                let index = (indices_val % *shape_val) * stride_val;
                result += index;
            },
            Option::None => { break; }
        };
    };

    result
}

fn broadcast_index_mapping_non_equal_shape(
    mut shape: Span<usize>, mut indices: Span<usize>
) -> usize {
    let mut result = 0_usize;
    let mut stride = stride(shape.clone());

    // Calculate the offset to align indices with the rightmost dimensions of the shape
    let mut offset = if shape.len() > indices.len() {
        shape.len() - indices.len()
    } else {
        0
    };

    loop {
        match shape.pop_back() {
            Option::Some(_) => {
                let stride_val = stride
                    .pop_back()
                    .unwrap_or(@1); // Default stride for non-existent dimensions is 1

                // Calculate the index, using 0 for dimensions beyond the length of indices
                let index_val = if offset > 0 {
                    offset -= 1; // Decrement offset until we align indices with the shape
                    0 // Use 0 for indices beyond the length of the indices span
                } else {
                    *indices
                        .pop_back()
                        .unwrap_or(@0) // Use actual index value or 0 if indices are exhausted
                };

                let index = index_val * *stride_val;
                result += index;
            },
            Option::None => { break; }
        };
    };

    result
}

fn broadcast_shape(mut shape1: Span<usize>, mut shape2: Span<usize>) -> Span<usize> {
    check_compatibility(shape1, shape2);
    let mut result: Array<usize> = array![];

    while !shape1.is_empty() || !shape2.is_empty() {
        let dim1 = *shape1.pop_back().unwrap_or(@1);
        let dim2 = *shape2.pop_back().unwrap_or(@1);

        let broadcasted_dim = u32_max(dim1, dim2);
        result.append(broadcasted_dim);
    };

    // PATCH: was result.reverse().span() (alexandria ArrayTraitExt)
    orion::utils::reverse(result.span()).span()
}
