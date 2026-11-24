// -----// IR Dump After PropagateReshape (propagate-reshape) //----- //
func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix", parallel_mode = "simd"} {
  %c1_i32 = arith.constant 1 : i32
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c8_i32 = arith.constant 8 : i32
  %c63_i32 = arith.constant 63 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant 0.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg14, %arg13 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg12 : i32
  %7 = arith.addi %arg6, %c63_i32 : i32
  %8 = arith.divsi %7, %c64_i32 : i32
  %9 = arith.addi %arg7, %c63_i32 : i32
  %10 = arith.divsi %9, %c64_i32 : i32
  %11 = arith.muli %10, %c8_i32 : i32
  %12 = arith.divsi %6, %11 : i32
  %13 = arith.muli %12, %c8_i32 : i32
  %14 = arith.subi %8, %13 : i32
  %15 = arith.minsi %14, %c8_i32 : i32
  %16 = arith.remsi %6, %15 : i32
  %17 = arith.addi %13, %16 : i32
  %18 = arith.remsi %6, %11 : i32
  %19 = arith.divsi %18, %15 : i32
  %20 = arith.muli %17, %c64_i32 : i32
  %21 = arith.muli %19, %c64_i32 : i32
  %22 = arith.addi %arg8, %c63_i32 : i32
  %23 = arith.divsi %22, %c64_i32 : i32
  %24 = arith.muli %arg10, %c64_i32 : i32
  %25 = arith.index_cast %20 : i32 to index
  %26 = arith.index_cast %arg9 : i32 to index
  %27 = arith.muli %25, %26 : index
  %28 = arith.index_cast %arg10 : i32 to index
  %29 = arith.index_cast %21 : i32 to index
  %30 = tensor.empty() : tensor<64x64xf32>
  %31:3 = scf.for %arg15 = %c0_i32 to %23 step %c1_i32 iter_args(%arg16 = %30, %arg17 = %27, %arg18 = %c0) -> (tensor<64x64xf32>, index, index)  : i32 {
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%arg17], sizes: [64, 64], strides: [%26, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
    %49 = arith.addi %arg18, %29 : index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%49], sizes: [64, 64], strides: [%28, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
    %50 = arith.muli %arg15, %c64_i32 : i32
    %51 = arith.subi %arg8, %50 : i32
    %alloc = memref.alloc() : memref<64x64xf32>
    %52 = arith.addi %25, %c64 : index
    %53 = arith.index_cast %arg6 : i32 to index
    %54 = arith.maxsi %25, %53 : index
    %55 = arith.minsi %52, %54 : index
    %56 = arith.subi %55, %25 : index
    %57 = arith.index_cast %51 : i32 to index
    %58 = arith.maxsi %57, %c0 : index
    %59 = arith.minsi %58, %c64 : index
    %60 = arith.minsi %56, %c64 : index
    %61 = arith.maxsi %60, %c0 : index
    %62 = arith.minsi %59, %c64 : index
    %63 = arith.maxsi %62, %c0 : index
    %64 = arith.cmpi slt, %61, %c64 : index
    %65 = arith.cmpi slt, %63, %c64 : index
    %66 = arith.ori %64, %65 : i1
    %subview_2 = memref.subview %reinterpret_cast_0[0, 0] [%61, %63] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    %subview_3 = memref.subview %alloc[0, 0] [%61, %63] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    hivm.hir.load ins(%subview_2 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_3 : memref<?x?xf32, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %66 : i1 may_implicit_transpose_with_last_axis = false
    %67 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
    %alloc_4 = memref.alloc() : memref<64x64xf32>
    %68 = arith.addi %29, %c64 : index
    %69 = arith.index_cast %arg7 : i32 to index
    %70 = arith.maxsi %29, %69 : index
    %71 = arith.minsi %68, %70 : index
    %72 = arith.subi %71, %29 : index
    %73 = arith.minsi %72, %c64 : index
    %74 = arith.maxsi %73, %c0 : index
    %75 = arith.cmpi slt, %74, %c64 : index
    %76 = arith.ori %65, %75 : i1
    %subview_5 = memref.subview %reinterpret_cast_1[0, 0] [%63, %74] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    %subview_6 = memref.subview %alloc_4[0, 0] [%63, %74] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    hivm.hir.load ins(%subview_5 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_6 : memref<?x?xf32, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %76 : i1 may_implicit_transpose_with_last_axis = false
    %77 = bufferization.to_tensor %alloc_4 restrict writable : memref<64x64xf32>
    %78 = arith.cmpi eq, %arg15, %c0_i32 : i32
    %79 = hivm.hir.mmadL1 ins(%67, %77, %78, %c0, %c0, %c0 : tensor<64x64xf32>, tensor<64x64xf32>, i1, index, index, index) outs(%arg16 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %80 = arith.addi %arg17, %c64 : index
    %81 = arith.index_cast %24 : i32 to index
    %82 = arith.addi %arg18, %81 : index
    scf.yield %79, %80, %82 : tensor<64x64xf32>, index, index
  }
  %32 = arith.index_cast %arg11 : i32 to index
  %33 = arith.muli %25, %32 : index
  %34 = arith.addi %33, %29 : index
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%34], sizes: [64, 64], strides: [%32, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
  %35 = arith.addi %25, %c64 : index
  %36 = arith.index_cast %arg6 : i32 to index
  %37 = arith.maxsi %25, %36 : index
  %38 = arith.minsi %35, %37 : index
  %39 = arith.subi %38, %25 : index
  %40 = arith.addi %29, %c64 : index
  %41 = arith.index_cast %arg7 : i32 to index
  %42 = arith.maxsi %29, %41 : index
  %43 = arith.minsi %40, %42 : index
  %44 = arith.subi %43, %29 : index
  %45 = arith.minsi %39, %c64 : index
  %46 = arith.maxsi %45, %c0 : index
  %47 = arith.minsi %44, %c64 : index
  %48 = arith.maxsi %47, %c0 : index
  %extracted_slice = tensor.extract_slice %31#0[0, 0] [%46, %48] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
  %subview = memref.subview %reinterpret_cast[0, 0] [%46, %48] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
  hivm.hir.store ins(%extracted_slice : tensor<?x?xf32>) outs(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>>)
  return
}

