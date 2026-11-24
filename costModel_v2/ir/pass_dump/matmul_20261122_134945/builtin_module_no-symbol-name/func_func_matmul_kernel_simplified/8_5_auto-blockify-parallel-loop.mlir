// -----// IR Dump After AutoBlockifyParallelLoop (auto-blockify-parallel-loop) //----- //
func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "mix", parallel_mode = "simd"} {
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %c24_i32 = arith.constant 24 : i32
  %c1_i32 = arith.constant 1 : i32
  %3 = arith.trunci %2 : i64 to i32
  scf.for %arg15 = %3 to %1 step %c24_i32  : i32 {
    %4 = arith.muli %arg15, %c1_i32 : i32
    %5 = arith.extsi %4 : i32 to i64
    %c1_i32_0 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c63_i32 = arith.constant 63 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    hivm.hir.set_mask_norm
    %6 = arith.trunci %5 : i64 to i32
    %7 = arith.muli %arg14, %arg13 : i32
    %8 = arith.divsi %6, %7 : i32
    %9 = arith.remsi %8, %arg12 : i32
    %10 = arith.addi %arg6, %c63_i32 : i32
    %11 = arith.divsi %10, %c64_i32 : i32
    %12 = arith.addi %arg7, %c63_i32 : i32
    %13 = arith.divsi %12, %c64_i32 : i32
    %14 = arith.muli %13, %c8_i32 : i32
    %15 = arith.divsi %9, %14 : i32
    %16 = arith.muli %15, %c8_i32 : i32
    %17 = arith.subi %11, %16 : i32
    %18 = arith.minsi %17, %c8_i32 : i32
    %19 = arith.remsi %9, %18 : i32
    %20 = arith.addi %16, %19 : i32
    %21 = arith.remsi %9, %14 : i32
    %22 = arith.divsi %21, %18 : i32
    %23 = arith.muli %20, %c64_i32 : i32
    %24 = arith.muli %22, %c64_i32 : i32
    %25 = arith.addi %arg8, %c63_i32 : i32
    %26 = arith.divsi %25, %c64_i32 : i32
    %27 = arith.muli %arg10, %c64_i32 : i32
    %28 = arith.index_cast %23 : i32 to index
    %29 = arith.index_cast %arg9 : i32 to index
    %30 = arith.muli %28, %29 : index
    %31 = arith.index_cast %arg10 : i32 to index
    %32 = arith.index_cast %24 : i32 to index
    %33 = tensor.empty() : tensor<64x64xf32>
    %34:3 = scf.for %arg16 = %c0_i32 to %26 step %c1_i32_0 iter_args(%arg17 = %33, %arg18 = %30, %arg19 = %c0) -> (tensor<64x64xf32>, index, index)  : i32 {
      %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%arg18], sizes: [64, 64], strides: [%29, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %52 = arith.addi %arg19, %32 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%52], sizes: [64, 64], strides: [%31, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %53 = arith.muli %arg16, %c64_i32 : i32
      %54 = arith.subi %arg8, %53 : i32
      %alloc = memref.alloc() : memref<64x64xf32>
      %55 = arith.addi %28, %c64 : index
      %56 = arith.index_cast %arg6 : i32 to index
      %57 = arith.maxsi %28, %56 : index
      %58 = arith.minsi %55, %57 : index
      %59 = arith.subi %58, %28 : index
      %60 = arith.index_cast %54 : i32 to index
      %61 = arith.maxsi %60, %c0 : index
      %62 = arith.minsi %61, %c64 : index
      %63 = arith.minsi %59, %c64 : index
      %64 = arith.maxsi %63, %c0 : index
      %65 = arith.minsi %62, %c64 : index
      %66 = arith.maxsi %65, %c0 : index
      %67 = arith.cmpi slt, %64, %c64 : index
      %68 = arith.cmpi slt, %66, %c64 : index
      %69 = arith.ori %67, %68 : i1
      %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%64, %66] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_4 = memref.subview %alloc[0, 0] [%64, %66] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      hivm.hir.load ins(%subview_3 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_4 : memref<?x?xf32, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %69 : i1 may_implicit_transpose_with_last_axis = false core_type = <CUBE>
      %70 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
      %alloc_5 = memref.alloc() : memref<64x64xf32>
      %71 = arith.addi %32, %c64 : index
      %72 = arith.index_cast %arg7 : i32 to index
      %73 = arith.maxsi %32, %72 : index
      %74 = arith.minsi %71, %73 : index
      %75 = arith.subi %74, %32 : index
      %76 = arith.minsi %75, %c64 : index
      %77 = arith.maxsi %76, %c0 : index
      %78 = arith.cmpi slt, %77, %c64 : index
      %79 = arith.ori %68, %78 : i1
      %subview_6 = memref.subview %reinterpret_cast_2[0, 0] [%66, %77] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_7 = memref.subview %alloc_5[0, 0] [%66, %77] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      hivm.hir.load ins(%subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_7 : memref<?x?xf32, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %79 : i1 may_implicit_transpose_with_last_axis = false core_type = <CUBE>
      %80 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x64xf32>
      %81 = arith.cmpi eq, %arg16, %c0_i32 : i32
      %82 = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%70, %80, %81, %64, %66, %77 : tensor<64x64xf32>, tensor<64x64xf32>, i1, index, index, index) outs(%arg17 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %83 = arith.addi %arg18, %c64 : index
      %84 = arith.index_cast %27 : i32 to index
      %85 = arith.addi %arg19, %84 : index
      scf.yield %82, %83, %85 : tensor<64x64xf32>, index, index
    } {fixpipe_for_mmad_result_already_inserted = true}
    %35 = arith.index_cast %arg11 : i32 to index
    %36 = arith.muli %28, %35 : index
    %37 = arith.addi %36, %32 : index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [64, 64], strides: [%35, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
    %38 = arith.addi %28, %c64 : index
    %39 = arith.index_cast %arg6 : i32 to index
    %40 = arith.maxsi %28, %39 : index
    %41 = arith.minsi %38, %40 : index
    %42 = arith.subi %41, %28 : index
    %43 = arith.addi %32, %c64 : index
    %44 = arith.index_cast %arg7 : i32 to index
    %45 = arith.maxsi %32, %44 : index
    %46 = arith.minsi %43, %45 : index
    %47 = arith.subi %46, %32 : index
    %48 = arith.minsi %42, %c64 : index
    %49 = arith.maxsi %48, %c0 : index
    %50 = arith.minsi %47, %c64 : index
    %51 = arith.maxsi %50, %c0 : index
    %extracted_slice = tensor.extract_slice %34#0[0, 0] [%49, %51] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%49, %51] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%extracted_slice : tensor<?x?xf32>) outs(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>>)
  } {autoblockify.subloop}
  return
}

