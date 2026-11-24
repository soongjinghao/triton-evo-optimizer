// -----// IR Dump After EnableStrideAlign (hivm-enable-stride-align) //----- //
func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %c63_i32 = arith.constant 63 : i32
  %c8_i32 = arith.constant 8 : i32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1_i32 = arith.constant 1 : i32
  %c24_i32 = arith.constant 24 : i32
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg14, %arg13 : i32
  %5 = arith.addi %arg6, %c63_i32 : i32
  %6 = arith.divsi %5, %c64_i32 : i32
  %7 = arith.addi %arg7, %c63_i32 : i32
  %8 = arith.divsi %7, %c64_i32 : i32
  %9 = arith.muli %8, %c8_i32 : i32
  %10 = arith.addi %arg8, %c63_i32 : i32
  %11 = arith.divsi %10, %c64_i32 : i32
  %12 = arith.muli %arg10, %c64_i32 : i32
  %13 = arith.index_cast %arg9 : i32 to index
  %14 = arith.index_cast %arg10 : i32 to index
  %15 = arith.index_cast %arg6 : i32 to index
  %16 = arith.index_cast %arg7 : i32 to index
  %17 = arith.index_cast %12 : i32 to index
  %18 = arith.index_cast %arg11 : i32 to index
  scf.for %arg15 = %3 to %1 step %c24_i32  : i32 {
    hivm.hir.set_mask_norm
    %19 = arith.divsi %arg15, %4 : i32
    %20 = arith.remsi %19, %arg12 : i32
    %21 = arith.divsi %20, %9 : i32
    %22 = arith.muli %21, %c8_i32 : i32
    %23 = arith.subi %6, %22 : i32
    %24 = arith.minsi %23, %c8_i32 : i32
    %25 = arith.remsi %20, %24 : i32
    %26 = arith.addi %22, %25 : i32
    %27 = arith.remsi %20, %9 : i32
    %28 = arith.divsi %27, %24 : i32
    %29 = arith.muli %26, %c64_i32 : i32
    %30 = arith.muli %28, %c64_i32 : i32
    %31 = arith.index_cast %29 : i32 to index
    %32 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%31, %13]
    %33 = arith.index_cast %30 : i32 to index
    %34 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%31, %15]
    %35 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%34, %31]
    %36 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%35, %31]
    %37 = affine.max affine_map<()[s0] -> (0, s0)>()[%36]
    %38 = arith.cmpi slt, %37, %c64 : index
    %39 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%33, %16]
    %40 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%39, %33]
    %41 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%40, %33]
    %42 = affine.max affine_map<()[s0] -> (0, s0)>()[%41]
    %43 = arith.cmpi slt, %42, %c64 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32, #hivm.address_space<cc>>
    %44:2 = scf.for %arg16 = %c0_i32 to %11 step %c1_i32 iter_args(%arg17 = %32, %arg18 = %c0) -> (index, index)  : i32 {
      %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%arg17], sizes: [64, 64], strides: [%13, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %46 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %33]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 64], strides: [%14, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %47 = arith.muli %arg16, %c64_i32 : i32
      %48 = arith.subi %arg8, %47 : i32
      %alloc_3 = memref.alloc() : memref<64x64xf32, #hivm.address_space<cbuf>>
      %49 = arith.index_cast %48 : i32 to index
      %50 = affine.max affine_map<()[s0] -> (0, s0)>()[%49]
      %51 = affine.min affine_map<()[s0] -> (64, s0)>()[%50]
      %52 = affine.max affine_map<()[s0] -> (0, s0)>()[%51]
      %53 = arith.cmpi slt, %52, %c64 : index
      %54 = arith.ori %38, %53 : i1
      %subview_4 = memref.subview %reinterpret_cast_1[0, 0] [%37, %52] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %alloc_3[0, 0] [%37, %52] [1, 1] : memref<64x64xf32, #hivm.address_space<cbuf>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.load ins(%subview_4 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %54 : i1 may_implicit_transpose_with_last_axis = false core_type = <CUBE>
      %alloc_6 = memref.alloc() : memref<64x64xf32, #hivm.address_space<cbuf>>
      %55 = arith.ori %53, %43 : i1
      %subview_7 = memref.subview %reinterpret_cast_2[0, 0] [%52, %42] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %alloc_6[0, 0] [%52, %42] [1, 1] : memref<64x64xf32, #hivm.address_space<cbuf>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>
      hivm.hir.load ins(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<cbuf>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %55 : i1 may_implicit_transpose_with_last_axis = false core_type = <CUBE>
      %56 = arith.cmpi eq, %arg16, %c0_i32 : i32
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%alloc_3, %alloc_6, %56, %37, %52, %42 : memref<64x64xf32, #hivm.address_space<cbuf>>, memref<64x64xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc : memref<64x64xf32, #hivm.address_space<cc>>)
      %57 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%arg17]
      %58 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg18, %17]
      scf.yield %57, %58 : index, index
    } {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly}
    %45 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%33, %31, %18]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%45], sizes: [64, 64], strides: [%18, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview = memref.subview %alloc[0, 0] [%37, %42] [1, 1] : memref<64x64xf32, #hivm.address_space<cc>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<cc>>
    %subview_0 = memref.subview %reinterpret_cast[0, 0] [%37, %42] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%subview : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<cc>>) outs(%subview_0 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  return
}

