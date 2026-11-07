// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_per_group_transpose(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1536_i64 = arith.constant 1536 : i64
  %c512_i64 = arith.constant 512 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c8_i32 = arith.constant 8 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.remsi %3, %arg9 : i32
  %5 = arith.divsi %3, %arg9 : i32
  %6 = arith.remsi %5, %arg8 : i32
  %7 = arith.muli %arg9, %arg8 : i32
  %8 = arith.divsi %3, %7 : i32
  %9 = arith.remsi %8, %arg7 : i32
  %10 = arith.index_cast %9 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%10], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %11 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %12 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%10]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %13 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %14 = arith.subi %13, %11 : i32
  %15 = arith.index_cast %14 : i32 to index
  %16 = arith.muli %11, %arg6 : i32
  %17 = arith.index_cast %16 : i32 to index
  %18 = arith.muli %4, %c8_i32 : i32
  %19 = arith.index_cast %18 : i32 to index
  %20 = arith.muli %arg8, %c16_i32 : i32
  %21 = arith.muli %6, %c16_i32 : i32
  %22 = arith.index_cast %arg6 : i32 to index
  %23 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%19, %22]
  %24 = affine.min affine_map<()[s0, s1] -> (s1 + 8, s0)>()[%23, %19]
  %25 = affine.min affine_map<()[s0, s1] -> (8, s0 - s1)>()[%24, %19]
  %26 = arith.cmpi slt, %25, %c8 : index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg10 = %c0_i32 to %14 step %20  : i32 {
    %27 = arith.index_cast %arg10 : i32 to index
    %28 = arith.index_cast %c0_i32 : i32 to index
    %29 = arith.index_cast %14 : i32 to index
    %30 = arith.index_cast %20 : i32 to index
    %31 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%27, %28, %30]
    %32 = arith.index_cast %31 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %33 = arith.select %32, %c1_i64, %c2_i64 : i64
    %34 = arith.index_cast %31 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64_2 = arith.constant 1 : i64
    %35 = arith.select %34, %c0_i64_1, %c1_i64_2 : i64
    %36 = hivm.hir.pointer_cast(%c512_i64, %c1536_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    annotation.mark %36 {hivm.multi_buffer = 2 : i32} : memref<8x16xf32, #hivm.address_space<ub>>
    %37 = hivm.hir.pointer_cast(%c0_i64, %c1024_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    annotation.mark %37 {hivm.multi_buffer = 2 : i32} : memref<16x8xf32, #hivm.address_space<ub>>
    %38 = arith.addi %arg10, %21 : i32
    %39 = arith.index_cast %38 : i32 to index
    %40 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>()[%19, %17, %39, %22]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%40], sizes: [16, 8], strides: [%22, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %41 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%39, %15]
    %42 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%41, %39]
    %43 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%42, %39]
    %44 = arith.cmpi slt, %43, %c16 : index
    %45 = arith.ori %44, %26 : i1
    %subview = memref.subview %reinterpret_cast_3[0, 0] [%43, %25] [1, 1] : memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %37[0, 0] [%43, %25] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    scf.if %45 {
      %collapse_shape = memref.collapse_shape %37 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %33]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %46 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>()[%17, %39, %19, %15]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [8, 16], strides: [%15, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %35]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vtranspose ins(%37 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%36 : memref<8x16xf32, #hivm.address_space<ub>>) permutation = [1, 0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %33]
    %subview_6 = memref.subview %36[0, 0] [%25, %43] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%25, %43] [1, 1] : memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_6 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %35]
  }
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
