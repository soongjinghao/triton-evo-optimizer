// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_per_group_transpose(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c6144_i64 = arith.constant 6144 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.muli %arg8, %c32_i32 : i32
  %6 = arith.index_cast %arg6 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %7 = arith.remsi %arg10, %arg9 : i32
    %8 = arith.divsi %arg10, %arg9 : i32
    %9 = arith.remsi %8, %arg8 : i32
    %10 = arith.divsi %arg10, %4 : i32
    %11 = arith.remsi %10, %arg7 : i32
    %12 = arith.index_cast %11 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %13 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %14 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%12]
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %15 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %16 = arith.subi %15, %13 : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = arith.muli %13, %arg6 : i32
    %19 = arith.index_cast %18 : i32 to index
    %20 = arith.muli %7, %c16_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = arith.muli %9, %c32_i32 : i32
    %23 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%21, %6]
    %24 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%23, %21]
    %25 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%24, %21]
    %26 = affine.max affine_map<()[s0] -> (0, s0)>()[%25]
    %27 = arith.cmpi slt, %26, %c16 : index
    scf.for %arg11 = %c0_i32 to %16 step %5  : i32 {
      %28 = arith.index_cast %arg11 : i32 to index
      %29 = arith.index_cast %c0_i32 : i32 to index
      %30 = arith.index_cast %16 : i32 to index
      %31 = arith.index_cast %5 : i32 to index
      %32 = arith.index_cast %arg10 : i32 to index
      %33 = arith.index_cast %3 : i32 to index
      %34 = arith.index_cast %1 : i32 to index
      %35 = arith.index_cast %c48_i32 : i32 to index
      %36 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%28, %29, %30, %31, %32, %33, %35]
      %37 = arith.index_cast %36 : index to i1
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %38 = arith.select %37, %c1_i64, %c2_i64 : i64
      %39 = arith.index_cast %36 : index to i1
      %c0_i64_1 = arith.constant 0 : i64
      %c1_i64_2 = arith.constant 1 : i64
      %40 = arith.select %39, %c0_i64_1, %c1_i64_2 : i64
      %41 = hivm.hir.pointer_cast(%c2048_i64, %c6144_i64) : memref<16x32xf32, #hivm.address_space<ub>>
      annotation.mark %41 {hivm.multi_buffer = 2 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
      %42 = hivm.hir.pointer_cast(%c0_i64, %c4096_i64) : memref<32x16xf32, #hivm.address_space<ub>>
      annotation.mark %42 {hivm.multi_buffer = 2 : i32} : memref<32x16xf32, #hivm.address_space<ub>>
      %43 = arith.addi %arg11, %22 : i32
      %44 = arith.index_cast %43 : i32 to index
      %45 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>()[%21, %19, %44, %6]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%45], sizes: [32, 16], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %46 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%44, %17]
      %47 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%46, %44]
      %48 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%47, %44]
      %49 = affine.max affine_map<()[s0] -> (0, s0)>()[%48]
      %50 = arith.cmpi slt, %49, %c32 : index
      %51 = arith.ori %50, %27 : i1
      %subview = memref.subview %reinterpret_cast_3[0, 0] [%49, %26] [1, 1] : memref<32x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %42[0, 0] [%49, %26] [1, 1] : memref<32x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %51 {
        %collapse_shape = memref.collapse_shape %42 [[0, 1]] : memref<32x16xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %38]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %52 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>()[%19, %44, %21, %17]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%52], sizes: [16, 32], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %40]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vtranspose ins(%42 : memref<32x16xf32, #hivm.address_space<ub>>) outs(%41 : memref<16x32xf32, #hivm.address_space<ub>>) permutation = [1, 0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %38]
      %subview_6 = memref.subview %41[0, 0] [%26, %49] [1, 1] : memref<16x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%26, %49] [1, 1] : memref<16x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_6 : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %40]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
