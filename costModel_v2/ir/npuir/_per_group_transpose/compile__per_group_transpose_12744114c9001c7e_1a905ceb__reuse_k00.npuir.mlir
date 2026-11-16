// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_per_group_transpose(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c24576_i64 = arith.constant 24576 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64 = arith.constant 0 : i64
  %c32_i32 = arith.constant 32 : i32
  %c64_i32 = arith.constant 64 : i32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = arith.index_cast %arg7 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg11 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %12 = arith.select %11, %c1_i64, %c2_i64 : i64
    %13 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64_1 = arith.constant 1 : i64
    %14 = arith.select %13, %c0_i64_0, %c1_i64_1 : i64
    %15 = hivm.hir.pointer_cast(%c8192_i64, %c24576_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<32x64xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<64x32xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %17 = arith.remsi %arg11, %arg10 : i32
    %18 = arith.divsi %arg11, %arg10 : i32
    %19 = arith.remsi %18, %arg9 : i32
    %20 = arith.divsi %arg11, %4 : i32
    %21 = arith.remsi %20, %arg8 : i32
    %22 = arith.index_cast %21 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%22], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %23 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %24 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %25 = arith.index_cast %24 : i32 to index
    %26 = arith.muli %23, %arg7 : i32
    %27 = arith.index_cast %26 : i32 to index
    %28 = arith.muli %19, %c64_i32 : i32
    %29 = arith.index_cast %28 : i32 to index
    %30 = arith.muli %17, %c32_i32 : i32
    %31 = arith.index_cast %30 : i32 to index
    %32 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>()[%31, %27, %29, %5]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%32], sizes: [64, 32], strides: [%5, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %33 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%29, %25]
    %34 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%33, %29]
    %35 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%31, %5]
    %36 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%35, %31]
    %37 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%34, %29]
    %38 = affine.max affine_map<()[s0] -> (0, s0)>()[%37]
    %39 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%36, %31]
    %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
    %41 = arith.cmpi slt, %38, %c64 : index
    %42 = arith.cmpi slt, %40, %c32 : index
    %43 = arith.ori %41, %42 : i1
    %subview = memref.subview %reinterpret_cast_3[0, 0] [%38, %40] [1, 1] : memref<64x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %16[0, 0] [%38, %40] [1, 1] : memref<64x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    scf.if %43 {
      %collapse_shape = memref.collapse_shape %16 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %12]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %44 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>()[%27, %29, %31, %25]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [32, 64], strides: [%25, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %14]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vtranspose ins(%16 : memref<64x32xf32, #hivm.address_space<ub>>) outs(%15 : memref<32x64xf32, #hivm.address_space<ub>>) permutation = [1, 0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %12]
    %subview_6 = memref.subview %15[0, 0] [%40, %38] [1, 1] : memref<32x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%40, %38] [1, 1] : memref<32x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_6 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %14]
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
