// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_pack_seq_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c8192_i64 = arith.constant 8192 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<32x32xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %5 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<1024xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg10 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_0, %c1_i64 : i64
    %13 = hivm.hir.pointer_cast(%c4096_i64, %c8192_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %14 = arith.remsi %arg10, %arg9 : i32
    %15 = arith.divsi %arg10, %arg9 : i32
    %16 = arith.remsi %15, %arg8 : i32
    %17 = arith.divsi %arg10, %4 : i32
    %18 = arith.remsi %17, %arg7 : i32
    %19 = arith.muli %16, %c32_i32 : i32
    %20 = arith.muli %14, %c32_i32 : i32
    %21 = arith.index_cast %18 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%21], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %22 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%21], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %23 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %24 = arith.index_cast %22 : i32 to index
    %25 = arith.index_cast %19 : i32 to index
    %26 = arith.index_cast %20 : i32 to index
    %27 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 4 + s2 * 4)>()[%26, %24, %25]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32], strides: [4, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
    %28 = arith.index_cast %23 : i32 to index
    %29 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%25, %28]
    %30 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%29, %25]
    %31 = affine.max affine_map<()[s0] -> (4, s0)>()[%25]
    %32 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%31, %25]
    %33 = affine.min affine_map<()[s0, s1, s2] -> (-s1 + s2, s0 - s1)>()[%30, %25, %32]
    %34 = affine.max affine_map<()[s0] -> (0, s0)>()[%33]
    %35 = affine.max affine_map<()[s0] -> (4, s0)>()[%26]
    %36 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%35, %26]
    %37 = affine.min affine_map<()[s0] -> (32, s0)>()[%34]
    %38 = affine.max affine_map<()[s0] -> (0, s0)>()[%37]
    %39 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%36, %26]
    %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
    %41 = arith.cmpi slt, %38, %c32 : index
    %42 = arith.cmpi slt, %40, %c32 : index
    %43 = arith.ori %41, %42 : i1
    %subview = memref.subview %reinterpret_cast_2[0, 0] [%38, %40] [1, 1] : memref<32x32xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %13[0, 0] [%38, %40] [1, 1] : memref<32x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    scf.if %43 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_8 = memref.collapse_shape %13 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_8 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %12]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %44 = arith.muli %18, %c4_i32 : i32
    %45 = arith.index_cast %44 : i32 to index
    %46 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 4 + s2 * 4)>()[%26, %45, %25]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [32, 32], strides: [4, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
    %47 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%32, %25]
    %48 = affine.max affine_map<()[s0] -> (0, s0)>()[%47]
    %subview_5 = memref.subview %5[0, 0] [%48, %40] [1, 1] : memref<32x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [%48, %40] [1, 1] : memref<32x32xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_5 : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>)
    %subview_7 = memref.subview %reinterpret_cast_4[0, 0] [%38, %40] [1, 1] : memref<32x32xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_3 : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %12]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
