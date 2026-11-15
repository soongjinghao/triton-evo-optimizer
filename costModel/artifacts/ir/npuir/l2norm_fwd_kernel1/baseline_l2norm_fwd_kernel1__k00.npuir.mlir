// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @l2norm_fwd_kernel1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1088_i64 = arith.constant 1088 : i64
  %c576_i64 = arith.constant 576 : i64
  %c544_i64 = arith.constant 544 : i64
  %c288_i64 = arith.constant 288 : i64
  %c832_i64 = arith.constant 832 : i64
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.index_cast %arg5 : i32 to index
  %6 = affine.max affine_map<()[s0] -> (0, s0)>()[%5]
  %7 = affine.min affine_map<()[s0] -> (64, s0)>()[%6]
  %8 = arith.cmpi slt, %7, %c64 : index
  %9 = hivm.hir.pointer_cast(%c0_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %arg6, %9[%c0] : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %10 = arith.index_cast %arg10 : i32 to index
    %11 = arith.index_cast %3 : i32 to index
    %12 = arith.index_cast %1 : i32 to index
    %13 = arith.index_cast %c48_i32 : i32 to index
    %14 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%10, %11, %13]
    %15 = arith.index_cast %14 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %16 = arith.select %15, %c0_i64_1, %c1_i64 : i64
    %17 = arith.index_cast %14 : index to i1
    %c1_i64_2 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %18 = arith.select %17, %c1_i64_2, %c2_i64 : i64
    %19 = hivm.hir.pointer_cast(%c576_i64, %c1088_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c32_i64, %c832_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %21 = arith.divsi %arg10, %4 : i32
    %22 = arith.remsi %21, %arg7 : i32
    %23 = arith.muli %22, %arg5 : i32
    %24 = arith.index_cast %23 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview = memref.subview %reinterpret_cast[0] [%7] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %20[0] [%7] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %8 {
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%20 : memref<64xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %18]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %25 = hivm.hir.pointer_cast(%c288_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%20, %20 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%25 : memref<64xf32, #hivm.address_space<ub>>)
    %26 = hivm.hir.pointer_cast(%c544_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%25 : memref<64xf32, #hivm.address_space<ub>>) outs(%26 : memref<1xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %27 = memref.load %26[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %28 = memref.load %9[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %29 = arith.addf %27, %28 : f32
    %30 = math.sqrt %29 : f32
    %31 = arith.divf %cst, %30 : f32
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %16]
    hivm.hir.vmul ins(%20, %31 : memref<64xf32, #hivm.address_space<ub>>, f32) outs(%19 : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %18]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%24], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %19[0] [%7] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %reinterpret_cast_4[0] [%7] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_5 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %16]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
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
