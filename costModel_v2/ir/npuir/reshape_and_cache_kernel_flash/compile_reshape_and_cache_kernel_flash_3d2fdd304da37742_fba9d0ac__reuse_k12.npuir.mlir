// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @reshape_and_cache_kernel_flash(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i64 {tt.divisibility = 16 : i32}, %arg11: i64 {tt.divisibility = 16 : i32}, %arg12: i64 {tt.divisibility = 16 : i32}, %arg13: i64 {tt.divisibility = 16 : i32}, %arg14: i64 {tt.divisibility = 16 : i32}, %arg15: i64, %arg16: i64 {tt.divisibility = 16 : i32}, %arg17: i64 {tt.divisibility = 16 : i32}, %arg18: i64 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32, %arg21: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<22xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c3072_i64 = arith.constant 3072 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c512_i32 = arith.constant 512 : i32
  %c512 = arith.constant 512 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg19, %arg20 : i32
  %1 = arith.muli %0, %arg21 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.index_cast %arg16 : i64 to index
  %5 = arith.index_cast %arg17 : i64 to index
  %6 = arith.index_cast %arg18 : i64 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
  scf.for %arg22 = %3 to %1 step %c48_i32  : i32 {
    %7 = arith.index_cast %arg22 : i32 to index
    %8 = arith.index_cast %3 : i32 to index
    %9 = arith.index_cast %1 : i32 to index
    %10 = arith.index_cast %c48_i32 : i32 to index
    %11 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%7, %8, %10]
    %12 = arith.index_cast %11 : index to i1
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %13 = arith.select %12, %c2_i64, %c3_i64 : i64
    %14 = arith.index_cast %11 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %15 = arith.select %14, %c0_i64_0, %c1_i64 : i64
    %16 = hivm.hir.pointer_cast(%c1024_i64, %c3072_i64) : memref<512xf16, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<512xf16, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %18 = arith.divsi %arg22, %arg21 : i32
    %19 = arith.remsi %18, %arg20 : i32
    %20 = arith.muli %19, %c512_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%4, %21]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%22], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %23 = affine.max affine_map<()[s0] -> (512, s0)>()[%21]
    %24 = affine.min affine_map<()[s0, s1] -> (s1 + 512, s0)>()[%23, %21]
    %25 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%24, %21]
    %26 = arith.cmpi slt, %25, %c512 : index
    %subview = memref.subview %reinterpret_cast[0] [%25] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_1 = memref.subview %17[0] [%25] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
    scf.if %26 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f16) outs(%17 : memref<512xf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
    hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
    %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%5, %21]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%27], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %reinterpret_cast_2[0] [%25] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %16[0] [%25] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
    scf.if %26 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f16) outs(%16 : memref<512xf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
    hivm.hir.load ins(%subview_3 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %28 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%6, %21]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%28], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%25] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_1 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%28], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_8 = memref.subview %reinterpret_cast_7[0] [%25] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_4 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
