// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @reshape_and_cache_flash_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: f32, %arg15: f32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false]> : vector<19xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c12288_i64 = arith.constant 12288 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1024_i32 = arith.constant 1024 : i32
  %c512_i32 = arith.constant 512 : i32
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg16, %arg17 : i32
  %1 = arith.muli %0, %arg18 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg18, %arg17 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
  scf.for %arg19 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg19 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = arith.index_cast %9 : index to i1
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %13 = arith.select %12, %c2_i64, %c3_i64 : i64
    %14 = hivm.hir.pointer_cast(%c4096_i64, %c12288_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c0_i64, %c8192_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %16 = arith.divsi %arg19, %arg18 : i32
    %17 = arith.remsi %16, %arg17 : i32
    %18 = arith.divsi %arg19, %4 : i32
    %19 = arith.remsi %18, %arg16 : i32
    %20 = arith.index_cast %19 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%20], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %21 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %22 = arith.cmpi slt, %21, %c0_i32 : i32
    scf.if %22 {
    } else {
      %23 = arith.muli %21, %c512_i32 : i32
      %24 = arith.muli %17, %c1024_i32 : i32
      %25 = arith.muli %19, %arg9 : i32
      %26 = arith.muli %19, %arg10 : i32
      %27 = arith.index_cast %25 : i32 to index
      %28 = arith.index_cast %24 : i32 to index
      %29 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%27, %28]
      %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %30 = affine.max affine_map<()[s0] -> (512, s0)>()[%28]
      %31 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%30, %28]
      %32 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%31, %28]
      %33 = arith.cmpi slt, %32, %c1024 : index
      %subview = memref.subview %reinterpret_cast_1[0] [%32] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %15[0] [%32] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %33 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%15 : memref<1024xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
      %34 = arith.index_cast %26 : i32 to index
      %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%34, %28]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %reinterpret_cast_3[0] [%32] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %14[0] [%32] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %33 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%14 : memref<1024xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
      hivm.hir.load ins(%subview_4 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %36 = arith.addi %23, %24 : i32
      %37 = arith.index_cast %36 : i32 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %reinterpret_cast_6[0] [%32] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
      %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%37], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %reinterpret_cast_8[0] [%32] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%subview_5 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
    } {pipeline.veconly}
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
