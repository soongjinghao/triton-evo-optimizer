// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_trtllm_prefill_attn_kvfp8_dequant(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c32768_i64 = arith.constant 32768 : i64
  %c22528_i64 = arith.constant 22528 : i64
  %c18432_i64 = arith.constant 18432 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c30720_i64 = arith.constant 30720 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64_1 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64_1 : i64
    %12 = hivm.hir.pointer_cast(%c22528_i64, %c32768_i64) : memref<2048xbf16, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<2048xbf16, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c0_i64, %c30720_i64) : memref<2048xi8, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2048xi8, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %14 = arith.divsi %arg11, %arg10 : i32
    %15 = arith.remsi %14, %arg9 : i32
    %16 = arith.divsi %arg11, %4 : i32
    %17 = arith.remsi %16, %arg8 : i32
    %18 = arith.extsi %17 : i32 to i64
    %19 = arith.extsi %15 : i32 to i64
    %20 = arith.muli %18, %c2_i64 : i64
    %21 = arith.addi %20, %19 : i64
    %22 = arith.index_cast %21 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%22], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %23 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %24 = arith.extsi %23 : i32 to i64
    %25 = arith.cmpi sle, %24, %c0_i64 : i64
    scf.if %25 {
    } else {
      %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %26 = memref.load %reinterpret_cast_2[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %27 = memref.load %reinterpret_cast_3[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %28 = arith.muli %24, %c2048_i64 : i64
      %29 = arith.index_cast %28 : i64 to index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [2048], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<2048xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %11]
      hivm.hir.load ins(%reinterpret_cast_4 : memref<2048xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%13 : memref<2048xi8, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %30 = hivm.hir.pointer_cast(%c2048_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%26 : f32) outs(%30 : memref<2048xf32, #hivm.address_space<ub>>)
      %31 = hivm.hir.pointer_cast(%c10240_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%27 : f32) outs(%31 : memref<2048xf32, #hivm.address_space<ub>>)
      %subview = memref.subview %30[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_5 = memref.subview %31[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%subview : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>)
      %32 = hivm.hir.pointer_cast(%c18432_i64) : memref<2048xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%13 : memref<2048xi8, #hivm.address_space<ub>>) outs(%32 : memref<2048xf16, #hivm.address_space<ub>>) cast = <cast_unsigned>
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %11]
      %33 = hivm.hir.pointer_cast(%c22528_i64, %c32768_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %11]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%32 : memref<2048xf16, #hivm.address_space<ub>>) outs(%33 : memref<2048xf32, #hivm.address_space<ub>>)
      %34 = hivm.hir.pointer_cast(%c22528_i64, %c32768_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%33, %31 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%34 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%34 : memref<2048xf32, #hivm.address_space<ub>>) outs(%12 : memref<2048xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %35 = arith.addi %21, %c1_i64 : i64
      %36 = arith.muli %35, %c2048_i64 : i64
      %37 = arith.index_cast %36 : i64 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [2048], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<2048xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%12 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<2048xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %11]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
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
