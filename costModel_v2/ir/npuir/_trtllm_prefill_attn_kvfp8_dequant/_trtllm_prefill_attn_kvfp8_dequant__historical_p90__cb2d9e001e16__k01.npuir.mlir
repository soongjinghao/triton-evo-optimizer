// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_trtllm_prefill_attn_kvfp8_dequant(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c32768_i64 = arith.constant 32768 : i64
  %c22528_i64 = arith.constant 22528 : i64
  %c18432_i64 = arith.constant 18432 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c30720_i64 = arith.constant 30720 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg11, %arg10 : i32
  %5 = arith.extsi %arg5 : i32 to i64
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg12 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg12 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64_1 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_0, %c1_i64_1 : i64
    %13 = hivm.hir.pointer_cast(%c22528_i64, %c32768_i64) : memref<2048xbf16, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2048xbf16, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c0_i64, %c30720_i64) : memref<2048xi8, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<2048xi8, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %15 = arith.divsi %arg12, %arg11 : i32
    %16 = arith.remsi %15, %arg10 : i32
    %17 = arith.divsi %arg12, %4 : i32
    %18 = arith.remsi %17, %arg9 : i32
    %19 = arith.extsi %18 : i32 to i64
    %20 = arith.extsi %16 : i32 to i64
    %21 = arith.muli %19, %5 : i64
    %22 = arith.addi %21, %20 : i64
    %23 = arith.index_cast %22 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%23], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %24 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %25 = arith.extsi %24 : i32 to i64
    %26 = arith.cmpi sle, %25, %c0_i64 : i64
    scf.if %26 {
    } else {
      %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %27 = memref.load %reinterpret_cast_2[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %28 = memref.load %reinterpret_cast_3[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %29 = arith.muli %25, %c2048_i64 : i64
      %30 = arith.index_cast %29 : i64 to index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [2048], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<2048xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %12]
      hivm.hir.load ins(%reinterpret_cast_4 : memref<2048xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%14 : memref<2048xi8, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %31 = hivm.hir.pointer_cast(%c2048_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%27 : f32) outs(%31 : memref<2048xf32, #hivm.address_space<ub>>)
      %32 = hivm.hir.pointer_cast(%c10240_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%28 : f32) outs(%32 : memref<2048xf32, #hivm.address_space<ub>>)
      %subview = memref.subview %31[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_5 = memref.subview %32[0] [1024] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%subview : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<1024xf32, strided<[1]>, #hivm.address_space<ub>>)
      %33 = hivm.hir.pointer_cast(%c18432_i64) : memref<2048xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%14 : memref<2048xi8, #hivm.address_space<ub>>) outs(%33 : memref<2048xf16, #hivm.address_space<ub>>) cast = <cast_unsigned>
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %12]
      %34 = hivm.hir.pointer_cast(%c22528_i64, %c32768_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %12]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%33 : memref<2048xf16, #hivm.address_space<ub>>) outs(%34 : memref<2048xf32, #hivm.address_space<ub>>)
      %35 = hivm.hir.pointer_cast(%c22528_i64, %c32768_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%34, %32 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%35 : memref<2048xf32, #hivm.address_space<ub>>)
      %36 = arith.addi %22, %c1_i64 : i64
      %37 = arith.muli %36, %c2048_i64 : i64
      %38 = arith.index_cast %37 : i64 to index
      %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%38], sizes: [2048], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<2048xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%35 : memref<2048xf32, #hivm.address_space<ub>>) outs(%13 : memref<2048xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%13 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<2048xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %12]
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
