// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @reshape_and_cache_kernel_flash(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i64 {tt.divisibility = 16 : i32}, %arg11: i64 {tt.divisibility = 16 : i32}, %arg12: i64 {tt.divisibility = 16 : i32}, %arg13: i64 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c3072_i64 = arith.constant 3072 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %c512_i32 = arith.constant 512 : i32
  %c16_i64 = arith.constant 16 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg17, %4 : i32
    %6 = arith.remsi %5, %arg14 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%7], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %8 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %9 = arith.cmpi slt, %8, %c0_i64 : i64
    scf.if %9 {
    } else {
      %10 = arith.divsi %8, %c16_i64 : i64
      %11 = arith.remsi %8, %c16_i64 : i64
      %12 = arith.extsi %6 : i32 to i64
      %13 = arith.muli %12, %arg10 : i64
      %14 = arith.muli %12, %arg11 : i64
      %15 = arith.muli %10, %arg12 : i64
      %16 = arith.muli %11, %arg13 : i64
      %17 = arith.addi %15, %16 : i64
      %18 = arith.index_cast %13 : i64 to index
      %19 = arith.index_cast %14 : i64 to index
      %20 = arith.index_cast %17 : i64 to index
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
      scf.for %arg18 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %21 = arith.index_cast %arg18 : i32 to index
        %22 = arith.index_cast %c0_i32 : i32 to index
        %23 = arith.index_cast %c4_i32 : i32 to index
        %24 = arith.index_cast %c1_i32 : i32 to index
        %25 = arith.index_cast %arg17 : i32 to index
        %26 = arith.index_cast %3 : i32 to index
        %27 = arith.index_cast %1 : i32 to index
        %28 = arith.index_cast %c48_i32 : i32 to index
        %29 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%21, %22, %23, %24, %25, %26, %28]
        %30 = arith.index_cast %29 : index to i1
        %c2_i64 = arith.constant 2 : i64
        %c3_i64 = arith.constant 3 : i64
        %31 = arith.select %30, %c2_i64, %c3_i64 : i64
        %32 = arith.index_cast %29 : index to i1
        %c0_i64_0 = arith.constant 0 : i64
        %c1_i64 = arith.constant 1 : i64
        %33 = arith.select %32, %c0_i64_0, %c1_i64 : i64
        %34 = hivm.hir.pointer_cast(%c1024_i64, %c3072_i64) : memref<512xf16, #hivm.address_space<ub>>
        annotation.mark %34 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
        %35 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<512xf16, #hivm.address_space<ub>>
        annotation.mark %35 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
        %36 = arith.muli %arg18, %c512_i32 : i32
        %37 = arith.index_cast %36 : i32 to index
        %38 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%18, %37]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%38], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %39 = affine.max affine_map<()[s0] -> (2048, s0)>()[%37]
        %40 = affine.min affine_map<()[s0, s1] -> (s1 + 512, s0)>()[%39, %37]
        %41 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%40, %37]
        %42 = arith.cmpi slt, %41, %c512 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%41] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %35[0] [%41] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %42 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%35 : memref<512xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %33]
        hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %43 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%19, %37]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %reinterpret_cast_3[0] [%41] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %34[0] [%41] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %42 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%34 : memref<512xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %31]
        hivm.hir.load ins(%subview_4 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %44 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%20, %37]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %reinterpret_cast_6[0] [%41] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %33]
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%44], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%41] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %31]
      }
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
