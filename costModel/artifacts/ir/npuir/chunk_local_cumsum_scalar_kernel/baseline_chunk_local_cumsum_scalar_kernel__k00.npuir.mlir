// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @chunk_local_cumsum_scalar_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3072_i64 = arith.constant 3072 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c0_i64 = arith.constant 0 : i64
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = arith.index_cast %arg5 : i32 to index
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg9 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_0, %c1_i64 : i64
    %13 = hivm.hir.pointer_cast(%c1024_i64, %c3072_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %15 = arith.divsi %arg9, %arg8 : i32
    %16 = arith.remsi %15, %arg7 : i32
    %17 = arith.divsi %arg9, %4 : i32
    %18 = arith.remsi %17, %arg6 : i32
    %19 = arith.divsi %16, %c32_i32 : i32
    %20 = arith.remsi %16, %c32_i32 : i32
    %21 = arith.muli %19, %arg5 : i32
    %22 = arith.muli %21, %c32_i32 : i32
    %23 = arith.index_cast %22 : i32 to index
    %24 = arith.index_cast %20 : i32 to index
    %25 = arith.muli %18, %c32_i32 : i32
    %26 = arith.maxsi %25, %c0_i32 : i32
    %27 = arith.index_cast %26 : i32 to index
    %28 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 32 + s1 + s2)>()[%27, %23, %24]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%28], sizes: [32], strides: [32] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%28], sizes: [32], strides: [32] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
    %subview = memref.subview %14[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %29 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %27]
    %30 = affine.min affine_map<()[s0] -> (32, s0)>()[%29]
    %31 = arith.subi %c0_i32, %25 : i32
    %32 = arith.maxsi %31, %c0_i32 : i32
    %33 = arith.index_cast %32 : i32 to index
    %34 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%33, %29]
    %35 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%30, %34]
    %36 = arith.cmpi slt, %35, %c32 : index
    %subview_2 = memref.subview %reinterpret_cast[0] [%35] [1] : memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %subview[%34] [%35] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
    scf.if %36 {
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %base_buffer_13, %offset_14, %sizes_15, %strides_16 = memref.extract_strided_metadata %subview : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast_17 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %12]
    hivm.hir.load ins(%subview_2 : memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %34 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    %subview_4 = memref.subview %13[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %subview_4 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %12]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg10 = %c0 to %c1 step %c1 {
      %37 = memref.load %reinterpret_cast_5[%c0, %arg10] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      memref.store %37, %reinterpret_cast_10[%c0, %arg10] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.for %arg11 = %c1 to %c32 step %c1 {
        %38 = memref.load %reinterpret_cast_5[%arg11, %arg10] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %39 = arith.subi %arg11, %c1 : index
        %40 = memref.load %reinterpret_cast_10[%39, %arg10] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %41 = arith.addf %38, %40 : f32
        memref.store %41, %reinterpret_cast_10[%arg11, %arg10] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      }
    }
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %12]
    %subview_11 = memref.subview %subview_4[%34] [%35] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
    %subview_12 = memref.subview %reinterpret_cast_1[0] [%35] [1] : memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_11 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) outs(%subview_12 : memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %12]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
