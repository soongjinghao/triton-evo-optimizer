// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @chunk_local_cumsum_scalar_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3072_i64 = arith.constant 3072 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c0_i64 = arith.constant 0 : i64
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %c512_i32 = arith.constant 512 : i32
  %c13_i32 = arith.constant 13 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
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
    hivm.hir.set_mask_norm
    %6 = arith.divsi %arg9, %4 : i32
    %7 = arith.remsi %6, %arg6 : i32
    %8 = arith.muli %7, %c13_i32 : i32
    %9 = arith.addi %8, %c13_i32 : i32
    %10 = arith.minsi %9, %c512_i32 : i32
    scf.for %arg10 = %8 to %10 step %c1_i32  : i32 {
      %11 = arith.index_cast %arg10 : i32 to index
      %12 = arith.index_cast %8 : i32 to index
      %13 = arith.index_cast %10 : i32 to index
      %14 = arith.index_cast %c1_i32 : i32 to index
      %15 = arith.index_cast %arg9 : i32 to index
      %16 = arith.index_cast %3 : i32 to index
      %17 = arith.index_cast %1 : i32 to index
      %18 = arith.index_cast %c48_i32 : i32 to index
      %19 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%11, %12, %13, %14, %15, %16, %18]
      %20 = arith.index_cast %19 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %21 = arith.select %20, %c0_i64_0, %c1_i64 : i64
      %22 = hivm.hir.pointer_cast(%c1024_i64, %c3072_i64) : memref<32x8xf32, #hivm.address_space<ub>>
      annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
      %23 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<32x8xf32, #hivm.address_space<ub>>
      annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
      %24 = arith.divsi %arg10, %c128_i32 : i32
      %25 = arith.remsi %arg10, %c128_i32 : i32
      %26 = arith.remsi %25, %c32_i32 : i32
      %27 = arith.divsi %25, %c32_i32 : i32
      %28 = arith.muli %24, %arg5 : i32
      %29 = arith.muli %28, %c32_i32 : i32
      %30 = arith.addi %29, %26 : i32
      %31 = arith.index_cast %30 : i32 to index
      %32 = arith.muli %27, %c32_i32 : i32
      %33 = arith.maxsi %32, %c0_i32 : i32
      %34 = arith.index_cast %33 : i32 to index
      %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 32)>()[%31, %34]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [32], strides: [32] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [32], strides: [32] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %23[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
      %36 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%5, %34]
      %37 = affine.min affine_map<()[s0] -> (32, s0)>()[%36]
      %38 = arith.subi %c0_i32, %32 : i32
      %39 = arith.maxsi %38, %c0_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%40, %36]
      %42 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%37, %41]
      %43 = arith.cmpi slt, %42, %c32 : index
      %subview_2 = memref.subview %reinterpret_cast[0] [%42] [1] : memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %subview[%41] [%42] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      scf.if %43 {
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %base_buffer_13, %offset_14, %sizes_15, %strides_16 = memref.extract_strided_metadata %subview : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast_17 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %21]
      hivm.hir.load ins(%subview_2 : memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %41 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %subview_4 = memref.subview %22[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %subview_4 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %21]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg11 = %c0 to %c1 step %c1 {
        %44 = memref.load %reinterpret_cast_5[%c0, %arg11] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        memref.store %44, %reinterpret_cast_10[%c0, %arg11] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        scf.for %arg12 = %c1 to %c32 step %c1 {
          %45 = memref.load %reinterpret_cast_5[%arg12, %arg11] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %46 = arith.subi %arg12, %c1 : index
          %47 = memref.load %reinterpret_cast_10[%46, %arg11] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %48 = arith.addf %45, %47 : f32
          memref.store %48, %reinterpret_cast_10[%arg12, %arg11] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        }
      }
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %21]
      %subview_11 = memref.subview %subview_4[%41] [%42] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>
      %subview_12 = memref.subview %reinterpret_cast_1[0] [%42] [1] : memref<32xf32, strided<[32], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_11 : memref<?xf32, strided<[8], offset: ?>, #hivm.address_space<ub>>) outs(%subview_12 : memref<?xf32, strided<[32], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %21]
    } {pipeline.veconly}
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
