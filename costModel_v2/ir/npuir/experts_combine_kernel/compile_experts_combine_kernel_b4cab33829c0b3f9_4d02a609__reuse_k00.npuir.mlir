// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @experts_combine_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c12288_i64 = arith.constant 12288 : i64
  %c20512_i64 = arith.constant 20512 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c16416_i64 = arith.constant 16416 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c12320_i64 = arith.constant 12320 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 1.41421354 : f32
  %c1024_i32 = arith.constant 1024 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg9, %4 : i32
    %6 = arith.remsi %5, %arg6 : i32
    %7 = arith.muli %6, %c4_i32 : i32
    scf.for %arg10 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %8 = arith.index_cast %arg10 : i32 to index
      %9 = arith.index_cast %c0_i32 : i32 to index
      %10 = arith.index_cast %c4_i32 : i32 to index
      %11 = arith.index_cast %c1_i32 : i32 to index
      %12 = arith.index_cast %arg9 : i32 to index
      %13 = arith.index_cast %3 : i32 to index
      %14 = arith.index_cast %1 : i32 to index
      %15 = arith.index_cast %c48_i32 : i32 to index
      %16 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%8, %9, %10, %11, %12, %13, %15]
      %17 = arith.index_cast %16 : index to i1
      %c2_i64 = arith.constant 2 : i64
      %c3_i64 = arith.constant 3 : i64
      %18 = arith.select %17, %c2_i64, %c3_i64 : i64
      %19 = arith.index_cast %16 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %20 = arith.select %19, %c0_i64_0, %c1_i64 : i64
      %21 = hivm.hir.pointer_cast(%c8192_i64, %c20512_i64) : memref<1024xf32, #hivm.address_space<ub>>
      annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
      %22 = hivm.hir.pointer_cast(%c4096_i64, %c16416_i64) : memref<1024xf32, #hivm.address_space<ub>>
      annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
      %23 = hivm.hir.pointer_cast(%c0_i64, %c12320_i64) : memref<1x1024xf32, #hivm.address_space<ub>>
      annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<1x1024xf32, #hivm.address_space<ub>>
      %24 = arith.addi %7, %arg10 : i32
      %25 = arith.muli %24, %c1024_i32 : i32
      %26 = arith.index_cast %25 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [1, 1024], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x1024xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape = memref.collapse_shape %reinterpret_cast [[0, 1]] : memref<1x1024xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_1 = memref.collapse_shape %23 [[0, 1]] : memref<1x1024xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %20]
      hivm.hir.load ins(%collapse_shape : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_1 : memref<1024xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%26], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %18]
      hivm.hir.load ins(%reinterpret_cast_2 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%22 : memref<1024xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vadd ins(%collapse_shape_1, %22 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%22 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %20]
      %27 = hivm.hir.pointer_cast(%c12288_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %20]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vdiv ins(%22, %cst : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%21 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%27 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %18]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%21 : memref<1024xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_3 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %20]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
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
