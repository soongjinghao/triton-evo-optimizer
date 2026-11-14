// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_log_softmax_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c147808_i64 = arith.constant 147808 : i64
  %c82272_i64 = arith.constant 82272 : i64
  %c82240_i64 = arith.constant 82240 : i64
  %c65856_i64 = arith.constant 65856 : i64
  %c65824_i64 = arith.constant 65824 : i64
  %c33056_i64 = arith.constant 33056 : i64
  %c32800_i64 = arith.constant 32800 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c115040_i64 = arith.constant 115040 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c8192 = arith.constant 8192 : index
  %cst_0 = arith.constant 0xFF800000 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.index_cast %arg6 : i32 to index
  %6 = affine.max affine_map<()[s0] -> (0, s0)>()[%5]
  %7 = affine.min affine_map<()[s0] -> (8192, s0)>()[%6]
  %8 = arith.cmpi slt, %7, %c8192 : index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %9 = arith.divsi %arg10, %4 : i32
    %10 = arith.remsi %9, %arg7 : i32
    scf.for %arg11 = %10 to %arg5 step %arg7  : i32 {
      %11 = arith.index_cast %arg11 : i32 to index
      %12 = arith.index_cast %10 : i32 to index
      %13 = arith.index_cast %arg5 : i32 to index
      %14 = arith.index_cast %arg7 : i32 to index
      %15 = arith.index_cast %arg10 : i32 to index
      %16 = arith.index_cast %3 : i32 to index
      %17 = arith.index_cast %1 : i32 to index
      %18 = arith.index_cast %c48_i32 : i32 to index
      %19 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%11, %12, %13, %14, %15, %16, %18]
      %20 = arith.index_cast %19 : index to i1
      %c0_i64_1 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %21 = arith.select %20, %c0_i64_1, %c1_i64 : i64
      %22 = arith.index_cast %19 : index to i1
      %c1_i64_2 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %23 = arith.select %22, %c1_i64_2, %c2_i64 : i64
      %24 = hivm.hir.pointer_cast(%c82272_i64, %c147808_i64) : memref<8192xf32, #hivm.address_space<ub>>
      annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<8192xf32, #hivm.address_space<ub>>
      %25 = hivm.hir.pointer_cast(%c0_i64, %c115040_i64) : memref<8192xf32, #hivm.address_space<ub>>
      annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<8192xf32, #hivm.address_space<ub>>
      %26 = arith.muli %arg11, %arg6 : i32
      %27 = arith.index_cast %26 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [8192], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%27], sizes: [8192], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview = memref.subview %reinterpret_cast[0] [%7] [1] : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %25[0] [%7] [1] : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %8 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%25 : memref<8192xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %23]
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %28 = hivm.hir.pointer_cast(%c32768_i64) : memref<1xf32, #hivm.address_space<ub>>
      %29 = hivm.hir.pointer_cast(%c32800_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vreduce <max> ins(%25 : memref<8192xf32, #hivm.address_space<ub>>) outs(%28 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%29 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %30 = memref.load %28[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %31 = arith.subf %cst, %30 : f32
      hivm.hir.vadd ins(%25, %31 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%25 : memref<8192xf32, #hivm.address_space<ub>>)
      %32 = hivm.hir.pointer_cast(%c33056_i64) : memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vexp ins(%25 : memref<8192xf32, #hivm.address_space<ub>>) outs(%32 : memref<8192xf32, #hivm.address_space<ub>>)
      %33 = hivm.hir.pointer_cast(%c65824_i64) : memref<1xf32, #hivm.address_space<ub>>
      %34 = hivm.hir.pointer_cast(%c65856_i64) : memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%32 : memref<8192xf32, #hivm.address_space<ub>>) outs(%33 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%34 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %35 = memref.load %33[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %36 = hivm.hir.pointer_cast(%c82240_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %35, %36[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %37 = hivm.hir.pointer_cast(%c82240_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vln ins(%36 : memref<1xf32, #hivm.address_space<ub>>) outs(%37 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %38 = memref.load %37[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %39 = arith.subf %cst, %38 : f32
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %21]
      hivm.hir.vadd ins(%25, %39 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%24 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %23]
      %subview_5 = memref.subview %24[0] [%7] [1] : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_6 = memref.subview %reinterpret_cast_3[0] [%7] [1] : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_5 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %21]
    } {pipeline.veconly}
  } {autoblockify.subloop}
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
