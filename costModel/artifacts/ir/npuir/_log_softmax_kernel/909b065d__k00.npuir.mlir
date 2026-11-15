// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_log_softmax_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c45728_i64 = arith.constant 45728 : i64
  %c21152_i64 = arith.constant 21152 : i64
  %c37536_i64 = arith.constant 37536 : i64
  %c12960_i64 = arith.constant 12960 : i64
  %c12928_i64 = arith.constant 12928 : i64
  %c8832_i64 = arith.constant 8832 : i64
  %c8800_i64 = arith.constant 8800 : i64
  %c8768_i64 = arith.constant 8768 : i64
  %c8736_i64 = arith.constant 8736 : i64
  %c8704_i64 = arith.constant 8704 : i64
  %c8672_i64 = arith.constant 8672 : i64
  %c8640_i64 = arith.constant 8640 : i64
  %c8608_i64 = arith.constant 8608 : i64
  %c8576_i64 = arith.constant 8576 : i64
  %c8544_i64 = arith.constant 8544 : i64
  %c8512_i64 = arith.constant 8512 : i64
  %c8480_i64 = arith.constant 8480 : i64
  %c8224_i64 = arith.constant 8224 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c29344_i64 = arith.constant 29344 : i64
  %c0_i64 = arith.constant 0 : i64
  %c2048_i32 = arith.constant 2048 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c2048 = arith.constant 2048 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = arith.extsi %arg5 : i32 to i64
  %6 = arith.extsi %arg6 : i32 to i64
  %7 = arith.index_cast %arg7 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %8 = arith.divsi %arg11, %4 : i32
    %9 = arith.remsi %8, %arg8 : i32
    %10 = arith.extsi %9 : i32 to i64
    %11 = arith.muli %10, %5 : i64
    %12 = arith.muli %10, %6 : i64
    %13 = arith.index_cast %11 : i64 to index
    %14:2 = scf.for %arg12 = %c0_i32 to %arg7 step %c2048_i32 iter_args(%arg13 = %cst, %arg14 = %cst_0) -> (f32, f32)  : i32 {
      %20 = hivm.hir.pointer_cast(%c0_i64, %c29344_i64) : memref<2048xf32, #hivm.address_space<ub>>
      annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<2048xf32, #hivm.address_space<ub>>
      %21 = arith.index_cast %arg12 : i32 to index
      %22 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%13, %21]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%22], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %23 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%21, %7]
      %24 = affine.min affine_map<()[s0, s1] -> (s1 + 2048, s0)>()[%23, %21]
      %25 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%24, %21]
      %26 = arith.cmpi slt, %25, %c2048 : index
      %subview = memref.subview %reinterpret_cast[0] [%25] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %20[0] [%25] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %26 {
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%20 : memref<2048xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %27 = hivm.hir.pointer_cast(%c8192_i64) : memref<1xf32, #hivm.address_space<ub>>
      %28 = hivm.hir.pointer_cast(%c8224_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vreduce <max> ins(%20 : memref<2048xf32, #hivm.address_space<ub>>) outs(%27 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%28 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %29 = memref.load %27[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %30 = hivm.hir.pointer_cast(%c8480_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %arg14, %30[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      %31 = hivm.hir.pointer_cast(%c8512_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %29, %31[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %32 = hivm.hir.pointer_cast(%c8544_i64) : memref<1xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.vcmp ins(%30, %30 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%32 : memref<1xi1, #hivm.address_space<ub>>)
      %33 = hivm.hir.pointer_cast(%c8544_i64) : memref<1xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%32 : memref<1xi1, #hivm.address_space<ub>>) outs(%33 : memref<1xi1, #hivm.address_space<ub>>)
      %34 = hivm.hir.pointer_cast(%c8576_i64) : memref<1xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcmp ins(%31, %31 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%34 : memref<1xi1, #hivm.address_space<ub>>)
      %35 = hivm.hir.pointer_cast(%c8576_i64) : memref<1xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%34 : memref<1xi1, #hivm.address_space<ub>>) outs(%35 : memref<1xi1, #hivm.address_space<ub>>)
      %36 = hivm.hir.pointer_cast(%c8608_i64) : memref<1xf32, #hivm.address_space<ub>>
      %37 = arith.maximumf %arg14, %29 : f32
      memref.store %37, %36[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %38 = hivm.hir.pointer_cast(%c8640_i64) : memref<1xf32, #hivm.address_space<ub>>
      %39 = hivm.hir.pointer_cast(%c8672_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vsel ins(%33, %31, %36 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%38 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%39 : memref<8xf32, #hivm.address_space<ub>>)
      %40 = hivm.hir.pointer_cast(%c8704_i64) : memref<1xf32, #hivm.address_space<ub>>
      %41 = hivm.hir.pointer_cast(%c8736_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%35, %30, %38 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%40 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%41 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %42 = memref.load %40[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %43 = arith.subf %arg14, %42 : f32
      %44 = hivm.hir.pointer_cast(%c8768_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %43, %44[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %45 = hivm.hir.pointer_cast(%c8768_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vexp ins(%44 : memref<1xf32, #hivm.address_space<ub>>) outs(%45 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %46 = memref.load %45[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %47 = arith.mulf %arg13, %46 : f32
      %48 = arith.subf %cst, %42 : f32
      hivm.hir.vadd ins(%20, %48 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%20 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vexp ins(%20 : memref<2048xf32, #hivm.address_space<ub>>) outs(%20 : memref<2048xf32, #hivm.address_space<ub>>)
      %49 = hivm.hir.pointer_cast(%c8800_i64) : memref<1xf32, #hivm.address_space<ub>>
      %50 = hivm.hir.pointer_cast(%c8832_i64) : memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%20 : memref<2048xf32, #hivm.address_space<ub>>) outs(%49 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%50 : memref<1024xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %51 = memref.load %49[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %52 = arith.addf %47, %51 : f32
      scf.yield %52, %42 : f32, f32
    } {pipeline.veconly}
    %15 = hivm.hir.pointer_cast(%c12928_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %14#0, %15[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %16 = hivm.hir.pointer_cast(%c12928_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vln ins(%15 : memref<1xf32, #hivm.address_space<ub>>) outs(%16 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %17 = memref.load %16[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %18 = arith.addf %17, %14#1 : f32
    %19 = arith.index_cast %12 : i64 to index
    scf.for %arg12 = %c0_i32 to %arg7 step %c2048_i32  : i32 {
      %20 = arith.index_cast %arg12 : i32 to index
      %21 = arith.index_cast %c0_i32 : i32 to index
      %22 = arith.index_cast %arg7 : i32 to index
      %23 = arith.index_cast %c2048_i32 : i32 to index
      %24 = arith.index_cast %arg11 : i32 to index
      %25 = arith.index_cast %3 : i32 to index
      %26 = arith.index_cast %1 : i32 to index
      %27 = arith.index_cast %c48_i32 : i32 to index
      %28 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%20, %21, %22, %23, %24, %25, %27]
      %29 = arith.index_cast %28 : index to i1
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %30 = arith.select %29, %c1_i64, %c2_i64 : i64
      %31 = arith.index_cast %28 : index to i1
      %c0_i64_1 = arith.constant 0 : i64
      %c1_i64_2 = arith.constant 1 : i64
      %32 = arith.select %31, %c0_i64_1, %c1_i64_2 : i64
      %33 = hivm.hir.pointer_cast(%c21152_i64, %c45728_i64) : memref<2048xf32, #hivm.address_space<ub>>
      annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<2048xf32, #hivm.address_space<ub>>
      %34 = hivm.hir.pointer_cast(%c12960_i64, %c37536_i64) : memref<2048xf32, #hivm.address_space<ub>>
      annotation.mark %34 {hivm.multi_buffer = 2 : i32} : memref<2048xf32, #hivm.address_space<ub>>
      %35 = arith.index_cast %arg12 : i32 to index
      %36 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%13, %35]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%36], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %37 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%35, %7]
      %38 = affine.min affine_map<()[s0, s1] -> (s1 + 2048, s0)>()[%37, %35]
      %39 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%38, %35]
      %40 = arith.cmpi slt, %39, %c2048 : index
      %subview = memref.subview %reinterpret_cast[0] [%39] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %34[0] [%39] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %40 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst : f32) outs(%34 : memref<2048xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %30]
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %41 = arith.subf %cst, %18 : f32
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %32]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vadd ins(%34, %41 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%33 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %30]
      %42 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%19, %35]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%42], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %33[0] [%39] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_6 = memref.subview %reinterpret_cast_4[0] [%39] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_5 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %32]
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
