// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_rms_norm_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: f32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c39168_i64 = arith.constant 39168 : i64
  %c22784_i64 = arith.constant 22784 : i64
  %c35072_i64 = arith.constant 35072 : i64
  %c18688_i64 = arith.constant 18688 : i64
  %c30976_i64 = arith.constant 30976 : i64
  %c14592_i64 = arith.constant 14592 : i64
  %c14560_i64 = arith.constant 14560 : i64
  %c14528_i64 = arith.constant 14528 : i64
  %c12448_i64 = arith.constant 12448 : i64
  %c12416_i64 = arith.constant 12416 : i64
  %c8320_i64 = arith.constant 8320 : i64
  %c26880_i64 = arith.constant 26880 : i64
  %c4224_i64 = arith.constant 4224 : i64
  %c14496_i64 = arith.constant 14496 : i64
  %c4192_i64 = arith.constant 4192 : i64
  %c4160_i64 = arith.constant 4160 : i64
  %c4128_i64 = arith.constant 4128 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c1024 = arith.constant 1024 : index
  %cst_0 = arith.constant 1.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<1024xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c4096_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %cst, %6[%c0] : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %7 = arith.extsi %arg6 : i32 to i64
  %8 = arith.extsi %arg7 : i32 to i64
  %9 = arith.index_cast %arg8 : i32 to index
  %10 = arith.sitofp %arg8 : i32 to f32
  %11 = hivm.hir.pointer_cast(%c4128_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %10, %11[%c0] : memref<1xf32, #hivm.address_space<ub>>
  %12 = hivm.hir.pointer_cast(%c4160_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %arg9, %12[%c0] : memref<1xf32, #hivm.address_space<ub>>
  %13 = hivm.hir.pointer_cast(%c4192_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %cst_0, %13[%c0] : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %14 = arith.divsi %arg13, %4 : i32
    %15 = arith.remsi %14, %arg10 : i32
    %16 = arith.extsi %15 : i32 to i64
    %17 = arith.muli %16, %7 : i64
    %18 = arith.muli %16, %8 : i64
    %19 = arith.index_cast %17 : i64 to index
    %20 = hivm.hir.pointer_cast(%c14496_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%6 : memref<1xf32, #hivm.address_space<ub>>) outs(%20 : memref<1xf32, #hivm.address_space<ub>>)
    %21 = scf.for %arg14 = %c0_i32 to %arg8 step %c1024_i32 iter_args(%arg15 = %20) -> (memref<1xf32, #hivm.address_space<ub>>)  : i32 {
      %31 = hivm.hir.pointer_cast(%c4224_i64, %c26880_i64) : memref<1024xf32, #hivm.address_space<ub>>
      annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
      %32 = arith.index_cast %arg14 : i32 to index
      %33 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%19, %32]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %34 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%32, %9]
      %35 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%34, %32]
      %36 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%35, %32]
      %37 = arith.cmpi slt, %36, %c1024 : index
      %subview = memref.subview %reinterpret_cast[0] [%36] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %31[0] [%36] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %37 {
        hivm.hir.vbrc ins(%cst : f32) outs(%31 : memref<1024xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%31, %31 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%31 : memref<1024xf32, #hivm.address_space<ub>>)
      %subview_2 = memref.subview %31[0] [%36] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %38 = hivm.hir.pointer_cast(%c8320_i64) : memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%5 : memref<1024xf32, #hivm.address_space<ub>>) outs(%38 : memref<1024xf32, #hivm.address_space<ub>>)
      %subview_3 = memref.subview %38[0] [%36] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      %39 = hivm.hir.pointer_cast(%c12416_i64) : memref<1xf32, #hivm.address_space<ub>>
      %40 = hivm.hir.pointer_cast(%c12448_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%38 : memref<1024xf32, #hivm.address_space<ub>>) outs(%39 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%40 : memref<512xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %41 = memref.load %39[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      %42 = hivm.hir.pointer_cast(%c14496_i64) : memref<1xf32, #hivm.address_space<ub>>
      %43 = memref.load %arg15[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %44 = arith.addf %43, %41 : f32
      memref.store %44, %42[%c0] : memref<1xf32, #hivm.address_space<ub>>
      scf.yield %42 : memref<1xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %22 = hivm.hir.pointer_cast(%c14528_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%21, %11 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%22 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %23 = memref.load %22[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %24 = memref.load %12[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %25 = arith.addf %23, %24 : f32
    %26 = hivm.hir.pointer_cast(%c14560_i64) : memref<1xf32, #hivm.address_space<ub>>
    %27 = math.sqrt %25 : f32
    memref.store %27, %26[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %28 = hivm.hir.pointer_cast(%c14560_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vdiv ins(%13, %26 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%28 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %29 = memref.load %28[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %30 = arith.index_cast %18 : i64 to index
    scf.for %arg14 = %c0_i32 to %arg8 step %c1024_i32  : i32 {
      %31 = arith.index_cast %arg14 : i32 to index
      %32 = arith.index_cast %c0_i32 : i32 to index
      %33 = arith.index_cast %arg8 : i32 to index
      %34 = arith.index_cast %c1024_i32 : i32 to index
      %35 = arith.index_cast %arg13 : i32 to index
      %36 = arith.index_cast %3 : i32 to index
      %37 = arith.index_cast %1 : i32 to index
      %38 = arith.index_cast %c48_i32 : i32 to index
      %39 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%31, %32, %33, %34, %35, %36, %38]
      %40 = arith.index_cast %39 : index to i1
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %41 = arith.select %40, %c1_i64, %c2_i64 : i64
      %42 = arith.index_cast %39 : index to i1
      %c0_i64_1 = arith.constant 0 : i64
      %c1_i64_2 = arith.constant 1 : i64
      %43 = arith.select %42, %c0_i64_1, %c1_i64_2 : i64
      %44 = hivm.hir.pointer_cast(%c22784_i64, %c39168_i64) : memref<1024xf32, #hivm.address_space<ub>>
      annotation.mark %44 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
      %45 = hivm.hir.pointer_cast(%c18688_i64, %c35072_i64) : memref<1024xf32, #hivm.address_space<ub>>
      annotation.mark %45 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
      %46 = hivm.hir.pointer_cast(%c14592_i64, %c30976_i64) : memref<1024xf32, #hivm.address_space<ub>>
      annotation.mark %46 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
      %47 = arith.index_cast %arg14 : i32 to index
      %48 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%19, %47]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%48], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %49 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%47, %9]
      %50 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%49, %47]
      %51 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%50, %47]
      %52 = arith.cmpi slt, %51, %c1024 : index
      %subview = memref.subview %reinterpret_cast[0] [%51] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %46[0] [%51] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %52 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst : f32) outs(%46 : memref<1024xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %41]
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %reinterpret_cast_4[0] [%51] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %45[0] [%51] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %52 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%45 : memref<1024xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_5 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.vmul ins(%46, %29 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%46 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %43]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%46, %45 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%44 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %41]
      %53 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %47]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg5 to offset: [%53], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %44[0] [%51] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_9 = memref.subview %reinterpret_cast_7[0] [%51] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %43]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
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
