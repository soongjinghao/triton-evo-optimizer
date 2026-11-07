// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_rms_norm_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: f32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c30784_i64 = arith.constant 30784 : i64
  %c18496_i64 = arith.constant 18496 : i64
  %c26688_i64 = arith.constant 26688 : i64
  %c14400_i64 = arith.constant 14400 : i64
  %c12320_i64 = arith.constant 12320 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c22592_i64 = arith.constant 22592 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c14368_i64 = arith.constant 14368 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c1024_i32 = arith.constant 1024 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 1.000000e+00 : f32
  %c1024 = arith.constant 1024 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg10 : i32
  %7 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%7 : memref<1024xf32, #hivm.address_space<ub>>)
  %8 = hivm.hir.pointer_cast(%c14368_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %cst_0, %8[%c0] : memref<1xf32, #hivm.address_space<ub>>
  %9 = arith.extsi %6 : i32 to i64
  %10 = arith.extsi %arg6 : i32 to i64
  %11 = arith.muli %9, %10 : i64
  %12 = arith.index_cast %11 : i64 to index
  %13 = arith.extsi %arg7 : i32 to i64
  %14 = arith.muli %9, %13 : i64
  %15 = arith.index_cast %14 : i64 to index
  %16 = arith.index_cast %arg8 : i32 to index
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %17 = scf.for %arg13 = %c0_i32 to %arg8 step %c1024_i32 iter_args(%arg14 = %8) -> (memref<1xf32, #hivm.address_space<ub>>)  : i32 {
    %24 = hivm.hir.pointer_cast(%c4096_i64, %c22592_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %25 = arith.index_cast %arg13 : i32 to index
    %26 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%12, %25]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %27 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%25, %16]
    %28 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%27, %25]
    %29 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%28, %25]
    %30 = arith.cmpi slt, %29, %c1024 : index
    %subview = memref.subview %reinterpret_cast[0] [%29] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_1 = memref.subview %24[0] [%29] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %30 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%24 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%24, %24 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%24 : memref<1024xf32, #hivm.address_space<ub>>)
    %subview_2 = memref.subview %24[0] [%29] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c8192_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%7 : memref<1024xf32, #hivm.address_space<ub>>) outs(%31 : memref<1024xf32, #hivm.address_space<ub>>)
    %subview_3 = memref.subview %31[0] [%29] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
    %32 = hivm.hir.pointer_cast(%c12288_i64) : memref<1xf32, #hivm.address_space<ub>>
    %33 = hivm.hir.pointer_cast(%c12320_i64) : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%31 : memref<1024xf32, #hivm.address_space<ub>>) outs(%32 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%33 : memref<512xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %34 = memref.load %32[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %35 = hivm.hir.pointer_cast(%c14368_i64) : memref<1xf32, #hivm.address_space<ub>>
    %36 = memref.load %arg14[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %37 = arith.addf %36, %34 : f32
    memref.store %37, %35[%c0] : memref<1xf32, #hivm.address_space<ub>>
    scf.yield %35 : memref<1xf32, #hivm.address_space<ub>>
  }
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %18 = arith.sitofp %arg8 : i32 to f32
  %19 = memref.load %17[%c0] : memref<1xf32, #hivm.address_space<ub>>
  %20 = arith.divf %19, %18 : f32
  %21 = arith.addf %20, %arg9 : f32
  %22 = math.sqrt %21 : f32
  %23 = arith.divf %cst, %22 : f32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg13 = %c0_i32 to %arg8 step %c1024_i32  : i32 {
    %24 = arith.index_cast %arg13 : i32 to index
    %25 = arith.index_cast %c0_i32 : i32 to index
    %26 = arith.index_cast %arg8 : i32 to index
    %27 = arith.index_cast %c1024_i32 : i32 to index
    %28 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%24, %25, %27]
    %29 = arith.index_cast %28 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %30 = arith.select %29, %c0_i64_1, %c1_i64 : i64
    %31 = hivm.hir.pointer_cast(%c14400_i64, %c26688_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c18496_i64, %c30784_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %33 = hivm.hir.pointer_cast(%c14400_i64, %c26688_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %34 = arith.index_cast %arg13 : i32 to index
    %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%12, %34]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %36 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%34, %16]
    %37 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%36, %34]
    %38 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%37, %34]
    %39 = arith.cmpi slt, %38, %c1024 : index
    %subview = memref.subview %reinterpret_cast[0] [%38] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %33[0] [%38] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %39 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%33 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %30]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%34], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %reinterpret_cast_3[0] [%38] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %32[0] [%38] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %39 {
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%32 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_4 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.vmul ins(%33, %23 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%33 : memref<1024xf32, #hivm.address_space<ub>>)
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%33, %32 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%31 : memref<1024xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %40 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%15, %34]
    %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%40], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %31[0] [%38] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_8 = memref.subview %reinterpret_cast_6[0] [%38] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %30]
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
