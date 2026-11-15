// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @l2norm_fwd_kernel1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c2112_i64 = arith.constant 2112 : i64
  %c2080_i64 = arith.constant 2080 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c2368_i64 = arith.constant 2368 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %c64_i32 = arith.constant 64 : i32
  %c4_i32 = arith.constant 4 : i32
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.muli %arg7, %c4_i32 : i32
  %6 = arith.index_cast %arg5 : i32 to index
  %7 = arith.index_cast %5 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %8 = arith.index_cast %arg10 : i32 to index
    %9 = arith.index_cast %3 : i32 to index
    %10 = arith.index_cast %1 : i32 to index
    %11 = arith.index_cast %c48_i32 : i32 to index
    %12 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%8, %9, %11]
    %13 = arith.index_cast %12 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %14 = arith.select %13, %c0_i64_1, %c1_i64 : i64
    %15 = hivm.hir.pointer_cast(%c0_i64, %c2368_i64) : memref<4x64xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<4x64xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c0_i64, %c2368_i64) : memref<4x64xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<4x64xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %17 = arith.divsi %arg10, %arg9 : i32
    %18 = arith.remsi %17, %arg8 : i32
    %19 = arith.divsi %arg10, %4 : i32
    %20 = arith.remsi %19, %arg7 : i32
    %21 = arith.muli %20, %c4_i32 : i32
    %22 = arith.muli %18, %c64_i32 : i32
    %23 = arith.index_cast %21 : i32 to index
    %24 = arith.index_cast %22 : i32 to index
    %25 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%24, %23, %6]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [4, 64], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %26 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%23, %7]
    %27 = affine.min affine_map<()[s0, s1] -> (s1 + 4, s0)>()[%26, %23]
    %28 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%24, %6]
    %29 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%28, %24]
    %30 = affine.min affine_map<()[s0, s1] -> (4, s0 - s1)>()[%27, %23]
    %31 = affine.max affine_map<()[s0] -> (0, s0)>()[%30]
    %32 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%29, %24]
    %33 = affine.max affine_map<()[s0] -> (0, s0)>()[%32]
    %34 = arith.cmpi slt, %31, %c4 : index
    %35 = arith.cmpi slt, %33, %c64 : index
    %36 = arith.ori %34, %35 : i1
    %subview = memref.subview %reinterpret_cast[0, 0] [%31, %33] [1, 1] : memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %16[0, 0] [%31, %33] [1, 1] : memref<4x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    scf.if %36 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_8 = memref.collapse_shape %16 [[0, 1]] : memref<4x64xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_8 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %14]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %37 = hivm.hir.pointer_cast(%c1024_i64) : memref<4x64xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %16 [[0, 1]] : memref<4x64xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_3 = memref.collapse_shape %37 [[0, 1]] : memref<4x64xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape, %collapse_shape : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<256xf32, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c2048_i64) : memref<4x1xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%37 : memref<4x64xf32, #hivm.address_space<ub>>) outs(%38 : memref<4x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_4 = memref.collapse_shape %38 [[0, 1]] : memref<4x1xf32, #hivm.address_space<ub>> into memref<4xf32, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c2048_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_4, %arg6 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%39 : memref<4xf32, #hivm.address_space<ub>>)
    %40 = hivm.hir.pointer_cast(%c2048_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsqrt ins(%39 : memref<4xf32, #hivm.address_space<ub>>) outs(%40 : memref<4xf32, #hivm.address_space<ub>>)
    %41 = hivm.hir.pointer_cast(%c2080_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%41 : memref<4xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%41, %40 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%41 : memref<4xf32, #hivm.address_space<ub>>)
    %expand_shape = memref.expand_shape %41 [[0, 1]] output_shape [4, 1] : memref<4xf32, #hivm.address_space<ub>> into memref<4x1xf32, #hivm.address_space<ub>>
    %42 = hivm.hir.pointer_cast(%c2112_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%16, %expand_shape : memref<4x64xf32, #hivm.address_space<ub>>, memref<4x1xf32, #hivm.address_space<ub>>) outs(%15 : memref<4x64xf32, #hivm.address_space<ub>>) temp_buffer(%42 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%25], sizes: [4, 64], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %15[0, 0] [%31, %33] [1, 1] : memref<4x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%31, %33] [1, 1] : memref<4x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_6 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %14]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
