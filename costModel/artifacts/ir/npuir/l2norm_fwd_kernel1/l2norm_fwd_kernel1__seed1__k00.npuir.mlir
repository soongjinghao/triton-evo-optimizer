// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @l2norm_fwd_kernel1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c8320_i64 = arith.constant 8320 : i64
  %c8256_i64 = arith.constant 8256 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c8832_i64 = arith.constant 8832 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.muli %arg7, %c16_i32 : i32
  %6 = arith.index_cast %arg5 : i32 to index
  %7 = arith.index_cast %5 : i32 to index
  %8 = affine.max affine_map<()[s0] -> (0, s0)>()[%6]
  %9 = affine.min affine_map<()[s0] -> (64, s0)>()[%8]
  %10 = affine.min affine_map<()[s0] -> (0, 64, s0)>()[%8]
  %11 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%9, %10]
  %12 = arith.cmpi slt, %11, %c64 : index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %13 = arith.index_cast %arg10 : i32 to index
    %14 = arith.index_cast %3 : i32 to index
    %15 = arith.index_cast %1 : i32 to index
    %16 = arith.index_cast %c48_i32 : i32 to index
    %17 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%13, %14, %16]
    %18 = arith.index_cast %17 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %19 = arith.select %18, %c0_i64_1, %c1_i64 : i64
    %20 = hivm.hir.pointer_cast(%c0_i64, %c8832_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<16x64xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c0_i64, %c8832_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<16x64xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %22 = arith.divsi %arg10, %4 : i32
    %23 = arith.remsi %22, %arg7 : i32
    %24 = arith.muli %23, %c16_i32 : i32
    %25 = arith.maxsi %24, %c0_i32 : i32
    %26 = arith.index_cast %25 : i32 to index
    %27 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%26, %6]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [16, 64], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%27], sizes: [16, 64], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %28 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%7, %26]
    %29 = affine.min affine_map<()[s0] -> (16, s0)>()[%28]
    %30 = arith.subi %c0_i32, %24 : i32
    %31 = arith.maxsi %30, %c0_i32 : i32
    %32 = arith.index_cast %31 : i32 to index
    %33 = affine.min affine_map<()[s0, s1] -> (16, s1, s0)>()[%32, %28]
    %34 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%29, %33]
    %35 = arith.cmpi slt, %34, %c16 : index
    %36 = arith.ori %35, %12 : i1
    %subview = memref.subview %reinterpret_cast[0, 0] [%34, %11] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %21[%33, %10] [%34, %11] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %36 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_8 = memref.collapse_shape %21 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_8 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %19]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %10 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %37 = hivm.hir.pointer_cast(%c4096_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %21 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %collapse_shape_4 = memref.collapse_shape %37 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape, %collapse_shape : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<1024xf32, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c8192_i64) : memref<16x1xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%37 : memref<16x64xf32, #hivm.address_space<ub>>) outs(%38 : memref<16x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_5 = memref.collapse_shape %38 [[0, 1]] : memref<16x1xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c8192_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_5, %arg6 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%39 : memref<16xf32, #hivm.address_space<ub>>)
    %40 = hivm.hir.pointer_cast(%c8192_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsqrt ins(%39 : memref<16xf32, #hivm.address_space<ub>>) outs(%40 : memref<16xf32, #hivm.address_space<ub>>)
    %41 = hivm.hir.pointer_cast(%c8256_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%41 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%41, %40 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%41 : memref<16xf32, #hivm.address_space<ub>>)
    %expand_shape = memref.expand_shape %41 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
    %42 = hivm.hir.pointer_cast(%c8320_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%21, %expand_shape : memref<16x64xf32, #hivm.address_space<ub>>, memref<16x1xf32, #hivm.address_space<ub>>) outs(%20 : memref<16x64xf32, #hivm.address_space<ub>>) temp_buffer(%42 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_6 = memref.subview %20[%33, %10] [%34, %11] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_7 = memref.subview %reinterpret_cast_2[0, 0] [%34, %11] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_6 : memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %19]
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
