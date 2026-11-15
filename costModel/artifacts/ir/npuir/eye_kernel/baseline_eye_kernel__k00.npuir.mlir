// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @eye_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, false, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c13696_i64 = arith.constant 13696 : i64
  %c13792_i64 = arith.constant 13792 : i64
  %c9600_i64 = arith.constant 9600 : i64
  %c5504_i64 = arith.constant 5504 : i64
  %c4480_i64 = arith.constant 4480 : i64
  %c384_i64 = arith.constant 384 : i64
  %c256_i64 = arith.constant 256 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %c32_i32 = arith.constant 32 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<32xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%5 : memref<32xi32, #hivm.address_space<ub>>)
  %6 = arith.index_cast %arg5 : i32 to index
  %7 = arith.index_cast %arg4 : i32 to index
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %8 = arith.index_cast %arg9 : i32 to index
    %9 = arith.index_cast %3 : i32 to index
    %10 = arith.index_cast %1 : i32 to index
    %11 = arith.index_cast %c48_i32 : i32 to index
    %12 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%8, %9, %11]
    %13 = arith.index_cast %12 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %14 = arith.select %13, %c0_i64_1, %c1_i64 : i64
    %15 = hivm.hir.pointer_cast(%c9600_i64, %c13792_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %16 = arith.divsi %arg9, %arg8 : i32
    %17 = arith.remsi %16, %arg7 : i32
    %18 = arith.divsi %arg9, %4 : i32
    %19 = arith.remsi %18, %arg6 : i32
    %20 = arith.muli %19, %c32_i32 : i32
    %21 = hivm.hir.pointer_cast(%c128_i64) : memref<32xi32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%5, %20 : memref<32xi32, #hivm.address_space<ub>>, i32) outs(%21 : memref<32xi32, #hivm.address_space<ub>>)
    %22 = arith.muli %17, %c32_i32 : i32
    %23 = hivm.hir.pointer_cast(%c256_i64) : memref<32xi32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%5, %22 : memref<32xi32, #hivm.address_space<ub>>, i32) outs(%23 : memref<32xi32, #hivm.address_space<ub>>)
    %expand_shape = memref.expand_shape %21 [[0, 1]] output_shape [32, 1] : memref<32xi32, #hivm.address_space<ub>> into memref<32x1xi32, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c384_i64) : memref<32x32xi32, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c4480_i64) : memref<256xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape : memref<32x1xi32, #hivm.address_space<ub>>) outs(%24 : memref<32x32xi32, #hivm.address_space<ub>>) temp_buffer(%25 : memref<256xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
    %expand_shape_2 = memref.expand_shape %23 [[0, 1]] output_shape [1, 32] : memref<32xi32, #hivm.address_space<ub>> into memref<1x32xi32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c5504_i64) : memref<32x32xi32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%expand_shape_2 : memref<1x32xi32, #hivm.address_space<ub>>) outs(%26 : memref<32x32xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %27 = hivm.hir.pointer_cast(%c384_i64) : memref<32x32xi1, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %24 [[0, 1]] : memref<32x32xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    %collapse_shape_3 = memref.collapse_shape %26 [[0, 1]] : memref<32x32xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    %collapse_shape_4 = memref.collapse_shape %27 [[0, 1]] : memref<32x32xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_3 : memref<1024xi32, #hivm.address_space<ub>>, memref<1024xi32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<1024xi1, #hivm.address_space<ub>>)
    %collapse_shape_5 = memref.collapse_shape %15 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c13696_i64) : memref<24xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %14]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %cst, %cst_0 : memref<1024xi1, #hivm.address_space<ub>>, f32, f32) outs(%collapse_shape_5 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%28 : memref<24xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %29 = arith.index_cast %20 : i32 to index
    %30 = arith.index_cast %22 : i32 to index
    %31 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%30, %29, %6]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [32, 32], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %32 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%29, %7]
    %33 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%32, %29]
    %34 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%30, %6]
    %35 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%34, %30]
    %36 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%33, %29]
    %37 = affine.max affine_map<()[s0] -> (0, s0)>()[%36]
    %38 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%35, %30]
    %39 = affine.max affine_map<()[s0] -> (0, s0)>()[%38]
    %subview = memref.subview %15[0, 0] [%37, %39] [1, 1] : memref<32x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %reinterpret_cast[0, 0] [%37, %39] [1, 1] : memref<32x32xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %14]
  } {autoblockify.subloop}
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
