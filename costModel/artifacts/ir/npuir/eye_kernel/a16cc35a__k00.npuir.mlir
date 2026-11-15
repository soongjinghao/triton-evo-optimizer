// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @eye_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, false, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c13440_i64 = arith.constant 13440 : i64
  %c13536_i64 = arith.constant 13536 : i64
  %c9344_i64 = arith.constant 9344 : i64
  %c8832_i64 = arith.constant 8832 : i64
  %c4736_i64 = arith.constant 4736 : i64
  %c640_i64 = arith.constant 640 : i64
  %c384_i64 = arith.constant 384 : i64
  %c320_i64 = arith.constant 320 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %c64_i32 = arith.constant 64 : i32
  %c16_i32 = arith.constant 16 : i32
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
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%5 : memref<16xi32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c64_i64) : memref<64xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%6 : memref<64xi32, #hivm.address_space<ub>>)
  %7 = arith.index_cast %arg5 : i32 to index
  %8 = arith.index_cast %arg4 : i32 to index
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %9 = arith.index_cast %arg9 : i32 to index
    %10 = arith.index_cast %3 : i32 to index
    %11 = arith.index_cast %1 : i32 to index
    %12 = arith.index_cast %c48_i32 : i32 to index
    %13 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%9, %10, %12]
    %14 = arith.index_cast %13 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %15 = arith.select %14, %c0_i64_1, %c1_i64 : i64
    %16 = hivm.hir.pointer_cast(%c9344_i64, %c13536_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<16x64xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %17 = arith.divsi %arg9, %arg8 : i32
    %18 = arith.remsi %17, %arg7 : i32
    %19 = arith.divsi %arg9, %4 : i32
    %20 = arith.remsi %19, %arg6 : i32
    %21 = arith.muli %20, %c16_i32 : i32
    %22 = hivm.hir.pointer_cast(%c320_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%5, %21 : memref<16xi32, #hivm.address_space<ub>>, i32) outs(%22 : memref<16xi32, #hivm.address_space<ub>>)
    %23 = arith.muli %18, %c64_i32 : i32
    %24 = hivm.hir.pointer_cast(%c384_i64) : memref<64xi32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%6, %23 : memref<64xi32, #hivm.address_space<ub>>, i32) outs(%24 : memref<64xi32, #hivm.address_space<ub>>)
    %expand_shape = memref.expand_shape %24 [[0, 1]] output_shape [1, 64] : memref<64xi32, #hivm.address_space<ub>> into memref<1x64xi32, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c640_i64) : memref<16x64xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape : memref<1x64xi32, #hivm.address_space<ub>>) outs(%25 : memref<16x64xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %26 = arith.index_cast %21 : i32 to index
    %27 = arith.index_cast %23 : i32 to index
    %28 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%27, %26, %7]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%28], sizes: [16, 64], strides: [%7, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %expand_shape_2 = memref.expand_shape %22 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c4736_i64) : memref<16x64xi32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c8832_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%expand_shape_2 : memref<16x1xi32, #hivm.address_space<ub>>) outs(%29 : memref<16x64xi32, #hivm.address_space<ub>>) temp_buffer(%30 : memref<128xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
    %31 = hivm.hir.pointer_cast(%c640_i64) : memref<16x64xi1, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %29 [[0, 1]] : memref<16x64xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    %collapse_shape_3 = memref.collapse_shape %25 [[0, 1]] : memref<16x64xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    %collapse_shape_4 = memref.collapse_shape %31 [[0, 1]] : memref<16x64xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_3 : memref<1024xi32, #hivm.address_space<ub>>, memref<1024xi32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<1024xi1, #hivm.address_space<ub>>)
    %collapse_shape_5 = memref.collapse_shape %16 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c13440_i64) : memref<24xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %15]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %cst, %cst_0 : memref<1024xi1, #hivm.address_space<ub>>, f32, f32) outs(%collapse_shape_5 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%32 : memref<24xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %33 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%26, %8]
    %34 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%33, %26]
    %35 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%27, %7]
    %36 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%35, %27]
    %37 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%34, %26]
    %38 = affine.max affine_map<()[s0] -> (0, s0)>()[%37]
    %39 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%36, %27]
    %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
    %subview = memref.subview %16[0, 0] [%38, %40] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %reinterpret_cast[0, 0] [%38, %40] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %15]
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
