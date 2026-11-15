// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @eye_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, false, false, false]> : vector<7xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c13184_i64 = arith.constant 13184 : i64
  %c13280_i64 = arith.constant 13280 : i64
  %c9088_i64 = arith.constant 9088 : i64
  %c8576_i64 = arith.constant 8576 : i64
  %c4480_i64 = arith.constant 4480 : i64
  %c4416_i64 = arith.constant 4416 : i64
  %c320_i64 = arith.constant 320 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c16_i32 = arith.constant 16 : i32
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg4, %arg5 : i32
  %1 = arith.muli %0, %arg6 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%2 : memref<16xi32, #hivm.address_space<ub>>)
  %3 = hivm.hir.pointer_cast(%c64_i64) : memref<64xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%3 : memref<64xi32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %3 [[0, 1]] output_shape [1, 64] : memref<64xi32, #hivm.address_space<ub>> into memref<1x64xi32, #hivm.address_space<ub>>
  %4 = hivm.hir.pointer_cast(%c320_i64) : memref<16x64xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<1x64xi32, #hivm.address_space<ub>>) outs(%4 : memref<16x64xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg7 = %c0_i32 to %c64_i32 step %c16_i32  : i32 {
    %5 = arith.index_cast %arg7 : i32 to index
    %6 = arith.index_cast %c0_i32 : i32 to index
    %7 = arith.index_cast %c64_i32 : i32 to index
    %8 = arith.index_cast %c16_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_1, %c1_i64 : i64
    %12 = hivm.hir.pointer_cast(%c9088_i64, %c13280_i64) : memref<16x64xf32, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<16x64xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c4416_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%2, %arg7 : memref<16xi32, #hivm.address_space<ub>>, i32) outs(%13 : memref<16xi32, #hivm.address_space<ub>>)
    %expand_shape_2 = memref.expand_shape %13 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c4480_i64) : memref<16x64xi32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c8576_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_2 : memref<16x1xi32, #hivm.address_space<ub>>) outs(%14 : memref<16x64xi32, #hivm.address_space<ub>>) temp_buffer(%15 : memref<128xi32, #hivm.address_space<ub>>) broadcast_dims = [1]
    %16 = hivm.hir.pointer_cast(%c4480_i64) : memref<16x64xi1, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %14 [[0, 1]] : memref<16x64xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    %collapse_shape_3 = memref.collapse_shape %4 [[0, 1]] : memref<16x64xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
    %collapse_shape_4 = memref.collapse_shape %16 [[0, 1]] : memref<16x64xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%collapse_shape, %collapse_shape_3 : memref<1024xi32, #hivm.address_space<ub>>, memref<1024xi32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<1024xi1, #hivm.address_space<ub>>)
    %collapse_shape_5 = memref.collapse_shape %12 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c13184_i64) : memref<24xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %11]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_4, %cst_0, %cst : memref<1024xi1, #hivm.address_space<ub>>, f32, f32) outs(%collapse_shape_5 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%17 : memref<24xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %18 = arith.index_cast %arg7 : i32 to index
    %19 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%18]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%19], sizes: [16, 64], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %20 = affine.max affine_map<()[s0] -> (64, s0)>()[%18]
    %21 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%20, %18]
    %22 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%21, %18]
    %23 = affine.max affine_map<()[s0] -> (0, s0)>()[%22]
    %subview = memref.subview %12[0, 0] [%23, 64] [1, 1] : memref<16x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %reinterpret_cast[0, 0] [%23, 64] [1, 1] : memref<16x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_7 = memref.collapse_shape %subview [[0, 1]] : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_8 = memref.collapse_shape %subview_6 [[0, 1]] : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%collapse_shape_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %11]
  }
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
