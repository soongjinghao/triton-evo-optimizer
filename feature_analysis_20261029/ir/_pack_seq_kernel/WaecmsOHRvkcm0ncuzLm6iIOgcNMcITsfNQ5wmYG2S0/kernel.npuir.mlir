// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_pack_seq_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c32_i32 = arith.constant 32 : i32
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %cst = arith.constant 0.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.remsi %3, %arg8 : i32
  %5 = arith.divsi %3, %arg8 : i32
  %6 = arith.remsi %5, %arg7 : i32
  %7 = arith.muli %arg8, %arg7 : i32
  %8 = arith.divsi %3, %7 : i32
  %9 = arith.remsi %8, %arg6 : i32
  %10 = hivm.hir.pointer_cast(%c0_i64) : memref<32x32xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %10 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<1024xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  %11 = arith.muli %6, %c32_i32 : i32
  %12 = arith.muli %4, %c32_i32 : i32
  %13 = scf.for %arg9 = %c0_i32 to %9 step %c1_i32 iter_args(%arg10 = %c0_i32) -> (i32)  : i32 {
    %37 = arith.index_cast %arg9 : i32 to index
    %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %38 = memref.load %reinterpret_cast_6[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %39 = arith.addi %arg10, %38 : i32
    scf.yield %39 : i32
  }
  %14 = arith.index_cast %9 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %15 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %16 = arith.index_cast %13 : i32 to index
  %17 = arith.index_cast %11 : i32 to index
  %18 = arith.index_cast %12 : i32 to index
  %19 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 4 + s2 * 4)>()[%18, %16, %17]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%19], sizes: [32, 32], strides: [4, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
  %20 = arith.muli %9, %c4_i32 : i32
  %21 = arith.index_cast %20 : i32 to index
  %22 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 4 + s2 * 4)>()[%18, %21, %17]
  %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%22], sizes: [32, 32], strides: [4, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
  %23 = affine.max affine_map<()[s0] -> (4, s0)>()[%17]
  %24 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%23, %17]
  %25 = affine.max affine_map<()[s0] -> (4, s0)>()[%18]
  %26 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%25, %18]
  %27 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%24, %17]
  %28 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%26, %18]
  %subview = memref.subview %10[0, 0] [%27, %28] [1, 1] : memref<32x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>
  %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%27, %28] [1, 1] : memref<32x32xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%subview : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>)
  %29 = hivm.hir.pointer_cast(%c4096_i64) : memref<32x32xf32, #hivm.address_space<ub>>
  %30 = arith.index_cast %15 : i32 to index
  %31 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%17, %30]
  %32 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%31, %17]
  %33 = affine.min affine_map<()[s0, s1, s2] -> (-s1 + s2, 32, s0 - s1)>()[%32, %17, %24]
  %34 = arith.cmpi slt, %33, %c32 : index
  %35 = arith.cmpi slt, %28, %c32 : index
  %36 = arith.ori %34, %35 : i1
  %subview_3 = memref.subview %reinterpret_cast_0[0, 0] [%33, %28] [1, 1] : memref<32x32xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_4 = memref.subview %29[0, 0] [%33, %28] [1, 1] : memref<32x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>
  scf.if %36 {
    %collapse_shape_6 = memref.collapse_shape %29 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_6 : memref<1024xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview_3 : memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  %subview_5 = memref.subview %reinterpret_cast_1[0, 0] [%33, %28] [1, 1] : memref<32x32xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_MTE3>]
  hivm.hir.store ins(%subview_4 : memref<?x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<?x?xf32, strided<[4, 1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
