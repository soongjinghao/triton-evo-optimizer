// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_unpack_seq_triton_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64 = arith.constant 0 : i64
  %c4096_i32 = arith.constant 4096 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg9 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %13 = arith.remsi %arg9, %arg8 : i32
    %14 = arith.divsi %arg9, %arg8 : i32
    %15 = arith.remsi %14, %arg7 : i32
    %16 = arith.divsi %arg9, %4 : i32
    %17 = arith.remsi %16, %arg6 : i32
    %18 = arith.muli %15, %c64_i32 : i32
    %19 = arith.muli %13, %c64_i32 : i32
    %20 = scf.for %arg10 = %c0_i32 to %17 step %c1_i32 iter_args(%arg11 = %c0_i32) -> (i32)  : i32 {
      %46 = arith.index_cast %arg10 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %47 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %48 = arith.addi %arg11, %47 : i32
      scf.yield %48 : i32
    }
    %21 = arith.index_cast %17 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%21], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %22 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %23 = arith.muli %17, %c4096_i32 : i32
    %24 = arith.index_cast %23 : i32 to index
    %25 = arith.index_cast %18 : i32 to index
    %26 = arith.index_cast %19 : i32 to index
    %27 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 64)>()[%26, %24, %25]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [64, 64], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %28 = arith.index_cast %20 : i32 to index
    %29 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 64)>()[%26, %28, %25]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%29], sizes: [64, 64], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %30 = arith.index_cast %22 : i32 to index
    %31 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%25, %30]
    %32 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%31, %25]
    %33 = affine.max affine_map<()[s0] -> (4096, s0)>()[%25]
    %34 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%33, %25]
    %35 = affine.min affine_map<()[s0, s1, s2] -> (-s1 + s2, s0 - s1)>()[%32, %25, %34]
    %36 = affine.max affine_map<()[s0] -> (0, s0)>()[%35]
    %37 = affine.max affine_map<()[s0] -> (64, s0)>()[%26]
    %38 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%37, %26]
    %39 = affine.min affine_map<()[s0] -> (64, s0)>()[%36]
    %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
    %41 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%38, %26]
    %42 = affine.max affine_map<()[s0] -> (0, s0)>()[%41]
    %43 = arith.cmpi slt, %40, %c64 : index
    %44 = arith.cmpi slt, %42, %c64 : index
    %45 = arith.ori %43, %44 : i1
    %subview = memref.subview %reinterpret_cast_1[0, 0] [%40, %42] [1, 1] : memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %12[0, 0] [%40, %42] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    scf.if %45 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape = memref.collapse_shape %12 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_4 = memref.subview %reinterpret_cast_2[0, 0] [%40, %42] [1, 1] : memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_3 : memref<?x?xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
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
