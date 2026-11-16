// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_per_group_transpose(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1536_i64 = arith.constant 1536 : i64
  %c512_i64 = arith.constant 512 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %c16_i32 = arith.constant 16 : i32
  %c8_i32 = arith.constant 8 : i32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.index_cast %arg6 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg10 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %12 = arith.select %11, %c1_i64, %c2_i64 : i64
    %13 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64_1 = arith.constant 1 : i64
    %14 = arith.select %13, %c0_i64_0, %c1_i64_1 : i64
    %15 = hivm.hir.pointer_cast(%c512_i64, %c1536_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<8x16xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c0_i64, %c1024_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<16x8xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %17 = arith.divsi %arg10, %4 : i32
    %18 = arith.remsi %17, %arg7 : i32
    %19 = arith.remsi %18, %arg7 : i32
    %20 = arith.divsi %18, %arg7 : i32
    %21 = arith.remsi %20, %arg8 : i32
    %22 = arith.divsi %18, %0 : i32
    %23 = arith.index_cast %19 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%23], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %24 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %25 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%23]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%25], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %26 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %27 = arith.subi %26, %24 : i32
    %28 = arith.index_cast %27 : i32 to index
    %29 = arith.muli %24, %arg6 : i32
    %30 = arith.index_cast %29 : i32 to index
    %31 = arith.muli %22, %c8_i32 : i32
    %32 = arith.index_cast %31 : i32 to index
    %33 = arith.muli %21, %c16_i32 : i32
    %34 = arith.index_cast %33 : i32 to index
    %35 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>()[%32, %30, %34, %5]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [16, 8], strides: [%5, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %36 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%34, %28]
    %37 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%36, %34]
    %38 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%32, %5]
    %39 = affine.min affine_map<()[s0, s1] -> (s1 + 8, s0)>()[%38, %32]
    %40 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%37, %34]
    %41 = affine.max affine_map<()[s0] -> (0, s0)>()[%40]
    %42 = affine.min affine_map<()[s0, s1] -> (8, s0 - s1)>()[%39, %32]
    %43 = affine.max affine_map<()[s0] -> (0, s0)>()[%42]
    %44 = arith.cmpi slt, %41, %c16 : index
    %45 = arith.cmpi slt, %43, %c8 : index
    %46 = arith.ori %44, %45 : i1
    %subview = memref.subview %reinterpret_cast_3[0, 0] [%41, %43] [1, 1] : memref<16x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %16[0, 0] [%41, %43] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    scf.if %46 {
      %collapse_shape = memref.collapse_shape %16 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %12]
    hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %47 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>()[%30, %34, %32, %28]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [8, 16], strides: [%28, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %14]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vtranspose ins(%16 : memref<16x8xf32, #hivm.address_space<ub>>) outs(%15 : memref<8x16xf32, #hivm.address_space<ub>>) permutation = [1, 0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %12]
    %subview_6 = memref.subview %15[0, 0] [%43, %41] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%43, %41] [1, 1] : memref<8x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_6 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %14]
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
