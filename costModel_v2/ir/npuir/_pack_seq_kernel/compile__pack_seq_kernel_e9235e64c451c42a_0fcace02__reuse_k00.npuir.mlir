// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_pack_seq_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c16_i32 = arith.constant 16 : i32
  %c4_i32 = arith.constant 4 : i32
  %c256_i32 = arith.constant 256 : i32
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg10, %arg9 : i32
    %6 = arith.remsi %5, %arg8 : i32
    %7 = arith.divsi %arg10, %4 : i32
    %8 = arith.remsi %7, %arg7 : i32
    %9 = arith.index_cast %6 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%9], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %10 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg6 to offset: [%9], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %11 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %12 = arith.muli %8, %c256_i32 : i32
    %13 = arith.muli %10, %c4_i32 : i32
    %14 = arith.muli %6, %c16_i32 : i32
    %15 = arith.index_cast %13 : i32 to index
    %16 = arith.index_cast %12 : i32 to index
    %17 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%15, %16]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%17], sizes: [256], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %18 = arith.index_cast %14 : i32 to index
    %19 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%18, %16]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%19], sizes: [256], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %20 = arith.muli %11, %c4_i32 : i32
    %21 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>
    %22 = arith.index_cast %20 : i32 to index
    %23 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%16, %22]
    %24 = affine.min affine_map<()[s0, s1] -> (s1 + 256, s0)>()[%23, %16]
    %25 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%24, %16]
    %26 = arith.cmpi slt, %25, %c256 : index
    %subview = memref.subview %reinterpret_cast_1[0] [%25] [1] : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %21[0] [%25] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %26 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%21 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %27 = affine.max affine_map<()[s0] -> (16, s0)>()[%16]
    %28 = affine.min affine_map<()[s0, s1] -> (s1 + 256, s0)>()[%27, %16]
    %29 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%28, %16]
    %subview_4 = memref.subview %21[0] [%29] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_5 = memref.subview %reinterpret_cast_2[0] [%29] [1] : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
