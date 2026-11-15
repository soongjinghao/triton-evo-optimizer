// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @copy_all_layer_kv_cache_tiled(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %c64_i32 = arith.constant 64 : i32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c0_i8 = arith.constant 0 : i8
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %5 = hivm.hir.pointer_cast(%c0_i64, %c64_i64) : memref<64xi8, #hivm.address_space<ub>>
    annotation.mark %5 {hivm.multi_buffer = 2 : i32} : memref<64xi8, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %6 = arith.remsi %arg11, %arg10 : i32
    %7 = arith.divsi %arg11, %arg10 : i32
    %8 = arith.remsi %7, %arg9 : i32
    %9 = arith.divsi %arg11, %4 : i32
    %10 = arith.remsi %9, %arg8 : i32
    %11 = arith.index_cast %10 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%11], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %12 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%11], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %13 = memref.load %reinterpret_cast_0[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %14 = arith.muli %8, %c64_i32 : i32
    %15 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%15], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %16 = memref.load %reinterpret_cast_1[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%15], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %17 = memref.load %reinterpret_cast_2[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %18 = arith.muli %16, %12 : i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = arith.index_cast %14 : i32 to index
    %21 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%19, %20]
    %22 = arith.index_cast %21 : index to i64
    %23 = arith.addi %13, %22 : i64
    %24 = hivm.hir.pointer_cast(%23) [%c64] : memref<?xi8, #hivm.address_space<gm>>
    annotation.mark %24 {address_space = #hivm.address_space<gm>} : memref<?xi8, #hivm.address_space<gm>>
    %reinterpret_cast_3 = memref.reinterpret_cast %24 to offset: [0], sizes: [64], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %25 = arith.muli %17, %12 : i64
    %26 = arith.index_cast %25 : i64 to index
    %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%26, %20]
    %28 = arith.index_cast %27 : index to i64
    %29 = arith.addi %13, %28 : i64
    %30 = hivm.hir.pointer_cast(%29) [%c64] : memref<?xi8, #hivm.address_space<gm>>
    annotation.mark %30 {address_space = #hivm.address_space<gm>} : memref<?xi8, #hivm.address_space<gm>>
    %reinterpret_cast_4 = memref.reinterpret_cast %30 to offset: [0], sizes: [64], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %31 = arith.index_cast %12 : i64 to index
    %32 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%20, %31]
    %33 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%32, %20]
    %34 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%33, %20]
    %35 = arith.cmpi slt, %34, %c64 : index
    %subview = memref.subview %reinterpret_cast_3[0] [%34] [1] : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %5[0] [%34] [1] : memref<64xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[1]>, #hivm.address_space<ub>>
    scf.if %35 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%c0_i8 : i8) outs(%5 : memref<64xi8, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%subview : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i8 : i8 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_6 = memref.subview %reinterpret_cast_4[0] [%34] [1] : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_5 : memref<?xi8, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>)
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
