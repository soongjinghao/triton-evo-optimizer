// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_copy_page_indices_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.index_cast %arg5 : i32 to index
  %6 = affine.max affine_map<()[s0] -> (0, s0)>()[%5]
  %7 = affine.min affine_map<()[s0] -> (256, s0)>()[%6]
  %8 = arith.cmpi slt, %7, %c256 : index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %9 = arith.index_cast %arg10 : i32 to index
    %10 = arith.index_cast %3 : i32 to index
    %11 = arith.index_cast %1 : i32 to index
    %12 = arith.index_cast %c48_i32 : i32 to index
    %13 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%9, %10, %12]
    %14 = arith.index_cast %13 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %15 = arith.select %14, %c0_i64_0, %c1_i64 : i64
    %16 = hivm.hir.pointer_cast(%c0_i64, %c1024_i64) : memref<256xi32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<256xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %17 = arith.divsi %arg10, %4 : i32
    %18 = arith.remsi %17, %arg7 : i32
    %19 = arith.muli %18, %arg5 : i32
    %20 = arith.index_cast %19 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%20], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview = memref.subview %reinterpret_cast[0] [%7] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_1 = memref.subview %16[0] [%7] [1] : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %8 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%c0_i32 : i32) outs(%16 : memref<256xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
    hivm.hir.load ins(%subview : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %21 = arith.index_cast %18 : i32 to index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%21], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %22 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %23 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%21]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%23], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %24 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %25 = arith.subi %24, %22 : i32
    %26 = arith.index_cast %22 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %27 = arith.index_cast %25 : i32 to index
    %28 = affine.max affine_map<()[s0] -> (0, s0)>()[%27]
    %29 = affine.min affine_map<()[s0] -> (256, s0)>()[%28]
    %subview_5 = memref.subview %16[0] [%29] [1] : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %reinterpret_cast_4[0] [%29] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_5 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
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
