// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_copy_page_indices_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c256_i32 = arith.constant 256 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg7 : i32
  %7 = arith.muli %6, %arg5 : i32
  %8 = arith.index_cast %7 : i32 to index
  %9 = arith.index_cast %6 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%9], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %10 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %11 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%9]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg6 to offset: [%11], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %12 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %13 = arith.subi %12, %10 : i32
  %14 = arith.index_cast %10 : i32 to index
  %15 = arith.index_cast %13 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg10 = %c0_i32 to %13 step %c256_i32  : i32 {
    %16 = arith.index_cast %arg10 : i32 to index
    %17 = arith.index_cast %c0_i32 : i32 to index
    %18 = arith.index_cast %13 : i32 to index
    %19 = arith.index_cast %c256_i32 : i32 to index
    %20 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%16, %17, %19]
    %21 = arith.index_cast %20 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %22 = arith.select %21, %c0_i64_1, %c1_i64 : i64
    %23 = hivm.hir.pointer_cast(%c0_i64, %c1024_i64) : memref<256xi32, #hivm.address_space<ub>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<256xi32, #hivm.address_space<ub>>
    %24 = arith.index_cast %arg10 : i32 to index
    %25 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%8, %24]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%25], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %26 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%24, %15]
    %27 = affine.min affine_map<()[s0, s1] -> (s1 + 256, s0)>()[%26, %24]
    %28 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%27, %24]
    %29 = arith.cmpi slt, %28, %c256 : index
    %subview = memref.subview %reinterpret_cast_2[0] [%28] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %23[0] [%28] [1] : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %29 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%c0_i32 : i32) outs(%23 : memref<256xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %22]
    hivm.hir.load ins(%subview : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%14, %24]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %reinterpret_cast_4[0] [%28] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_3 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %22]
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
