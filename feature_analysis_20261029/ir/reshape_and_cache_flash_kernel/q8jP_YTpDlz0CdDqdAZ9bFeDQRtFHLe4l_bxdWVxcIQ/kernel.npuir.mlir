// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @reshape_and_cache_flash_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: f32, %arg15: f32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false]> : vector<19xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c512_i32 = arith.constant 512 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg16, %arg17 : i32
  %1 = arith.muli %0, %arg18 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg18, %arg17 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg16 : i32
  %7 = arith.index_cast %6 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%7], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %8 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %9 = arith.cmpi slt, %8, %c0_i32 : i32
  scf.if %9 {
  } else {
    %10 = arith.divsi %8, %arg13 : i32
    %11 = arith.remsi %8, %arg13 : i32
    %12 = arith.muli %10, %arg8 : i32
    %13 = arith.muli %11, %c512_i32 : i32
    %14 = arith.addi %12, %13 : i32
    %15 = arith.muli %6, %arg9 : i32
    %16 = arith.index_cast %15 : i32 to index
    %17 = arith.muli %6, %arg10 : i32
    %18 = arith.index_cast %17 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [4096], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %19 = hivm.hir.pointer_cast(%c0_i64) : memref<4096xf32, #hivm.address_space<ub>>
    %subview = memref.subview %reinterpret_cast_0[0] [512] [1] : memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_1 = memref.subview %19[0] [512] [1] : memref<4096xf32, #hivm.address_space<ub>> to memref<512xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%19 : memref<4096xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%subview : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<512xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%18], sizes: [4096], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %20 = hivm.hir.pointer_cast(%c16384_i64) : memref<4096xf32, #hivm.address_space<ub>>
    %subview_3 = memref.subview %reinterpret_cast_2[0] [512] [1] : memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %20[0] [512] [1] : memref<4096xf32, #hivm.address_space<ub>> to memref<512xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%20 : memref<4096xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%subview_3 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<512xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %21 = arith.index_cast %14 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%21], sizes: [4096], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [512] [1] : memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.store ins(%subview_1 : memref<512xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%21], sizes: [4096], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_8 = memref.subview %reinterpret_cast_7[0] [512] [1] : memref<4096xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_4 : memref<512xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  }
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
