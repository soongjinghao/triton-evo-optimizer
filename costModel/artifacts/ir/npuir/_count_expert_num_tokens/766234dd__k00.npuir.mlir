// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_count_expert_num_tokens(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c16832_i64 = arith.constant 16832 : i64
  %c12704_i64 = arith.constant 12704 : i64
  %c12416_i64 = arith.constant 12416 : i64
  %c12384_i64 = arith.constant 12384 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c12736_i64 = arith.constant 12736 : i64
  %c0_i64 = arith.constant 0 : i64
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %c-1_i32 = arith.constant -1 : i32
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg7, %arg6 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  scf.for %arg8 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg8 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = hivm.hir.pointer_cast(%c12704_i64, %c16832_i64) : memref<1xi32, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c0_i64, %c12736_i64) : memref<1024xi32, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<1024xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %14 = arith.divsi %arg8, %4 : i32
    %15 = arith.remsi %14, %arg5 : i32
    %16 = arith.cmpi sge, %15, %c4_i32 : i32
    scf.if %16 {
    } else {
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1024xi32, strided<[1]>, #hivm.address_space<gm>>
      %subview = memref.subview %reinterpret_cast[0] [1000] [1] : memref<1024xi32, strided<[1]>, #hivm.address_space<gm>> to memref<1000xi32, strided<[1]>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %13[0] [1000] [1] : memref<1024xi32, #hivm.address_space<ub>> to memref<1000xi32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%c-1_i32 : i32) outs(%13 : memref<1024xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%subview : memref<1000xi32, strided<[1]>, #hivm.address_space<gm>>) outs(%subview_1 : memref<1000xi32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c-1_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %17 = hivm.hir.pointer_cast(%c4096_i64) : memref<1024xi32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%15 : i32) outs(%17 : memref<1024xi32, #hivm.address_space<ub>>)
      %18 = hivm.hir.pointer_cast(%c0_i64, %c12736_i64) : memref<1024xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%13, %17 : memref<1024xi32, #hivm.address_space<ub>>, memref<1024xi32, #hivm.address_space<ub>>) outs(%18 : memref<1024xi1, #hivm.address_space<ub>>)
      %19 = hivm.hir.pointer_cast(%c8192_i64) : memref<1024xi32, #hivm.address_space<ub>>
      %20 = hivm.hir.pointer_cast(%c12288_i64) : memref<24xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%18, %c1_i32, %c0_i32 : memref<1024xi1, #hivm.address_space<ub>>, i32, i32) outs(%19 : memref<1024xi32, #hivm.address_space<ub>>) temp_buffer(%20 : memref<24xi32, #hivm.address_space<ub>>)
      %21 = hivm.hir.pointer_cast(%c12384_i64) : memref<1xi32, #hivm.address_space<ub>>
      %22 = hivm.hir.pointer_cast(%c12416_i64) : memref<72xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%19 : memref<1024xi32, #hivm.address_space<ub>>) outs(%21 : memref<1xi32, #hivm.address_space<ub>>) temp_buffer(%22 : memref<72xi32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %23 = memref.load %21[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %11]
      memref.store %23, %12[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %24 = arith.index_cast %15 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%24], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%12 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %11]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
