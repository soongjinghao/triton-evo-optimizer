// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage2(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, false, false, false]> : vector<7xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c160_i64 = arith.constant 160 : i64
  %c0_i64 = arith.constant 0 : i64
  %c128_i64 = arith.constant 128 : i64
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg4, %arg5 : i32
  %1 = arith.muli %0, %arg6 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg6, %arg5 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg7 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg7, %4 : i32
    %6 = arith.remsi %5, %arg4 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = affine.apply affine_map<()[s0] -> (s0 + 4)>()[%7]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [4], strides: [4] : memref<?xi32, #hivm.address_space<gm>> to memref<4xi32, strided<[4], offset: ?>, #hivm.address_space<gm>>
    %9 = hivm.hir.pointer_cast(%c128_i64) : memref<4xi32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8xi32, #hivm.address_space<ub>>
    %subview = memref.subview %10[0, 0] [4, 1] [1, 1] : memref<4x8xi32, #hivm.address_space<ub>> to memref<4xi32, strided<[8]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%reinterpret_cast : memref<4xi32, strided<[4], offset: ?>, #hivm.address_space<gm>>) outs(%subview : memref<4xi32, strided<[8]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vdeinterleave ins(%subview : memref<4xi32, strided<[8]>, #hivm.address_space<ub>>) outs(%9 : memref<4xi32, #hivm.address_space<ub>>) channel_num = 8 index_mode = <CHANNEL_0>
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %11 = hivm.hir.pointer_cast(%c160_i64) : memref<4x8xi32, #hivm.address_space<ub>>
    %subview_0 = memref.subview %11[0, 0] [4, 1] [1, 1] : memref<4x8xi32, #hivm.address_space<ub>> to memref<4xi32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %9 : memref<4xi32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_1 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [4, 1], strides: [1, 1] : memref<i32, #hivm.address_space<ub>> to memref<4x1xi32, strided<[1, 1]>, #hivm.address_space<ub>>
    %base_buffer_2, %offset_3, %sizes_4, %strides_5 = memref.extract_strided_metadata %subview_0 : memref<4xi32, strided<[8]>, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_6 = memref.reinterpret_cast %base_buffer_2 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<i32, #hivm.address_space<ub>> to memref<4x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg8 = %c0 to %c1 step %c1 {
      %12 = memref.load %reinterpret_cast_1[%c0, %arg8] : memref<4x1xi32, strided<[1, 1]>, #hivm.address_space<ub>>
      memref.store %12, %reinterpret_cast_6[%c0, %arg8] : memref<4x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.for %arg9 = %c1 to %c4 step %c1 {
        %13 = memref.load %reinterpret_cast_1[%arg9, %arg8] : memref<4x1xi32, strided<[1, 1]>, #hivm.address_space<ub>>
        %14 = arith.subi %arg9, %c1 : index
        %15 = memref.load %reinterpret_cast_6[%14, %arg8] : memref<4x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>
        %16 = arith.addi %13, %15 : i32
        memref.store %16, %reinterpret_cast_6[%arg9, %arg8] : memref<4x1xi32, strided<[8, 1]>, #hivm.address_space<ub>>
      }
    }
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_0 : memref<4xi32, strided<[8]>, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<4xi32, strided<[4], offset: ?>, #hivm.address_space<gm>>)
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
