// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_count_expert_num_tokens(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c12704_i64 = arith.constant 12704 : i64
  %c12416_i64 = arith.constant 12416 : i64
  %c12384_i64 = arith.constant 12384 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c-1_i32 = arith.constant -1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg5, %arg6 : i32
  %1 = arith.muli %0, %arg7 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg7, %arg6 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg5 : i32
  %7 = arith.cmpi sge, %6, %c4_i32 : i32
  scf.if %7 {
  } else {
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1024xi32, strided<[1]>, #hivm.address_space<gm>>
    %8 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xi32, #hivm.address_space<ub>>
    %subview = memref.subview %reinterpret_cast[0] [1000] [1] : memref<1024xi32, strided<[1]>, #hivm.address_space<gm>> to memref<1000xi32, strided<[1]>, #hivm.address_space<gm>>
    %subview_0 = memref.subview %8[0] [1000] [1] : memref<1024xi32, #hivm.address_space<ub>> to memref<1000xi32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%c-1_i32 : i32) outs(%8 : memref<1024xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%subview : memref<1000xi32, strided<[1]>, #hivm.address_space<gm>>) outs(%subview_0 : memref<1000xi32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c-1_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %9 = hivm.hir.pointer_cast(%c4096_i64) : memref<1024xi32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%6 : i32) outs(%9 : memref<1024xi32, #hivm.address_space<ub>>)
    %10 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%8, %9 : memref<1024xi32, #hivm.address_space<ub>>, memref<1024xi32, #hivm.address_space<ub>>) outs(%10 : memref<1024xi1, #hivm.address_space<ub>>)
    %11 = hivm.hir.pointer_cast(%c8192_i64) : memref<1024xi32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c12288_i64) : memref<24xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%10, %c1_i32, %c0_i32 : memref<1024xi1, #hivm.address_space<ub>>, i32, i32) outs(%11 : memref<1024xi32, #hivm.address_space<ub>>) temp_buffer(%12 : memref<24xi32, #hivm.address_space<ub>>)
    %13 = hivm.hir.pointer_cast(%c12384_i64) : memref<1xi32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c12416_i64) : memref<72xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%11 : memref<1024xi32, #hivm.address_space<ub>>) outs(%13 : memref<1xi32, #hivm.address_space<ub>>) temp_buffer(%14 : memref<72xi32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %15 = memref.load %13[%c0] : memref<1xi32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c12704_i64) : memref<1xi32, #hivm.address_space<ub>>
    memref.store %15, %16[%c0] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %17 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%16 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_1 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) atomic = <add>
  }
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
