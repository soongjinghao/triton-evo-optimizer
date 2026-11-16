// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage3(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c160_i64 = arith.constant 160 : i64
  %c128_i64 = arith.constant 128 : i64
  %c96_i64 = arith.constant 96 : i64
  %c64_i64 = arith.constant 64 : i64
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %c-1_i32 = arith.constant -1 : i32
  %cst = arith.constant 3.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c32_i32 = arith.constant 32 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [16], sizes: [4], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<4xi32, strided<[1], offset: 16>, #hivm.address_space<gm>>
  %2 = hivm.hir.pointer_cast(%c0_i64) : memref<4xi32, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast : memref<4xi32, strided<[1], offset: 16>, #hivm.address_space<gm>>) outs(%2 : memref<4xi32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  %3 = hivm.hir.pointer_cast(%c0_i64) : memref<4xi32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vadd ins(%2, %c32_i32 : memref<4xi32, #hivm.address_space<ub>>, i32) outs(%3 : memref<4xi32, #hivm.address_space<ub>>)
  %4 = hivm.hir.pointer_cast(%c0_i64) : memref<4xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%3, %c-1_i32 : memref<4xi32, #hivm.address_space<ub>>, i32) outs(%4 : memref<4xi32, #hivm.address_space<ub>>)
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<4xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%4 : memref<4xi32, #hivm.address_space<ub>>) outs(%5 : memref<4xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c0_i64) : memref<4xf32, #hivm.address_space<ub>>
  %7 = hivm.hir.pointer_cast(%c32_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vdiv ins(%5, %cst : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%6 : memref<4xf32, #hivm.address_space<ub>>) temp_buffer(%7 : memref<8xf32, #hivm.address_space<ub>>)
  %8 = hivm.hir.pointer_cast(%c0_i64) : memref<4xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%6 : memref<4xf32, #hivm.address_space<ub>>) outs(%8 : memref<4xi32, #hivm.address_space<ub>>) round_mode = <trunc>
  %9 = hivm.hir.pointer_cast(%c0_i64) : memref<4xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmul ins(%8, %c32_i32 : memref<4xi32, #hivm.address_space<ub>>, i32) outs(%9 : memref<4xi32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  %10 = hivm.hir.pointer_cast(%c64_i64) : memref<4xi32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  %11 = memref.load %9[%c0] : memref<4xi32, #hivm.address_space<ub>>
  memref.store %11, %10[%c0] : memref<4xi32, #hivm.address_space<ub>>
  scf.for %arg9 = %c1 to %c4 step %c1 {
    %16 = memref.load %9[%arg9] : memref<4xi32, #hivm.address_space<ub>>
    %17 = arith.subi %arg9, %c1 : index
    %18 = memref.load %10[%17] : memref<4xi32, #hivm.address_space<ub>>
    %19 = arith.addi %16, %18 : i32
    memref.store %19, %10[%arg9] : memref<4xi32, #hivm.address_space<ub>>
  }
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [1], sizes: [4], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<4xi32, strided<[1], offset: 1>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%10 : memref<4xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_0 : memref<4xi32, strided<[1], offset: 1>, #hivm.address_space<gm>>)
  %12 = hivm.hir.pointer_cast(%c96_i64) : memref<1xi32, #hivm.address_space<ub>>
  %13 = hivm.hir.pointer_cast(%c128_i64) : memref<8xi32, #hivm.address_space<ub>>
  hivm.hir.vreduce <sum> ins(%9 : memref<4xi32, #hivm.address_space<ub>>) outs(%12 : memref<1xi32, #hivm.address_space<ub>>) temp_buffer(%13 : memref<8xi32, #hivm.address_space<ub>>) reduce_dims = [0]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  %14 = memref.load %12[%c0] : memref<1xi32, #hivm.address_space<ub>>
  %15 = hivm.hir.pointer_cast(%c160_i64) : memref<1xi32, #hivm.address_space<ub>>
  memref.store %14, %15[%c0] : memref<1xi32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
  %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1]>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%15 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_1 : memref<1xi32, strided<[1]>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
