// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_count_expert_num_tokens(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c448_i64 = arith.constant 448 : i64
  %c384_i64 = arith.constant 384 : i64
  %c352_i64 = arith.constant 352 : i64
  %c256_i64 = arith.constant 256 : i64
  %c224_i64 = arith.constant 224 : i64
  %c192_i64 = arith.constant 192 : i64
  %c128_i64 = arith.constant 128 : i64
  %c96_i64 = arith.constant 96 : i64
  %c64_i64 = arith.constant 64 : i64
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c-1_i32 = arith.constant -1 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg6 : i32
  %7 = arith.cmpi sge, %6, %c4_i32 : i32
  scf.if %7 {
  } else {
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<8xi32, strided<[1]>, #hivm.address_space<gm>>
    %8 = hivm.hir.pointer_cast(%c0_i64) : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast : memref<8xi32, strided<[1]>, #hivm.address_space<gm>>) outs(%8 : memref<8xi32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    %9 = hivm.hir.pointer_cast(%c32_i64) : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmax ins(%8, %c0_i32 : memref<8xi32, #hivm.address_space<ub>>, i32) outs(%9 : memref<8xi32, #hivm.address_space<ub>>)
    %10 = hivm.hir.pointer_cast(%c32_i64) : memref<8xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%9, %8 : memref<8xi32, #hivm.address_space<ub>>, memref<8xi32, #hivm.address_space<ub>>) outs(%10 : memref<8xi1, #hivm.address_space<ub>>)
    %11 = hivm.hir.pointer_cast(%c64_i64) : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg9 = %c0 to %c8 step %c1 {
      %23 = memref.load %8[%arg9] {DiscreteMemAccess} : memref<8xi32, #hivm.address_space<ub>>
      %24 = arith.index_cast %23 : i32 to index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%24], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %25 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.store %25, %11[%arg9] : memref<8xi32, #hivm.address_space<ub>>
    } {ExtractedLoadOrStore}
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %12 = hivm.hir.pointer_cast(%c96_i64) : memref<8xi32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c128_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%10, %11, %c-1_i32 : memref<8xi1, #hivm.address_space<ub>>, memref<8xi32, #hivm.address_space<ub>>, i32) outs(%12 : memref<8xi32, #hivm.address_space<ub>>) temp_buffer(%13 : memref<16xi32, #hivm.address_space<ub>>)
    %14 = hivm.hir.pointer_cast(%c192_i64) : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%6 : i32) outs(%14 : memref<8xi32, #hivm.address_space<ub>>)
    %15 = hivm.hir.pointer_cast(%c96_i64) : memref<8xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%12, %14 : memref<8xi32, #hivm.address_space<ub>>, memref<8xi32, #hivm.address_space<ub>>) outs(%15 : memref<8xi1, #hivm.address_space<ub>>)
    %16 = hivm.hir.pointer_cast(%c224_i64) : memref<8xi32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c256_i64) : memref<24xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%15, %c1_i32, %c0_i32 : memref<8xi1, #hivm.address_space<ub>>, i32, i32) outs(%16 : memref<8xi32, #hivm.address_space<ub>>) temp_buffer(%17 : memref<24xi32, #hivm.address_space<ub>>)
    %18 = hivm.hir.pointer_cast(%c352_i64) : memref<1xi32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c384_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%16 : memref<8xi32, #hivm.address_space<ub>>) outs(%18 : memref<1xi32, #hivm.address_space<ub>>) temp_buffer(%19 : memref<16xi32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %20 = memref.load %18[%c0] : memref<1xi32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c448_i64) : memref<1xi32, #hivm.address_space<ub>>
    memref.store %20, %21[%c0] : memref<1xi32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %22 = arith.index_cast %6 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%22], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%21 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_0 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) atomic = <add>
  }
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
