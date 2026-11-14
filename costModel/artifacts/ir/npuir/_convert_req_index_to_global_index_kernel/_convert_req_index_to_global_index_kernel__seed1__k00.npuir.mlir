// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_convert_req_index_to_global_index_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c73792_i64 = arith.constant 73792 : i64
  %c82048_i64 = arith.constant 82048 : i64
  %c65600_i64 = arith.constant 65600 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c57344_i64 = arith.constant 57344 : i64
  %c49152_i64 = arith.constant 49152 : i64
  %c40960_i64 = arith.constant 40960 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c73856_i64 = arith.constant 73856 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c0_i64 = arith.constant 0 : i64
  %c6_i32 = arith.constant 6 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2048 = arith.constant 2048 : index
  %c63_i32 = arith.constant 63 : i32
  %c0_i32 = arith.constant 0 : i32
  %c11_i32 = arith.constant 11 : i32
  %c-1_i32 = arith.constant -1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<2048xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%c63_i32 : i32) outs(%5 : memref<2048xi32, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg13 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_0, %c1_i64 : i64
    %13 = hivm.hir.pointer_cast(%c65600_i64, %c82048_i64) : memref<2048xi32, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<2048xi32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c8192_i64, %c73856_i64) : memref<2048xi32, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<2048xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %15 = arith.divsi %arg13, %4 : i32
    %16 = arith.remsi %15, %arg10 : i32
    %17 = arith.index_cast %16 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%17], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %18 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %19 = arith.muli %16, %arg8 : i32
    %20 = arith.muli %16, %arg9 : i32
    %21 = arith.muli %18, %arg7 : i32
    %22 = arith.extsi %21 : i32 to i64
    %23 = arith.index_cast %19 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%23], sizes: [2048], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<2048xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %12]
    hivm.hir.load ins(%reinterpret_cast_1 : memref<2048xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%14 : memref<2048xi32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %24 = hivm.hir.pointer_cast(%c16384_i64) : memref<2048xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vshr ins(%14, %c6_i32 : memref<2048xi32, #hivm.address_space<ub>>, i32) outs(%24 : memref<2048xi32, #hivm.address_space<ub>>) round : true
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %25 = hivm.hir.pointer_cast(%c24576_i64) : memref<2048xi32, #hivm.address_space<ub>>
    hivm.hir.vand ins(%14, %5 : memref<2048xi32, #hivm.address_space<ub>>, memref<2048xi32, #hivm.address_space<ub>>) outs(%25 : memref<2048xi32, #hivm.address_space<ub>>)
    %26 = hivm.hir.pointer_cast(%c32768_i64) : memref<2048xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmax ins(%14, %c0_i32 : memref<2048xi32, #hivm.address_space<ub>>, i32) outs(%26 : memref<2048xi32, #hivm.address_space<ub>>)
    %27 = hivm.hir.pointer_cast(%c32768_i64) : memref<2048xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%26, %14 : memref<2048xi32, #hivm.address_space<ub>>, memref<2048xi32, #hivm.address_space<ub>>) outs(%27 : memref<2048xi1, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %12]
    %28 = hivm.hir.pointer_cast(%c40960_i64) : memref<2048xi32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%24, %c11_i32 : memref<2048xi32, #hivm.address_space<ub>>, i32) outs(%28 : memref<2048xi32, #hivm.address_space<ub>>)
    %29 = hivm.hir.pointer_cast(%c40960_i64) : memref<2048xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%28, %24 : memref<2048xi32, #hivm.address_space<ub>>, memref<2048xi32, #hivm.address_space<ub>>) outs(%29 : memref<2048xi1, #hivm.address_space<ub>>)
    %30 = hivm.hir.pointer_cast(%c40960_i64) : memref<2048xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%29 : memref<2048xi1, #hivm.address_space<ub>>) outs(%30 : memref<2048xi1, #hivm.address_space<ub>>)
    %31 = hivm.hir.pointer_cast(%c32768_i64) : memref<2048xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vand ins(%27, %30 : memref<2048xi1, #hivm.address_space<ub>>, memref<2048xi1, #hivm.address_space<ub>>) outs(%31 : memref<2048xi1, #hivm.address_space<ub>>)
    %32 = hivm.hir.pointer_cast(%c49152_i64) : memref<2048xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg14 = %c0 to %c2048 step %c1 {
      %39 = memref.load %24[%arg14] {DiscreteMemAccess} : memref<2048xi32, #hivm.address_space<ub>>
      %40 = arith.extsi %39 : i32 to i64
      %41 = arith.addi %22, %40 : i64
      %42 = arith.index_cast %41 : i64 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%42], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %43 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.store %43, %32[%arg14] : memref<2048xi32, #hivm.address_space<ub>>
    } {ExtractedLoadOrStore}
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %33 = hivm.hir.pointer_cast(%c57344_i64) : memref<2048xi32, #hivm.address_space<ub>>
    %34 = hivm.hir.pointer_cast(%c65536_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%31, %32, %c0_i32 : memref<2048xi1, #hivm.address_space<ub>>, memref<2048xi32, #hivm.address_space<ub>>, i32) outs(%33 : memref<2048xi32, #hivm.address_space<ub>>) temp_buffer(%34 : memref<16xi32, #hivm.address_space<ub>>)
    %35 = hivm.hir.pointer_cast(%c57344_i64) : memref<2048xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vshl ins(%33, %c6_i32 : memref<2048xi32, #hivm.address_space<ub>>, i32) outs(%35 : memref<2048xi32, #hivm.address_space<ub>>)
    %36 = hivm.hir.pointer_cast(%c57344_i64) : memref<2048xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vor ins(%35, %25 : memref<2048xi32, #hivm.address_space<ub>>, memref<2048xi32, #hivm.address_space<ub>>) outs(%36 : memref<2048xi32, #hivm.address_space<ub>>)
    %37 = hivm.hir.pointer_cast(%c73792_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %12]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%31, %36, %c-1_i32 : memref<2048xi1, #hivm.address_space<ub>>, memref<2048xi32, #hivm.address_space<ub>>, i32) outs(%13 : memref<2048xi32, #hivm.address_space<ub>>) temp_buffer(%37 : memref<16xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %38 = arith.index_cast %20 : i32 to index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%38], sizes: [2048], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<2048xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%13 : memref<2048xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<2048xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %12]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
