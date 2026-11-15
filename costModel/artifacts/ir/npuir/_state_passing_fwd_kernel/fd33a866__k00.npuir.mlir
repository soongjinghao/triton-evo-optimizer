// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_state_passing_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg9: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c160_i64 = arith.constant 160 : i64
  %c128_i64 = arith.constant 128 : i64
  %c416_i64 = arith.constant 416 : i64
  %c0_i64 = arith.constant 0 : i64
  %c288_i64 = arith.constant 288 : i64
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c8_i64 = arith.constant 8 : i64
  %c7_i64 = arith.constant 7 : i64
  %c32_i32 = arith.constant 32 : i32
  %c384_i32 = arith.constant 384 : i32
  %c96_i32 = arith.constant 96 : i32
  %c3_i32 = arith.constant 3 : i32
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.remsi %3, %arg12 : i32
  %5 = arith.divsi %3, %arg12 : i32
  %6 = arith.remsi %5, %arg11 : i32
  %7 = arith.muli %arg12, %arg11 : i32
  %8 = arith.divsi %3, %7 : i32
  %9 = arith.remsi %8, %arg10 : i32
  %10 = arith.muli %9, %c32_i32 : i32
  %11 = arith.muli %6, %c384_i32 : i32
  %12 = arith.muli %4, %c32_i32 : i32
  %13 = arith.muli %6, %c96_i32 : i32
  %14 = arith.extsi %13 : i32 to i64
  %15 = arith.extsi %12 : i32 to i64
  %16 = arith.addi %14, %15 : i64
  %17 = arith.addi %16, %c7_i64 : i64
  %18 = arith.index_cast %12 : i32 to index
  %19 = arith.index_cast %13 : i32 to index
  %20 = arith.index_cast %10 : i32 to index
  %21 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%20, %18, %19]
  %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%21], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %22 = hivm.hir.pointer_cast(%c288_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%22 : memref<32xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  %23 = arith.index_cast %11 : i32 to index
  %24 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%20, %23, %18]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%24], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%22 : memref<32xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%21], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  %25:4 = scf.for %arg13 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg14 = %22, %arg15 = %17, %arg16 = %20, %arg17 = %20) -> (memref<32xf32, #hivm.address_space<ub>>, i64, index, index)  : i32 {
    %26 = arith.index_cast %arg13 : i32 to index
    %27 = arith.index_cast %c0_i32 : i32 to index
    %28 = arith.index_cast %c4_i32 : i32 to index
    %29 = arith.index_cast %c1_i32 : i32 to index
    %30 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%26, %27, %29]
    %31 = arith.index_cast %30 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %32 = arith.select %31, %c0_i64_2, %c1_i64 : i64
    %33 = hivm.hir.pointer_cast(%c0_i64, %c416_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %34 = arith.index_cast %arg15 : i64 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%34], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %35 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%arg16, %23, %18]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %36 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 + 96)>()[%arg17, %23, %18]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%36], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %32]
    hivm.hir.load ins(%reinterpret_cast_4 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%33 : memref<32xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %37 = memref.load %reinterpret_cast_3[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %38 = hivm.hir.pointer_cast(%c128_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %37, %38[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %39 = hivm.hir.pointer_cast(%c128_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vexp ins(%38 : memref<1xf32, #hivm.address_space<ub>>) outs(%39 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %40 = memref.load %39[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %41 = hivm.hir.pointer_cast(%c160_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%arg14, %40 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%41 : memref<32xf32, #hivm.address_space<ub>>)
    %42 = hivm.hir.pointer_cast(%c288_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%41, %33 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%42 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %32]
    %43 = arith.cmpi slt, %arg13, %c3_i32 : i32
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.if %43 {
      hivm.hir.store ins(%42 : memref<32xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_5 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    } else {
      hivm.hir.store ins(%42 : memref<32xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    %44 = arith.addi %arg15, %c8_i64 : i64
    %45 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%arg16]
    %46 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%arg17]
    scf.yield %42, %44, %45, %46 : memref<32xf32, #hivm.address_space<ub>>, i64, index, index
  } {tt.divisibility_arg4 = dense<16> : tensor<1xi32>}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
