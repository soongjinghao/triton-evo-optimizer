// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_state_passing_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg9: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c160_i64 = arith.constant 160 : i64
  %c128_i64 = arith.constant 128 : i64
  %c416_i64 = arith.constant 416 : i64
  %c0_i64 = arith.constant 0 : i64
  %c544_i64 = arith.constant 544 : i64
  %c288_i64 = arith.constant 288 : i64
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  %c96_i32 = arith.constant 96 : i32
  %c384_i32 = arith.constant 384 : i32
  %c32_i32 = arith.constant 32 : i32
  %c7_i64 = arith.constant 7 : i64
  %c8_i64 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    %5 = hivm.hir.pointer_cast(%c288_i64, %c544_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %5 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %6 = arith.remsi %arg13, %arg12 : i32
    %7 = arith.divsi %arg13, %arg12 : i32
    %8 = arith.remsi %7, %arg11 : i32
    %9 = arith.divsi %arg13, %4 : i32
    %10 = arith.remsi %9, %arg10 : i32
    %11 = arith.muli %10, %c32_i32 : i32
    %12 = arith.muli %8, %c384_i32 : i32
    %13 = arith.muli %6, %c32_i32 : i32
    %14 = arith.muli %8, %c96_i32 : i32
    %15 = arith.extsi %14 : i32 to i64
    %16 = arith.extsi %13 : i32 to i64
    %17 = arith.addi %15, %16 : i64
    %18 = arith.addi %17, %c7_i64 : i64
    %19 = arith.index_cast %13 : i32 to index
    %20 = arith.index_cast %14 : i32 to index
    %21 = arith.index_cast %11 : i32 to index
    %22 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%21, %19, %20]
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%22], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%5 : memref<32xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %23 = arith.index_cast %12 : i32 to index
    %24 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%21, %23, %19]
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%24], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%5 : memref<32xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%22], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %25:4 = scf.for %arg14 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg15 = %5, %arg16 = %18, %arg17 = %21, %arg18 = %21) -> (memref<32xf32, #hivm.address_space<ub>>, i64, index, index)  : i32 {
      %26 = arith.index_cast %arg14 : i32 to index
      %27 = arith.index_cast %c0_i32 : i32 to index
      %28 = arith.index_cast %c4_i32 : i32 to index
      %29 = arith.index_cast %c1_i32 : i32 to index
      %30 = arith.index_cast %arg13 : i32 to index
      %31 = arith.index_cast %3 : i32 to index
      %32 = arith.index_cast %1 : i32 to index
      %33 = arith.index_cast %c48_i32 : i32 to index
      %34 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%26, %27, %28, %29, %30, %31, %33]
      %35 = arith.index_cast %34 : index to i1
      %c0_i64_2 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %36 = arith.select %35, %c0_i64_2, %c1_i64 : i64
      %37 = arith.index_cast %34 : index to i1
      %c1_i64_3 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %38 = arith.select %37, %c1_i64_3, %c2_i64 : i64
      %39 = hivm.hir.pointer_cast(%c288_i64, %c544_i64) : memref<32xf32, #hivm.address_space<ub>>
      annotation.mark %39 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
      %40 = hivm.hir.pointer_cast(%c0_i64, %c416_i64) : memref<32xf32, #hivm.address_space<ub>>
      annotation.mark %40 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
      %41 = arith.index_cast %arg16 : i64 to index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%41], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %42 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%arg17, %23, %19]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %43 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 + 96)>()[%arg18, %23, %19]
      %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %36]
      hivm.hir.load ins(%reinterpret_cast_5 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%40 : memref<32xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %44 = memref.load %reinterpret_cast_4[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %45 = hivm.hir.pointer_cast(%c128_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %44, %45[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %46 = hivm.hir.pointer_cast(%c128_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vexp ins(%45 : memref<1xf32, #hivm.address_space<ub>>) outs(%46 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %47 = memref.load %46[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %48 = hivm.hir.pointer_cast(%c160_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%arg15, %47 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%48 : memref<32xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %38]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%48, %40 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%39 : memref<32xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %36]
      %49 = arith.cmpi slt, %arg14, %c3_i32 : i32
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %49 {
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%39 : memref<32xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      } else {
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%39 : memref<32xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %38]
      %50 = arith.addi %arg16, %c8_i64 : i64
      %51 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%arg17]
      %52 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%arg18]
      scf.yield %39, %50, %51, %52 : memref<32xf32, #hivm.address_space<ub>>, i64, index, index
    } {pipeline.veconly, tt.divisibility_arg4 = dense<16> : tensor<1xi32>}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
