// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_convert_req_index_to_global_index_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c5792_i64 = arith.constant 5792 : i64
  %c6368_i64 = arith.constant 6368 : i64
  %c5280_i64 = arith.constant 5280 : i64
  %c5216_i64 = arith.constant 5216 : i64
  %c4704_i64 = arith.constant 4704 : i64
  %c4192_i64 = arith.constant 4192 : i64
  %c3680_i64 = arith.constant 3680 : i64
  %c3168_i64 = arith.constant 3168 : i64
  %c3104_i64 = arith.constant 3104 : i64
  %c2592_i64 = arith.constant 2592 : i64
  %c2080_i64 = arith.constant 2080 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c1536_i64 = arith.constant 1536 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c5856_i64 = arith.constant 5856 : i64
  %c512_i64 = arith.constant 512 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c64_i32 = arith.constant 64 : i32
  %c11_i32 = arith.constant 11 : i32
  %c-1_i32 = arith.constant -1 : i32
  %cst = arith.constant 6.400000e+01 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<128xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%c64_i32 : i32) outs(%5 : memref<128xi32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c0_i64) : memref<128xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%5, %c0_i32 : memref<128xi32, #hivm.address_space<ub>>, i32) outs(%6 : memref<128xi1, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    %7 = arith.index_cast %arg13 : i32 to index
    %8 = arith.index_cast %3 : i32 to index
    %9 = arith.index_cast %1 : i32 to index
    %10 = arith.index_cast %c48_i32 : i32 to index
    %11 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%7, %8, %10]
    %12 = arith.index_cast %11 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %13 = arith.select %12, %c0_i64_0, %c1_i64 : i64
    %14 = hivm.hir.pointer_cast(%c5280_i64, %c6368_i64) : memref<128xi32, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<128xi32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c512_i64, %c5856_i64) : memref<128xi32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %16 = arith.divsi %arg13, %arg12 : i32
    %17 = arith.remsi %16, %arg11 : i32
    %18 = arith.divsi %arg13, %4 : i32
    %19 = arith.remsi %18, %arg10 : i32
    %20 = arith.muli %17, %c128_i32 : i32
    %21 = arith.index_cast %19 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%21], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %22 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %23 = arith.muli %19, %arg8 : i32
    %24 = arith.index_cast %23 : i32 to index
    %25 = arith.index_cast %20 : i32 to index
    %26 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%24, %25]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%26], sizes: [128], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %13]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %13]
    hivm.hir.load ins(%reinterpret_cast_1 : memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%15 : memref<128xi32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    %27 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmax ins(%15, %c0_i32 : memref<128xi32, #hivm.address_space<ub>>, i32) outs(%27 : memref<128xi32, #hivm.address_space<ub>>)
    %28 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%27, %15 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%28 : memref<128xi1, #hivm.address_space<ub>>)
    %29 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%28 : memref<128xi1, #hivm.address_space<ub>>) outs(%29 : memref<128xi1, #hivm.address_space<ub>>)
    %30 = hivm.hir.pointer_cast(%c1536_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%15 : memref<128xi32, #hivm.address_space<ub>>) outs(%30 : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %13]
    %31 = hivm.hir.pointer_cast(%c1536_i64) : memref<128xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c2048_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%30, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%31 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%32 : memref<8xf32, #hivm.address_space<ub>>)
    %33 = hivm.hir.pointer_cast(%c1536_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%31 : memref<128xf32, #hivm.address_space<ub>>) outs(%33 : memref<128xi32, #hivm.address_space<ub>>) round_mode = <trunc>
    %34 = hivm.hir.pointer_cast(%c2080_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg14 = %c0 to %c128 step %c1 {
      %55 = memref.load %15[%arg14] : memref<128xi32, #hivm.address_space<ub>>
      %56 = arith.remsi %55, %c64_i32 : i32
      memref.store %56, %34[%arg14] : memref<128xi32, #hivm.address_space<ub>>
    }
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %35 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xi32, #hivm.address_space<ub>>
    %36 = hivm.hir.pointer_cast(%c3104_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%6, %c-1_i32, %34 : memref<128xi1, #hivm.address_space<ub>>, i32, memref<128xi32, #hivm.address_space<ub>>) outs(%35 : memref<128xi32, #hivm.address_space<ub>>) temp_buffer(%36 : memref<16xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
    %37 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmax ins(%33, %c11_i32 : memref<128xi32, #hivm.address_space<ub>>, i32) outs(%37 : memref<128xi32, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%37, %33 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%38 : memref<128xi1, #hivm.address_space<ub>>)
    %39 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%38 : memref<128xi1, #hivm.address_space<ub>>) outs(%39 : memref<128xi1, #hivm.address_space<ub>>)
    %40 = hivm.hir.pointer_cast(%c3680_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%33, %c0_i32 : memref<128xi32, #hivm.address_space<ub>>, i32) outs(%40 : memref<128xi32, #hivm.address_space<ub>>)
    %41 = hivm.hir.pointer_cast(%c1536_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%40, %33 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%41 : memref<128xi1, #hivm.address_space<ub>>)
    %42 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vand ins(%39, %41 : memref<128xi1, #hivm.address_space<ub>>, memref<128xi1, #hivm.address_space<ub>>) outs(%42 : memref<128xi1, #hivm.address_space<ub>>)
    %43 = arith.muli %22, %arg7 : i32
    %44 = arith.extsi %43 : i32 to i64
    %45 = hivm.hir.pointer_cast(%c4192_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg14 = %c0 to %c128 step %c1 {
      %55 = memref.load %15[%arg14] {DiscreteMemAccess} : memref<128xi32, #hivm.address_space<ub>>
      %56 = arith.divsi %55, %c64_i32 : i32
      %57 = arith.extsi %56 : i32 to i64
      %58 = arith.addi %44, %57 : i64
      %59 = arith.index_cast %58 : i64 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%59], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %60 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      memref.store %60, %45[%arg14] : memref<128xi32, #hivm.address_space<ub>>
    } {ExtractedLoadOrStore}
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %13]
    %46 = hivm.hir.pointer_cast(%c4704_i64) : memref<128xi32, #hivm.address_space<ub>>
    %47 = hivm.hir.pointer_cast(%c5216_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%42, %45, %c0_i32 : memref<128xi1, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>, i32) outs(%46 : memref<128xi32, #hivm.address_space<ub>>) temp_buffer(%47 : memref<16xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%42 : memref<128xi1, #hivm.address_space<ub>>) outs(%42 : memref<128xi1, #hivm.address_space<ub>>)
    %48 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vor ins(%29, %42 : memref<128xi1, #hivm.address_space<ub>>, memref<128xi1, #hivm.address_space<ub>>) outs(%48 : memref<128xi1, #hivm.address_space<ub>>)
    %49 = hivm.hir.pointer_cast(%c4704_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%46, %c64_i32 : memref<128xi32, #hivm.address_space<ub>>, i32) outs(%49 : memref<128xi32, #hivm.address_space<ub>>)
    %50 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%49, %35 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%50 : memref<128xi32, #hivm.address_space<ub>>)
    %51 = hivm.hir.pointer_cast(%c5792_i64) : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %13]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%48, %c-1_i32, %50 : memref<128xi1, #hivm.address_space<ub>>, i32, memref<128xi32, #hivm.address_space<ub>>) outs(%14 : memref<128xi32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<16xi32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %52 = arith.muli %19, %arg9 : i32
    %53 = arith.index_cast %52 : i32 to index
    %54 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%53, %25]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%54], sizes: [128], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%14 : memref<128xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %13]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
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
