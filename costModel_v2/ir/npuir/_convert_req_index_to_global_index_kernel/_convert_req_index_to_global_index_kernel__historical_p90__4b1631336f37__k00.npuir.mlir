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
  %c16_i32 = arith.constant 16 : i32
  %cst = arith.constant 6.400000e+01 : f32
  %c1_i32 = arith.constant 1 : i32
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
    hivm.hir.set_mask_norm
    %7 = arith.divsi %arg13, %4 : i32
    %8 = arith.remsi %7, %arg10 : i32
    %9 = arith.index_cast %8 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %10 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %11 = arith.muli %8, %arg8 : i32
    %12 = arith.muli %10, %arg7 : i32
    %13 = arith.extsi %12 : i32 to i64
    %14 = arith.muli %8, %arg9 : i32
    %15 = arith.index_cast %11 : i32 to index
    %16 = arith.index_cast %14 : i32 to index
    scf.for %arg14 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      %17 = arith.index_cast %arg14 : i32 to index
      %18 = arith.index_cast %c0_i32 : i32 to index
      %19 = arith.index_cast %c16_i32 : i32 to index
      %20 = arith.index_cast %c1_i32 : i32 to index
      %21 = arith.index_cast %arg13 : i32 to index
      %22 = arith.index_cast %3 : i32 to index
      %23 = arith.index_cast %1 : i32 to index
      %24 = arith.index_cast %c48_i32 : i32 to index
      %25 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%17, %18, %19, %20, %21, %22, %24]
      %26 = arith.index_cast %25 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %27 = arith.select %26, %c0_i64_0, %c1_i64 : i64
      %28 = hivm.hir.pointer_cast(%c5280_i64, %c6368_i64) : memref<128xi32, #hivm.address_space<ub>>
      annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<128xi32, #hivm.address_space<ub>>
      %29 = hivm.hir.pointer_cast(%c512_i64, %c5856_i64) : memref<128xi32, #hivm.address_space<ub>>
      annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<128xi32, #hivm.address_space<ub>>
      %30 = arith.muli %arg14, %c128_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %32 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%15, %31]
      %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%32], sizes: [128], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %27]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %27]
      hivm.hir.load ins(%reinterpret_cast_1 : memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%29 : memref<128xi32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %33 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmax ins(%29, %c0_i32 : memref<128xi32, #hivm.address_space<ub>>, i32) outs(%33 : memref<128xi32, #hivm.address_space<ub>>)
      %34 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%33, %29 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%34 : memref<128xi1, #hivm.address_space<ub>>)
      %35 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%34 : memref<128xi1, #hivm.address_space<ub>>) outs(%35 : memref<128xi1, #hivm.address_space<ub>>)
      %36 = hivm.hir.pointer_cast(%c1536_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%29 : memref<128xi32, #hivm.address_space<ub>>) outs(%36 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %27]
      %37 = hivm.hir.pointer_cast(%c1536_i64) : memref<128xf32, #hivm.address_space<ub>>
      %38 = hivm.hir.pointer_cast(%c2048_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vdiv ins(%36, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%37 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%38 : memref<8xf32, #hivm.address_space<ub>>)
      %39 = hivm.hir.pointer_cast(%c1536_i64) : memref<128xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%37 : memref<128xf32, #hivm.address_space<ub>>) outs(%39 : memref<128xi32, #hivm.address_space<ub>>) round_mode = <trunc>
      %40 = hivm.hir.pointer_cast(%c2080_i64) : memref<128xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg15 = %c0 to %c128 step %c1 {
        %57 = memref.load %29[%arg15] : memref<128xi32, #hivm.address_space<ub>>
        %58 = arith.remsi %57, %c64_i32 : i32
        memref.store %58, %40[%arg15] : memref<128xi32, #hivm.address_space<ub>>
      }
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %41 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xi32, #hivm.address_space<ub>>
      %42 = hivm.hir.pointer_cast(%c3104_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vsel ins(%6, %c-1_i32, %40 : memref<128xi1, #hivm.address_space<ub>>, i32, memref<128xi32, #hivm.address_space<ub>>) outs(%41 : memref<128xi32, #hivm.address_space<ub>>) temp_buffer(%42 : memref<16xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
      %43 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmax ins(%39, %c11_i32 : memref<128xi32, #hivm.address_space<ub>>, i32) outs(%43 : memref<128xi32, #hivm.address_space<ub>>)
      %44 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%43, %39 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%44 : memref<128xi1, #hivm.address_space<ub>>)
      %45 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%44 : memref<128xi1, #hivm.address_space<ub>>) outs(%45 : memref<128xi1, #hivm.address_space<ub>>)
      %46 = hivm.hir.pointer_cast(%c3680_i64) : memref<128xi32, #hivm.address_space<ub>>
      hivm.hir.vmax ins(%39, %c0_i32 : memref<128xi32, #hivm.address_space<ub>>, i32) outs(%46 : memref<128xi32, #hivm.address_space<ub>>)
      %47 = hivm.hir.pointer_cast(%c1536_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%46, %39 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%47 : memref<128xi1, #hivm.address_space<ub>>)
      %48 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vand ins(%45, %47 : memref<128xi1, #hivm.address_space<ub>>, memref<128xi1, #hivm.address_space<ub>>) outs(%48 : memref<128xi1, #hivm.address_space<ub>>)
      %49 = hivm.hir.pointer_cast(%c4192_i64) : memref<128xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg15 = %c0 to %c128 step %c1 {
        %57 = memref.load %29[%arg15] {DiscreteMemAccess} : memref<128xi32, #hivm.address_space<ub>>
        %58 = arith.divsi %57, %c64_i32 : i32
        %59 = arith.extsi %58 : i32 to i64
        %60 = arith.addi %13, %59 : i64
        %61 = arith.index_cast %60 : i64 to index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%61], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %62 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.store %62, %49[%arg15] : memref<128xi32, #hivm.address_space<ub>>
      } {ExtractedLoadOrStore}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %27]
      %50 = hivm.hir.pointer_cast(%c4704_i64) : memref<128xi32, #hivm.address_space<ub>>
      %51 = hivm.hir.pointer_cast(%c5216_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%48, %49, %c0_i32 : memref<128xi1, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>, i32) outs(%50 : memref<128xi32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<16xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%48 : memref<128xi1, #hivm.address_space<ub>>) outs(%48 : memref<128xi1, #hivm.address_space<ub>>)
      %52 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vor ins(%35, %48 : memref<128xi1, #hivm.address_space<ub>>, memref<128xi1, #hivm.address_space<ub>>) outs(%52 : memref<128xi1, #hivm.address_space<ub>>)
      %53 = hivm.hir.pointer_cast(%c4704_i64) : memref<128xi32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%50, %c64_i32 : memref<128xi32, #hivm.address_space<ub>>, i32) outs(%53 : memref<128xi32, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c2592_i64) : memref<128xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%53, %41 : memref<128xi32, #hivm.address_space<ub>>, memref<128xi32, #hivm.address_space<ub>>) outs(%54 : memref<128xi32, #hivm.address_space<ub>>)
      %55 = hivm.hir.pointer_cast(%c5792_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %27]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%52, %c-1_i32, %54 : memref<128xi1, #hivm.address_space<ub>>, i32, memref<128xi32, #hivm.address_space<ub>>) outs(%28 : memref<128xi32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<16xi32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %56 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%16, %31]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%56], sizes: [128], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%28 : memref<128xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_2 : memref<128xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %27]
    } {pipeline.veconly}
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
