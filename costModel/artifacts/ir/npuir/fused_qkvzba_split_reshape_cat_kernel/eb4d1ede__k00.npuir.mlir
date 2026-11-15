// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @fused_qkvzba_split_reshape_cat_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c3168_i64 = arith.constant 3168 : i64
  %c1568_i64 = arith.constant 1568 : i64
  %c3136_i64 = arith.constant 3136 : i64
  %c1536_i64 = arith.constant 1536 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c1856_i64 = arith.constant 1856 : i64
  %c256_i64 = arith.constant 256 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c4096_i32 = arith.constant 4096 : i32
  %c6144_i32 = arith.constant 6144 : i32
  %c32_i32 = arith.constant 32 : i32
  %c2048_i32 = arith.constant 2048 : i32
  %c4_i32 = arith.constant 4 : i32
  %c64_i32 = arith.constant 64 : i32
  %c384_i32 = arith.constant 384 : i32
  %c2_i32 = arith.constant 2 : i32
  %c8_i32 = arith.constant 8 : i32
  %c512_i64 = arith.constant 512 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c32_i64 = arith.constant 32 : i64
  %c16_i64 = arith.constant 16 : i64
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg11, %arg10 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID7>]
  scf.for %arg12 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg12, %4 : i32
    %6 = arith.remsi %5, %arg9 : i32
    %7 = arith.divsi %6, %c8_i32 : i32
    %8 = arith.remsi %6, %c8_i32 : i32
    %9 = arith.muli %7, %c2_i32 : i32
    %10 = arith.muli %7, %c6144_i32 : i32
    %11 = arith.extsi %10 : i32 to i64
    %12 = arith.muli %8, %c384_i32 : i32
    %13 = arith.extsi %12 : i32 to i64
    %14 = arith.addi %11, %13 : i64
    %15 = arith.muli %7, %c4096_i32 : i32
    %16 = arith.extsi %15 : i32 to i64
    %17 = arith.muli %8, %c64_i32 : i32
    %18 = arith.extsi %17 : i32 to i64
    %19 = arith.addi %16, %18 : i64
    %20 = arith.addi %16, %c512_i64 : i64
    %21 = arith.addi %20, %18 : i64
    %22 = arith.addi %16, %c1024_i64 : i64
    %23 = arith.muli %8, %c128_i32 : i32
    %24 = arith.extsi %23 : i32 to i64
    %25 = arith.addi %22, %24 : i64
    %26 = arith.muli %7, %c2048_i32 : i32
    %27 = arith.extsi %26 : i32 to i64
    %28 = arith.addi %27, %24 : i64
    %29 = arith.muli %7, %c64_i32 : i32
    %30 = arith.extsi %29 : i32 to i64
    %31 = arith.muli %8, %c4_i32 : i32
    %32 = arith.extsi %31 : i32 to i64
    %33 = arith.addi %30, %32 : i64
    %34 = arith.muli %7, %c32_i32 : i32
    %35 = arith.extsi %34 : i32 to i64
    %36 = arith.muli %8, %c2_i32 : i32
    %37 = arith.extsi %36 : i32 to i64
    %38 = arith.addi %35, %37 : i64
    %39:9 = scf.for %arg13 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg14 = %9, %arg15 = %14, %arg16 = %19, %arg17 = %21, %arg18 = %25, %arg19 = %28, %arg20 = %33, %arg21 = %38, %arg22 = %38) -> (i32, i64, i64, i64, i64, i64, i64, i64, i64)  : i32 {
      %40 = arith.index_cast %arg13 : i32 to index
      %41 = arith.index_cast %c0_i32 : i32 to index
      %42 = arith.index_cast %c2_i32 : i32 to index
      %43 = arith.index_cast %c1_i32 : i32 to index
      %44 = arith.index_cast %arg12 : i32 to index
      %45 = arith.index_cast %3 : i32 to index
      %46 = arith.index_cast %1 : i32 to index
      %47 = arith.index_cast %c48_i32 : i32 to index
      %48 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%40, %41, %42, %43, %44, %45, %47]
      %49 = arith.index_cast %48 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %50 = arith.select %49, %c0_i64_0, %c1_i64 : i64
      %51 = arith.index_cast %48 : index to i1
      %c2_i64 = arith.constant 2 : i64
      %c3_i64 = arith.constant 3 : i64
      %52 = arith.select %51, %c2_i64, %c3_i64 : i64
      %53 = arith.index_cast %48 : index to i1
      %c4_i64 = arith.constant 4 : i64
      %c5_i64 = arith.constant 5 : i64
      %54 = arith.select %53, %c4_i64, %c5_i64 : i64
      %55 = arith.index_cast %48 : index to i1
      %c6_i64 = arith.constant 6 : i64
      %c7_i64 = arith.constant 7 : i64
      %56 = arith.select %55, %c6_i64, %c7_i64 : i64
      %57 = hivm.hir.pointer_cast(%c1568_i64, %c3168_i64) : memref<2xf32, #hivm.address_space<ub>>
      annotation.mark %57 {hivm.multi_buffer = 2 : i32} : memref<2xf32, #hivm.address_space<ub>>
      %58 = hivm.hir.pointer_cast(%c1536_i64, %c3136_i64) : memref<2xf32, #hivm.address_space<ub>>
      annotation.mark %58 {hivm.multi_buffer = 2 : i32} : memref<2xf32, #hivm.address_space<ub>>
      %59 = hivm.hir.pointer_cast(%c1024_i64, %c2624_i64) : memref<128xf32, #hivm.address_space<ub>>
      annotation.mark %59 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
      %60 = hivm.hir.pointer_cast(%c512_i64, %c2112_i64) : memref<128xf32, #hivm.address_space<ub>>
      annotation.mark %60 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
      %61 = hivm.hir.pointer_cast(%c256_i64, %c1856_i64) : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %61 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
      %62 = hivm.hir.pointer_cast(%c0_i64, %c1600_i64) : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %62 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
      %63 = arith.cmpi slt, %arg14, %c4_i32 : i32
      scf.if %63 {
        %73 = arith.index_cast %arg15 : i64 to index
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%73], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %56]
        hivm.hir.load ins(%reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%62 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
        %74 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%73]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%74], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.load ins(%reinterpret_cast_1 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%61 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
        %75 = affine.apply affine_map<()[s0] -> (s0 + 128)>()[%73]
        %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%75], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.load ins(%reinterpret_cast_2 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%60 : memref<128xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %76 = affine.apply affine_map<()[s0] -> (s0 + 256)>()[%73]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%76], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %54]
        hivm.hir.load ins(%reinterpret_cast_3 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%59 : memref<128xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %77 = arith.index_cast %arg16 : i64 to index
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%77], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%62 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %78 = arith.index_cast %arg17 : i64 to index
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%78], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%61 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_5 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %79 = arith.index_cast %arg18 : i64 to index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%79], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%60 : memref<128xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %56]
        %80 = arith.index_cast %arg19 : i64 to index
        %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%80], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%59 : memref<128xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_7 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %54]
        %81 = arith.index_cast %arg20 : i64 to index
        %reinterpret_cast_8 = memref.reinterpret_cast %arg8 to offset: [%81], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %52]
        hivm.hir.load ins(%reinterpret_cast_8 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%58 : memref<2xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %82 = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%81]
        %reinterpret_cast_9 = memref.reinterpret_cast %arg8 to offset: [%82], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %50]
        hivm.hir.load ins(%reinterpret_cast_9 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%57 : memref<2xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %83 = arith.index_cast %arg21 : i64 to index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%83], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.store ins(%58 : memref<2xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_10 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %52]
        %84 = arith.index_cast %arg22 : i64 to index
        %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%84], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%57 : memref<2xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_11 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %50]
      } {pipeline.veconly}
      %64 = arith.addi %arg14, %c1_i32 : i32
      %65 = arith.addi %arg15, %c3072_i64 : i64
      %66 = arith.addi %arg16, %c2048_i64 : i64
      %67 = arith.addi %arg17, %c2048_i64 : i64
      %68 = arith.addi %arg18, %c2048_i64 : i64
      %69 = arith.addi %arg19, %c1024_i64 : i64
      %70 = arith.addi %arg20, %c32_i64 : i64
      %71 = arith.addi %arg21, %c16_i64 : i64
      %72 = arith.addi %arg22, %c16_i64 : i64
      scf.yield %64, %65, %66, %67, %68, %69, %70, %71, %72 : i32, i64, i64, i64, i64, i64, i64, i64, i64
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID7>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
