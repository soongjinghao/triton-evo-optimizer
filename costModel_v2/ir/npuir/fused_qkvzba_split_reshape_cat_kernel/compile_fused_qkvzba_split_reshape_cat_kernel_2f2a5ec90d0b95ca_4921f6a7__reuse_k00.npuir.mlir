// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @fused_qkvzba_split_reshape_cat_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c3168_i64 = arith.constant 3168 : i64
  %c1568_i64 = arith.constant 1568 : i64
  %c3136_i64 = arith.constant 3136 : i64
  %c1536_i64 = arith.constant 1536 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c2112_i64 = arith.constant 2112 : i64
  %c512_i64 = arith.constant 512 : i64
  %c1856_i64 = arith.constant 1856 : i64
  %c256_i64 = arith.constant 256 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c0_i64 = arith.constant 0 : i64
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %c2048_i32 = arith.constant 2048 : i32
  %c3072_i32 = arith.constant 3072 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c64_i32 = arith.constant 64 : i32
  %c384_i32 = arith.constant 384 : i32
  %c4_i32 = arith.constant 4 : i32
  %c2_i32 = arith.constant 2 : i32
  %c8_i32 = arith.constant 8 : i32
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
    scf.for %arg13 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %10 = arith.index_cast %arg13 : i32 to index
      %11 = arith.index_cast %c0_i32 : i32 to index
      %12 = arith.index_cast %c2_i32 : i32 to index
      %13 = arith.index_cast %c1_i32 : i32 to index
      %14 = arith.index_cast %arg12 : i32 to index
      %15 = arith.index_cast %3 : i32 to index
      %16 = arith.index_cast %1 : i32 to index
      %17 = arith.index_cast %c48_i32 : i32 to index
      %18 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%10, %11, %12, %13, %14, %15, %17]
      %19 = arith.index_cast %18 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %20 = arith.select %19, %c0_i64_0, %c1_i64 : i64
      %21 = arith.index_cast %18 : index to i1
      %c2_i64 = arith.constant 2 : i64
      %c3_i64 = arith.constant 3 : i64
      %22 = arith.select %21, %c2_i64, %c3_i64 : i64
      %23 = arith.index_cast %18 : index to i1
      %c4_i64 = arith.constant 4 : i64
      %c5_i64 = arith.constant 5 : i64
      %24 = arith.select %23, %c4_i64, %c5_i64 : i64
      %25 = arith.index_cast %18 : index to i1
      %c6_i64 = arith.constant 6 : i64
      %c7_i64 = arith.constant 7 : i64
      %26 = arith.select %25, %c6_i64, %c7_i64 : i64
      %27 = hivm.hir.pointer_cast(%c1568_i64, %c3168_i64) : memref<2xf32, #hivm.address_space<ub>>
      annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<2xf32, #hivm.address_space<ub>>
      %28 = hivm.hir.pointer_cast(%c1536_i64, %c3136_i64) : memref<2xf32, #hivm.address_space<ub>>
      annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<2xf32, #hivm.address_space<ub>>
      %29 = hivm.hir.pointer_cast(%c1024_i64, %c2624_i64) : memref<128xf32, #hivm.address_space<ub>>
      annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
      %30 = hivm.hir.pointer_cast(%c512_i64, %c2112_i64) : memref<128xf32, #hivm.address_space<ub>>
      annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
      %31 = hivm.hir.pointer_cast(%c256_i64, %c1856_i64) : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
      %32 = hivm.hir.pointer_cast(%c0_i64, %c1600_i64) : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
      %33 = arith.addi %9, %arg13 : i32
      %34 = arith.cmpi slt, %33, %c4_i32 : i32
      scf.if %34 {
        %35 = arith.muli %33, %c3072_i32 : i32
        %36 = arith.muli %8, %c384_i32 : i32
        %37 = arith.index_cast %35 : i32 to index
        %38 = arith.index_cast %36 : i32 to index
        %39 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%37, %38]
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%39], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %26]
        hivm.hir.load ins(%reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%32 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
        %40 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 64)>()[%37, %38]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%40], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.load ins(%reinterpret_cast_1 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%31 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
        %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 128)>()[%37, %38]
        %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%41], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.load ins(%reinterpret_cast_2 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%30 : memref<128xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %42 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 256)>()[%37, %38]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%42], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %24]
        hivm.hir.load ins(%reinterpret_cast_3 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%29 : memref<128xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %43 = arith.muli %33, %c2048_i32 : i32
        %44 = arith.muli %8, %c64_i32 : i32
        %45 = arith.muli %8, %c128_i32 : i32
        %46 = arith.index_cast %43 : i32 to index
        %47 = arith.index_cast %44 : i32 to index
        %48 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%46, %47]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%48], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%32 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 512)>()[%47, %46]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%31 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_5 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %50 = arith.index_cast %45 : i32 to index
        %51 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 1024)>()[%50, %46]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%51], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%30 : memref<128xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %26]
        %52 = arith.muli %33, %c1024_i32 : i32
        %53 = arith.index_cast %52 : i32 to index
        %54 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%53, %50]
        %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%29 : memref<128xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_7 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %24]
        %55 = arith.muli %33, %c32_i32 : i32
        %56 = arith.muli %8, %c4_i32 : i32
        %57 = arith.index_cast %55 : i32 to index
        %58 = arith.index_cast %56 : i32 to index
        %59 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%57, %58]
        %reinterpret_cast_8 = memref.reinterpret_cast %arg8 to offset: [%59], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %22]
        hivm.hir.load ins(%reinterpret_cast_8 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%28 : memref<2xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %60 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 2)>()[%57, %58]
        %reinterpret_cast_9 = memref.reinterpret_cast %arg8 to offset: [%60], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %20]
        hivm.hir.load ins(%reinterpret_cast_9 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%27 : memref<2xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %61 = arith.muli %33, %c16_i32 : i32
        %62 = arith.muli %8, %c2_i32 : i32
        %63 = arith.index_cast %61 : i32 to index
        %64 = arith.index_cast %62 : i32 to index
        %65 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%63, %64]
        %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.store ins(%28 : memref<2xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_10 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %22]
        %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%65], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%27 : memref<2xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_11 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %20]
      } {pipeline.veconly}
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
