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
  %c8_i32 = arith.constant 8 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg11, %arg10 : i32
  scf.for %arg12 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg12, %4 : i32
    %6 = arith.remsi %5, %arg9 : i32
    %7 = arith.divsi %6, %c4_i32 : i32
    %8 = arith.remsi %6, %c4_i32 : i32
    %9 = arith.muli %7, %c2_i32 : i32
    scf.for %arg13 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %10 = arith.addi %9, %arg13 : i32
      %11 = arith.cmpi slt, %10, %c4_i32 : i32
      scf.if %11 {
        %12 = arith.muli %8, %c2_i32 : i32
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID5>]
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID6>]
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID7>]
        scf.for %arg14 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
          %13 = arith.index_cast %arg14 : i32 to index
          %14 = arith.index_cast %c0_i32 : i32 to index
          %15 = arith.index_cast %c2_i32 : i32 to index
          %16 = arith.index_cast %c1_i32 : i32 to index
          %17 = arith.index_cast %arg13 : i32 to index
          %18 = arith.index_cast %c0_i32 : i32 to index
          %19 = arith.index_cast %c2_i32 : i32 to index
          %20 = arith.index_cast %c1_i32 : i32 to index
          %21 = arith.index_cast %arg12 : i32 to index
          %22 = arith.index_cast %3 : i32 to index
          %23 = arith.index_cast %1 : i32 to index
          %24 = arith.index_cast %c48_i32 : i32 to index
          %25 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s7) * ((-s1 + s2 + s3 - 1) floordiv s3) + ((s8 - s9) floordiv s10) * (((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7))) mod 2)>()[%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %24]
          %26 = arith.index_cast %25 : index to i1
          %c0_i64_0 = arith.constant 0 : i64
          %c1_i64 = arith.constant 1 : i64
          %27 = arith.select %26, %c0_i64_0, %c1_i64 : i64
          %28 = arith.index_cast %25 : index to i1
          %c2_i64 = arith.constant 2 : i64
          %c3_i64 = arith.constant 3 : i64
          %29 = arith.select %28, %c2_i64, %c3_i64 : i64
          %30 = arith.index_cast %25 : index to i1
          %c4_i64 = arith.constant 4 : i64
          %c5_i64 = arith.constant 5 : i64
          %31 = arith.select %30, %c4_i64, %c5_i64 : i64
          %32 = arith.index_cast %25 : index to i1
          %c6_i64 = arith.constant 6 : i64
          %c7_i64 = arith.constant 7 : i64
          %33 = arith.select %32, %c6_i64, %c7_i64 : i64
          %34 = hivm.hir.pointer_cast(%c1568_i64, %c3168_i64) : memref<2xf32, #hivm.address_space<ub>>
          annotation.mark %34 {hivm.multi_buffer = 2 : i32} : memref<2xf32, #hivm.address_space<ub>>
          %35 = hivm.hir.pointer_cast(%c1536_i64, %c3136_i64) : memref<2xf32, #hivm.address_space<ub>>
          annotation.mark %35 {hivm.multi_buffer = 2 : i32} : memref<2xf32, #hivm.address_space<ub>>
          %36 = hivm.hir.pointer_cast(%c1024_i64, %c2624_i64) : memref<128xf32, #hivm.address_space<ub>>
          annotation.mark %36 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
          %37 = hivm.hir.pointer_cast(%c512_i64, %c2112_i64) : memref<128xf32, #hivm.address_space<ub>>
          annotation.mark %37 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
          %38 = hivm.hir.pointer_cast(%c256_i64, %c1856_i64) : memref<64xf32, #hivm.address_space<ub>>
          annotation.mark %38 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
          %39 = hivm.hir.pointer_cast(%c0_i64, %c1600_i64) : memref<64xf32, #hivm.address_space<ub>>
          annotation.mark %39 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
          %40 = arith.addi %12, %arg14 : i32
          %41 = arith.cmpi slt, %40, %c8_i32 : i32
          scf.if %41 {
            %42 = arith.muli %10, %c3072_i32 : i32
            %43 = arith.muli %40, %c384_i32 : i32
            %44 = arith.index_cast %42 : i32 to index
            %45 = arith.index_cast %43 : i32 to index
            %46 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%44, %45]
            %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%46], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %33]
            hivm.hir.load ins(%reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%39 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
            %47 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 64)>()[%44, %45]
            %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%47], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.load ins(%reinterpret_cast_1 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%38 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
            %48 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 128)>()[%44, %45]
            %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%48], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.load ins(%reinterpret_cast_2 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%37 : memref<128xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
            %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 256)>()[%44, %45]
            %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%49], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %31]
            hivm.hir.load ins(%reinterpret_cast_3 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%36 : memref<128xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
            %50 = arith.muli %10, %c2048_i32 : i32
            %51 = arith.muli %40, %c64_i32 : i32
            %52 = arith.muli %40, %c128_i32 : i32
            %53 = arith.index_cast %50 : i32 to index
            %54 = arith.index_cast %51 : i32 to index
            %55 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%53, %54]
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%55], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
            hivm.hir.pipe_barrier[<PIPE_MTE3>]
            hivm.hir.store ins(%39 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
            %56 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 512)>()[%54, %53]
            %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%56], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
            hivm.hir.pipe_barrier[<PIPE_MTE3>]
            hivm.hir.store ins(%38 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_5 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
            %57 = arith.index_cast %52 : i32 to index
            %58 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 1024)>()[%57, %53]
            %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%58], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
            hivm.hir.pipe_barrier[<PIPE_MTE3>]
            hivm.hir.store ins(%37 : memref<128xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %33]
            %59 = arith.muli %10, %c1024_i32 : i32
            %60 = arith.index_cast %59 : i32 to index
            %61 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%60, %57]
            %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%61], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.store ins(%36 : memref<128xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_7 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %31]
            %62 = arith.muli %10, %c32_i32 : i32
            %63 = arith.muli %40, %c4_i32 : i32
            %64 = arith.index_cast %62 : i32 to index
            %65 = arith.index_cast %63 : i32 to index
            %66 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%64, %65]
            %reinterpret_cast_8 = memref.reinterpret_cast %arg8 to offset: [%66], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %29]
            hivm.hir.load ins(%reinterpret_cast_8 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%35 : memref<2xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
            %67 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 2)>()[%64, %65]
            %reinterpret_cast_9 = memref.reinterpret_cast %arg8 to offset: [%67], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %27]
            hivm.hir.load ins(%reinterpret_cast_9 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%34 : memref<2xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
            %68 = arith.muli %10, %c16_i32 : i32
            %69 = arith.muli %40, %c2_i32 : i32
            %70 = arith.index_cast %68 : i32 to index
            %71 = arith.index_cast %69 : i32 to index
            %72 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%70, %71]
            %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%72], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
            hivm.hir.store ins(%35 : memref<2xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_10 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %29]
            %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%72], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.store ins(%34 : memref<2xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_11 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %27]
          } {pipeline.veconly}
        }
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID4>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID5>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID6>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID7>]
      } {pipeline.veconly}
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
