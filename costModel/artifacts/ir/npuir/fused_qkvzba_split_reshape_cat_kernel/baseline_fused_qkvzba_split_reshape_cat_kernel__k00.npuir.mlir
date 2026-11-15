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
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c64_i32 = arith.constant 64 : i32
  %c384_i32 = arith.constant 384 : i32
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
    %5 = arith.index_cast %arg12 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %11 = arith.select %10, %c6_i64, %c7_i64 : i64
    %12 = arith.index_cast %9 : index to i1
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %13 = arith.select %12, %c4_i64, %c5_i64 : i64
    %14 = arith.index_cast %9 : index to i1
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %15 = arith.select %14, %c2_i64, %c3_i64 : i64
    %16 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %17 = arith.select %16, %c0_i64_0, %c1_i64 : i64
    %18 = hivm.hir.pointer_cast(%c1568_i64, %c3168_i64) : memref<2xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<2xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c1536_i64, %c3136_i64) : memref<2xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<2xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c1024_i64, %c2624_i64) : memref<128xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c512_i64, %c2112_i64) : memref<128xf32, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c256_i64, %c1856_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c0_i64, %c1600_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %24 = arith.divsi %arg12, %arg11 : i32
    %25 = arith.remsi %24, %arg10 : i32
    %26 = arith.divsi %arg12, %4 : i32
    %27 = arith.remsi %26, %arg9 : i32
    %28 = arith.muli %27, %c3072_i32 : i32
    %29 = arith.muli %25, %c384_i32 : i32
    %30 = arith.index_cast %28 : i32 to index
    %31 = arith.index_cast %29 : i32 to index
    %32 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %31]
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%32], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %17]
    hivm.hir.load ins(%reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%23 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
    %33 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 64)>()[%30, %31]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%33], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_1 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%22 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
    %34 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 128)>()[%30, %31]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%34], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%21 : memref<128xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
    %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 256)>()[%30, %31]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%35], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
    hivm.hir.load ins(%reinterpret_cast_3 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%20 : memref<128xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %36 = arith.muli %27, %c2048_i32 : i32
    %37 = arith.muli %25, %c64_i32 : i32
    %38 = arith.muli %25, %c128_i32 : i32
    %39 = arith.index_cast %36 : i32 to index
    %40 = arith.index_cast %37 : i32 to index
    %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%39, %40]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%23 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %42 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 512)>()[%40, %39]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%22 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_5 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %43 = arith.index_cast %38 : i32 to index
    %44 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 1024)>()[%43, %39]
    %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%44], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%21 : memref<128xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %17]
    %45 = arith.muli %27, %c1024_i32 : i32
    %46 = arith.index_cast %45 : i32 to index
    %47 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%46, %43]
    %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%20 : memref<128xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_7 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
    %48 = arith.muli %27, %c32_i32 : i32
    %49 = arith.muli %25, %c4_i32 : i32
    %50 = arith.index_cast %48 : i32 to index
    %51 = arith.index_cast %49 : i32 to index
    %52 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%50, %51]
    %reinterpret_cast_8 = memref.reinterpret_cast %arg8 to offset: [%52], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
    hivm.hir.load ins(%reinterpret_cast_8 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%19 : memref<2xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
    %53 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 2)>()[%50, %51]
    %reinterpret_cast_9 = memref.reinterpret_cast %arg8 to offset: [%53], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
    hivm.hir.load ins(%reinterpret_cast_9 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%18 : memref<2xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %54 = arith.muli %27, %c16_i32 : i32
    %55 = arith.muli %25, %c2_i32 : i32
    %56 = arith.index_cast %54 : i32 to index
    %57 = arith.index_cast %55 : i32 to index
    %58 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%56, %57]
    %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%58], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.store ins(%19 : memref<2xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_10 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
    %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%58], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%18 : memref<2xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_11 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
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
