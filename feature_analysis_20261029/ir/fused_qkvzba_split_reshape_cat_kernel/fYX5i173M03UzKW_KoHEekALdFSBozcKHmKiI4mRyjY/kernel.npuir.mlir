// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @fused_qkvzba_split_reshape_cat_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1568_i64 = arith.constant 1568 : i64
  %c1536_i64 = arith.constant 1536 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c512_i64 = arith.constant 512 : i64
  %c256_i64 = arith.constant 256 : i64
  %c0_i64 = arith.constant 0 : i64
  %c384_i32 = arith.constant 384 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c2_i32 = arith.constant 2 : i32
  %c16_i32 = arith.constant 16 : i32
  %c4_i32 = arith.constant 4 : i32
  %c3072_i32 = arith.constant 3072 : i32
  %c2048_i32 = arith.constant 2048 : i32
  %c128_i32 = arith.constant 128 : i32
  %c32_i32 = arith.constant 32 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.divsi %3, %arg11 : i32
  %5 = arith.remsi %4, %arg10 : i32
  %6 = arith.muli %arg11, %arg10 : i32
  %7 = arith.divsi %3, %6 : i32
  %8 = arith.remsi %7, %arg9 : i32
  %9 = arith.muli %8, %c3072_i32 : i32
  %10 = arith.index_cast %9 : i32 to index
  %11 = arith.muli %5, %c384_i32 : i32
  %12 = arith.index_cast %11 : i32 to index
  %13 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%10, %12]
  %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%13], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %14 = hivm.hir.pointer_cast(%c0_i64) : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%14 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
  %15 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 64)>()[%10, %12]
  %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [%15], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %16 = hivm.hir.pointer_cast(%c256_i64) : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast_0 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%16 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
  %17 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 128)>()[%10, %12]
  %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%17], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %18 = hivm.hir.pointer_cast(%c512_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast_1 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%18 : memref<128xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
  %19 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 256)>()[%10, %12]
  %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%19], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %20 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast_2 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%20 : memref<128xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  %21 = arith.muli %8, %c2048_i32 : i32
  %22 = arith.index_cast %21 : i32 to index
  %23 = arith.muli %5, %c64_i32 : i32
  %24 = arith.index_cast %23 : i32 to index
  %25 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%22, %24]
  %26 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 512)>()[%24, %22]
  %27 = arith.muli %5, %c128_i32 : i32
  %28 = arith.index_cast %27 : i32 to index
  %29 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 1024)>()[%28, %22]
  %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
  hivm.hir.store ins(%14 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_3 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
  hivm.hir.pipe_barrier[<PIPE_MTE3>]
  hivm.hir.store ins(%16 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_MTE3>]
  hivm.hir.store ins(%18 : memref<128xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_5 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  %30 = arith.muli %8, %c1024_i32 : i32
  %31 = arith.index_cast %30 : i32 to index
  %32 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%31, %28]
  %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%32], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%20 : memref<128xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  %33 = arith.muli %8, %c32_i32 : i32
  %34 = arith.index_cast %33 : i32 to index
  %35 = arith.muli %5, %c4_i32 : i32
  %36 = arith.index_cast %35 : i32 to index
  %37 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%34, %36]
  %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%37], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %38 = hivm.hir.pointer_cast(%c1536_i64) : memref<2xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast_7 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%38 : memref<2xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
  %39 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 2)>()[%34, %36]
  %reinterpret_cast_8 = memref.reinterpret_cast %arg8 to offset: [%39], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %40 = hivm.hir.pointer_cast(%c1568_i64) : memref<2xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast_8 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%40 : memref<2xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  %41 = arith.muli %8, %c16_i32 : i32
  %42 = arith.index_cast %41 : i32 to index
  %43 = arith.muli %5, %c2_i32 : i32
  %44 = arith.index_cast %43 : i32 to index
  %45 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%42, %44]
  %reinterpret_cast_9 = memref.reinterpret_cast %arg5 to offset: [%45], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
  hivm.hir.store ins(%38 : memref<2xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_9 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%45], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%40 : memref<2xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_10 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
