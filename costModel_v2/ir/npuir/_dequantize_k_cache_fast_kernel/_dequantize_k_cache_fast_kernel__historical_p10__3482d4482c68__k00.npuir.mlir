// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_dequantize_k_cache_fast_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1792_i64 = arith.constant 1792 : i64
  %c768_i64 = arith.constant 768 : i64
  %c1280_i64 = arith.constant 1280 : i64
  %c256_i64 = arith.constant 256 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %c4_i32 = arith.constant 4 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg13, %arg12 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg14 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg14 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %11 = arith.select %10, %c2_i64, %c3_i64 : i64
    %12 = hivm.hir.pointer_cast(%c768_i64, %c1792_i64) : memref<128xbf16, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %13 = arith.divsi %arg14, %4 : i32
    %14 = arith.remsi %13, %arg11 : i32
    %15 = arith.muli %14, %arg8 : i32
    %16 = arith.muli %14, %arg9 : i32
    %17 = arith.muli %14, %arg7 : i32
    %18 = arith.index_cast %15 : i32 to index
    %19 = arith.index_cast %16 : i32 to index
    %20 = arith.index_cast %17 : i32 to index
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    scf.for %arg15 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %22 = arith.index_cast %arg15 : i32 to index
      %23 = arith.index_cast %c0_i32 : i32 to index
      %24 = arith.index_cast %c4_i32 : i32 to index
      %25 = arith.index_cast %c1_i32 : i32 to index
      %26 = arith.index_cast %arg14 : i32 to index
      %27 = arith.index_cast %3 : i32 to index
      %28 = arith.index_cast %1 : i32 to index
      %29 = arith.index_cast %c48_i32 : i32 to index
      %30 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%22, %23, %24, %25, %26, %27, %29]
      %31 = arith.index_cast %30 : index to i1
      %c1_i64 = arith.constant 1 : i64
      %c2_i64_1 = arith.constant 2 : i64
      %32 = arith.select %31, %c1_i64, %c2_i64_1 : i64
      %33 = arith.index_cast %30 : index to i1
      %c0_i64_2 = arith.constant 0 : i64
      %c1_i64_3 = arith.constant 1 : i64
      %34 = arith.select %33, %c0_i64_2, %c1_i64_3 : i64
      %35 = hivm.hir.pointer_cast(%c256_i64, %c1280_i64) : memref<128xbf16, #hivm.address_space<ub>>
      annotation.mark %35 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
      %36 = hivm.hir.pointer_cast(%c0_i64, %c1024_i64) : memref<128xbf16, #hivm.address_space<ub>>
      annotation.mark %36 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
      %37 = arith.muli %arg15, %c128_i32 : i32
      %38 = arith.index_cast %37 : i32 to index
      %39 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%18, %38]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%39], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %40 = arith.index_cast %arg15 : i32 to index
      %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%19, %40]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%41], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %42 = affine.max affine_map<()[s0] -> (512, s0)>()[%38]
      %43 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%42, %38]
      %44 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%43, %38]
      %45 = arith.cmpi slt, %44, %c128 : index
      %subview_6 = memref.subview %reinterpret_cast_4[0] [%44] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %36[0] [%44] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      scf.if %45 {
        hivm.hir.vbrc ins(%cst : bf16) outs(%36 : memref<128xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %32]
      hivm.hir.load ins(%subview_6 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %46 = hivm.hir.pointer_cast(%c256_i64, %c1280_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %34]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%36 : memref<128xbf16, #hivm.address_space<ub>>) outs(%46 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %32]
      %47 = memref.load %reinterpret_cast_5[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %48 = hivm.hir.pointer_cast(%c256_i64, %c1280_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%46, %47 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%48 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%48 : memref<128xf32, #hivm.address_space<ub>>) outs(%35 : memref<128xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%20, %38]
      %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%49], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %35[0] [%44] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      %subview_10 = memref.subview %reinterpret_cast_8[0] [%44] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_9 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_10 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %34]
    } {pipeline.veconly}
    %21 = affine.apply affine_map<()[s0] -> (s0 + 1024)>()[%20]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%21], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %11]
    hivm.hir.vbrc ins(%cst : bf16) outs(%12 : memref<128xbf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview = memref.subview %12[0] [0] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<0xbf16, strided<[1]>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %reinterpret_cast[0] [0] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<0xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview : memref<0xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_0 : memref<0xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %11]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
