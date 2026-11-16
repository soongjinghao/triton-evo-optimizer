// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_dequantize_k_cache_fast_kernel_opt(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1792_i64 = arith.constant 1792 : i64
  %c768_i64 = arith.constant 768 : i64
  %c1280_i64 = arith.constant 1280 : i64
  %c256_i64 = arith.constant 256 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %c576_i32 = arith.constant 576 : i32
  %c128_i32 = arith.constant 128 : i32
  %c4_i32 = arith.constant 4 : i32
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg13, %arg12 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg14 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg14 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = arith.index_cast %9 : index to i1
    %c1_i64_1 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %13 = arith.select %12, %c1_i64_1, %c2_i64 : i64
    %14 = hivm.hir.pointer_cast(%c768_i64, %c1792_i64) : memref<128xbf16, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c256_i64, %c1280_i64) : memref<128xbf16, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c0_i64, %c1024_i64) : memref<128xbf16, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %17 = arith.divsi %arg14, %arg13 : i32
    %18 = arith.remsi %17, %arg12 : i32
    %19 = arith.divsi %arg14, %4 : i32
    %20 = arith.remsi %19, %arg11 : i32
    %21 = arith.cmpi slt, %18, %c4_i32 : i32
    scf.if %21 {
      %22 = arith.muli %18, %c128_i32 : i32
      %23 = arith.muli %20, %arg8 : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.index_cast %22 : i32 to index
      %26 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%24, %25]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %27 = arith.muli %20, %arg9 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = arith.index_cast %18 : i32 to index
      %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%28, %29]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%30], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %31 = affine.max affine_map<()[s0] -> (512, s0)>()[%25]
      %32 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%31, %25]
      %33 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%32, %25]
      %34 = arith.cmpi slt, %33, %c128 : index
      %subview = memref.subview %reinterpret_cast[0] [%33] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %16[0] [%33] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      scf.if %34 {
        hivm.hir.vbrc ins(%cst : bf16) outs(%16 : memref<128xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %11]
      hivm.hir.load ins(%subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %35 = hivm.hir.pointer_cast(%c256_i64, %c1280_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %13]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%16 : memref<128xbf16, #hivm.address_space<ub>>) outs(%35 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %11]
      %36 = memref.load %reinterpret_cast_2[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %37 = hivm.hir.pointer_cast(%c256_i64, %c1280_i64) : memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%35, %36 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%37 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%37 : memref<128xf32, #hivm.address_space<ub>>) outs(%15 : memref<128xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %38 = arith.muli %20, %c576_i32 : i32
      %39 = arith.index_cast %38 : i32 to index
      %40 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%39, %25]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%40], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %15[0] [%33] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      %subview_6 = memref.subview %reinterpret_cast_4[0] [%33] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_5 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %13]
    } else {
      %22 = arith.subi %18, %c4_i32 : i32
      %23 = arith.muli %22, %c128_i32 : i32
      %24 = arith.muli %20, %arg10 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = arith.index_cast %23 : i32 to index
      %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%25, %26]
      %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%27], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %28 = arith.muli %20, %c576_i32 : i32
      %29 = arith.index_cast %28 : i32 to index
      %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 512)>()[%26, %29]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %31 = affine.max affine_map<()[s0] -> (64, s0)>()[%26]
      %32 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%31, %26]
      %33 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%32, %26]
      %34 = arith.cmpi slt, %33, %c128 : index
      %subview = memref.subview %reinterpret_cast[0] [%33] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %14[0] [%33] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
      scf.if %34 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : bf16) outs(%14 : memref<128xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
      hivm.hir.load ins(%subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %subview_4 = memref.subview %reinterpret_cast_2[0] [%33] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_3 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
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
