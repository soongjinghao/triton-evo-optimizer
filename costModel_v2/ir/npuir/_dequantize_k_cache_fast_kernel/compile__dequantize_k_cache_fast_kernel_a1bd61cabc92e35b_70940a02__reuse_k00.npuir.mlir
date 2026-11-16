// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_dequantize_k_cache_fast_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1792_i64 = arith.constant 1792 : i64
  %c768_i64 = arith.constant 768 : i64
  %c1280_i64 = arith.constant 1280 : i64
  %c256_i64 = arith.constant 256 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c4_i32 = arith.constant 4 : i32
  %c2_i32 = arith.constant 2 : i32
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
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg14 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg14, %arg13 : i32
    %6 = arith.remsi %5, %arg12 : i32
    %7 = arith.divsi %arg14, %4 : i32
    %8 = arith.remsi %7, %arg11 : i32
    %9 = arith.muli %6, %c2_i32 : i32
    scf.for %arg15 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %10 = arith.index_cast %arg15 : i32 to index
      %11 = arith.index_cast %c0_i32 : i32 to index
      %12 = arith.index_cast %c2_i32 : i32 to index
      %13 = arith.index_cast %c1_i32 : i32 to index
      %14 = arith.index_cast %arg14 : i32 to index
      %15 = arith.index_cast %3 : i32 to index
      %16 = arith.index_cast %1 : i32 to index
      %17 = arith.index_cast %c48_i32 : i32 to index
      %18 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%10, %11, %12, %13, %14, %15, %17]
      %19 = arith.index_cast %18 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %20 = arith.select %19, %c0_i64_0, %c1_i64 : i64
      %21 = arith.index_cast %18 : index to i1
      %c1_i64_1 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %22 = arith.select %21, %c1_i64_1, %c2_i64 : i64
      %23 = hivm.hir.pointer_cast(%c768_i64, %c1792_i64) : memref<128xbf16, #hivm.address_space<ub>>
      annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
      %24 = hivm.hir.pointer_cast(%c256_i64, %c1280_i64) : memref<128xbf16, #hivm.address_space<ub>>
      annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
      %25 = hivm.hir.pointer_cast(%c0_i64, %c1024_i64) : memref<128xbf16, #hivm.address_space<ub>>
      annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<128xbf16, #hivm.address_space<ub>>
      %26 = arith.addi %9, %arg15 : i32
      %27 = arith.cmpi slt, %26, %c4_i32 : i32
      scf.if %27 {
        %28 = arith.muli %26, %c128_i32 : i32
        %29 = arith.muli %8, %arg8 : i32
        %30 = arith.index_cast %29 : i32 to index
        %31 = arith.index_cast %28 : i32 to index
        %32 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %31]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%32], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %33 = arith.muli %8, %arg9 : i32
        %34 = arith.index_cast %33 : i32 to index
        %35 = arith.index_cast %26 : i32 to index
        %36 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%34, %35]
        %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%36], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %37 = affine.max affine_map<()[s0] -> (512, s0)>()[%31]
        %38 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%37, %31]
        %39 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%38, %31]
        %40 = arith.cmpi slt, %39, %c128 : index
        %subview = memref.subview %reinterpret_cast[0] [%39] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_3 = memref.subview %25[0] [%39] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %40 {
          hivm.hir.vbrc ins(%cst : bf16) outs(%25 : memref<128xbf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %20]
        hivm.hir.load ins(%subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %41 = hivm.hir.pointer_cast(%c256_i64, %c1280_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %22]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vcast ins(%25 : memref<128xbf16, #hivm.address_space<ub>>) outs(%41 : memref<128xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %20]
        %42 = memref.load %reinterpret_cast_2[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %43 = hivm.hir.pointer_cast(%c256_i64, %c1280_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%41, %42 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%43 : memref<128xf32, #hivm.address_space<ub>>)
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vcast ins(%43 : memref<128xf32, #hivm.address_space<ub>>) outs(%24 : memref<128xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %44 = arith.muli %8, %arg7 : i32
        %45 = arith.index_cast %44 : i32 to index
        %46 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%45, %31]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%46], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %24[0] [%39] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
        %subview_6 = memref.subview %reinterpret_cast_4[0] [%39] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_5 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %22]
      } else {
        %28 = arith.cmpi eq, %26, %c4_i32 : i32
        scf.if %28 {
          %29 = arith.subi %26, %c4_i32 : i32
          %30 = arith.muli %29, %c128_i32 : i32
          %31 = arith.muli %8, %arg10 : i32
          %32 = arith.index_cast %31 : i32 to index
          %33 = arith.index_cast %30 : i32 to index
          %34 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%32, %33]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%34], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %35 = arith.muli %8, %arg7 : i32
          %36 = arith.index_cast %35 : i32 to index
          %37 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 512)>()[%33, %36]
          %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%37], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %38 = affine.max affine_map<()[s0] -> (64, s0)>()[%33]
          %39 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%38, %33]
          %40 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%39, %33]
          %41 = arith.cmpi slt, %40, %c128 : index
          %subview = memref.subview %reinterpret_cast[0] [%40] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_3 = memref.subview %23[0] [%40] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>
          scf.if %41 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.vbrc ins(%cst : bf16) outs(%23 : memref<128xbf16, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %20]
          hivm.hir.load ins(%subview : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : bf16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          %subview_4 = memref.subview %reinterpret_cast_2[0] [%40] [1] : memref<128xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          hivm.hir.store ins(%subview_3 : memref<?xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %20]
        }
      } {pipeline.veconly}
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
