// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_trtllm_prefill_attn_kvfp8_dequant(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c163840_i64 = arith.constant 163840 : i64
  %c81920_i64 = arith.constant 81920 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c155648_i64 = arith.constant 155648 : i64
  %c57344_i64 = arith.constant 57344 : i64
  %c122880_i64 = arith.constant 122880 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c114688_i64 = arith.constant 114688 : i64
  %c1_i64 = arith.constant 1 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg11, %arg10 : i32
  %5 = arith.extsi %arg5 : i32 to i64
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  scf.for %arg12 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg12 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64_1 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_0, %c1_i64_1 : i64
    %13 = arith.index_cast %10 : index to i1
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %14 = arith.select %13, %c2_i64, %c3_i64 : i64
    %15 = hivm.hir.pointer_cast(%c81920_i64, %c163840_i64) : memref<8192xbf16, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<8192xbf16, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c57344_i64, %c155648_i64) : memref<8192xi8, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<8192xi8, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c24576_i64, %c122880_i64) : memref<8192xbf16, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<8192xbf16, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c0_i64, %c114688_i64) : memref<8192xi8, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<8192xi8, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %19 = arith.divsi %arg12, %arg11 : i32
    %20 = arith.remsi %19, %arg10 : i32
    %21 = arith.divsi %arg12, %4 : i32
    %22 = arith.remsi %21, %arg9 : i32
    %23 = arith.extsi %22 : i32 to i64
    %24 = arith.extsi %20 : i32 to i64
    %25 = arith.muli %24, %5 : i64
    %26 = arith.addi %25, %23 : i64
    %27 = arith.index_cast %26 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%27], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %28 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %29 = arith.extsi %28 : i32 to i64
    %30 = arith.cmpi sle, %29, %c0_i64 : i64
    scf.if %30 {
    } else {
      %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %31 = memref.load %reinterpret_cast_2[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %32 = arith.muli %29, %c16384_i64 : i64
      %33 = arith.addi %26, %c1_i64 : i64
      %34 = arith.muli %33, %c16384_i64 : i64
      %35 = arith.index_cast %32 : i64 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [8192], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<8192xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %14]
      hivm.hir.load ins(%reinterpret_cast_3 : memref<8192xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%18 : memref<8192xi8, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %36 = hivm.hir.pointer_cast(%c8192_i64) : memref<8192xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%18 : memref<8192xi8, #hivm.address_space<ub>>) outs(%36 : memref<8192xf16, #hivm.address_space<ub>>) cast = <cast_unsigned>
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %14]
      %37 = hivm.hir.pointer_cast(%c24576_i64, %c122880_i64) : memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %14]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%36 : memref<8192xf16, #hivm.address_space<ub>>) outs(%37 : memref<8192xf32, #hivm.address_space<ub>>)
      %38 = hivm.hir.pointer_cast(%c24576_i64, %c122880_i64) : memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%37, %31 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%38 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%38 : memref<8192xf32, #hivm.address_space<ub>>) outs(%17 : memref<8192xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %39 = arith.index_cast %34 : i64 to index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%39], sizes: [8192], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<8192xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%17 : memref<8192xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<8192xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %14]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %40 = memref.load %reinterpret_cast_5[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %41 = affine.apply affine_map<()[s0] -> (s0 + 8192)>()[%35]
      %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [8192], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<8192xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %12]
      hivm.hir.load ins(%reinterpret_cast_6 : memref<8192xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%16 : memref<8192xi8, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %42 = hivm.hir.pointer_cast(%c65536_i64) : memref<8192xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%16 : memref<8192xi8, #hivm.address_space<ub>>) outs(%42 : memref<8192xf16, #hivm.address_space<ub>>) cast = <cast_unsigned>
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %12]
      %43 = hivm.hir.pointer_cast(%c81920_i64, %c163840_i64) : memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %12]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%42 : memref<8192xf16, #hivm.address_space<ub>>) outs(%43 : memref<8192xf32, #hivm.address_space<ub>>)
      %44 = hivm.hir.pointer_cast(%c81920_i64, %c163840_i64) : memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%43, %40 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%44 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%44 : memref<8192xf32, #hivm.address_space<ub>>) outs(%15 : memref<8192xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %45 = affine.apply affine_map<()[s0] -> (s0 + 8192)>()[%39]
      %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%45], sizes: [8192], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<8192xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%15 : memref<8192xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_7 : memref<8192xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %12]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
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
