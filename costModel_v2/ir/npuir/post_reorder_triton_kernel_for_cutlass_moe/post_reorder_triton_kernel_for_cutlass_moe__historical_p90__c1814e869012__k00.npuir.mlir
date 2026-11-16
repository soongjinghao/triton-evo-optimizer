// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @post_reorder_triton_kernel_for_cutlass_moe(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c768_i64 = arith.constant 768 : i64
  %c256_i64 = arith.constant 256 : i64
  %c512_i64 = arith.constant 512 : i64
  %c0_i64 = arith.constant 0 : i64
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg13, %arg12 : i32
  %5 = arith.extsi %arg8 : i32 to i64
  %6 = arith.extsi %arg10 : i32 to i64
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg14 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %7 = arith.divsi %arg14, %4 : i32
    %8 = arith.remsi %7, %arg11 : i32
    %9 = arith.divsi %8, %arg8 : i32
    %10 = arith.remsi %8, %arg8 : i32
    %11 = arith.extsi %9 : i32 to i64
    %12 = arith.extsi %10 : i32 to i64
    %13 = arith.muli %11, %5 : i64
    %14 = arith.muli %11, %6 : i64
    %15 = arith.addi %13, %12 : i64
    %16 = arith.index_cast %15 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    scf.for %arg15 = %c0_i32 to %arg10 step %c128_i32  : i32 {
      %17 = arith.index_cast %arg15 : i32 to index
      %18 = arith.index_cast %c0_i32 : i32 to index
      %19 = arith.index_cast %arg10 : i32 to index
      %20 = arith.index_cast %c128_i32 : i32 to index
      %21 = arith.index_cast %arg14 : i32 to index
      %22 = arith.index_cast %3 : i32 to index
      %23 = arith.index_cast %1 : i32 to index
      %24 = arith.index_cast %c48_i32 : i32 to index
      %25 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%17, %18, %19, %20, %21, %22, %24]
      %26 = arith.index_cast %25 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %27 = arith.select %26, %c0_i64_0, %c1_i64 : i64
      %28 = hivm.hir.pointer_cast(%c256_i64, %c768_i64) : memref<128xf16, #hivm.address_space<ub>>
      annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
      %29 = hivm.hir.pointer_cast(%c0_i64, %c512_i64) : memref<128xf16, #hivm.address_space<ub>>
      annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
      %30 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %31 = arith.cmpi ne, %30, %arg9 : i32
      scf.if %31 {
        %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %32 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %33 = arith.extsi %32 : i32 to i64
        %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%16], sizes: [1], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %34 = memref.load %reinterpret_cast_2[%c0] : memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %35 = arith.muli %33, %6 : i64
        %36 = arith.index_cast %35 : i64 to index
        %37 = arith.index_cast %arg15 : i32 to index
        %38 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%36, %37]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%38], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %39 = arith.index_cast %arg10 : i32 to index
        %40 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%37, %39]
        %41 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%40, %37]
        %42 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%41, %37]
        %43 = arith.cmpi slt, %42, %c128 : index
        %subview = memref.subview %reinterpret_cast_3[0] [%42] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %29[0] [%42] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %43 {
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vbrc ins(%cst : f16) outs(%29 : memref<128xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %27]
        hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %27]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%29, %34 : memref<128xf16, #hivm.address_space<ub>>, f16) outs(%28 : memref<128xf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %27]
        %44 = arith.index_cast %14 : i64 to index
        %45 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%44, %37]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%45], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_6 = memref.subview %28[0] [%42] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        %subview_7 = memref.subview %reinterpret_cast_5[0] [%42] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_6 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %27]
      } {pipeline.veconly}
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
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
