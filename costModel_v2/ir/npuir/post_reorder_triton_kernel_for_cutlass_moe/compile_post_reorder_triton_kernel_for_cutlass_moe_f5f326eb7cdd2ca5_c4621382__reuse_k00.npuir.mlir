// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @post_reorder_triton_kernel_for_cutlass_moe(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c768_i64 = arith.constant 768 : i64
  %c256_i64 = arith.constant 256 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c512_i64 = arith.constant 512 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f16
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg13, %arg12 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf16, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f16) outs(%5 : memref<128xf16, #hivm.address_space<ub>>)
  %6 = arith.extsi %arg8 : i32 to i64
  %7 = arith.extsi %arg10 : i32 to i64
  %8 = arith.index_cast %arg10 : i32 to index
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg14 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %9 = arith.divsi %arg14, %4 : i32
    %10 = arith.remsi %9, %arg11 : i32
    %11 = arith.extsi %10 : i32 to i64
    %12 = arith.muli %11, %6 : i64
    %13 = arith.muli %11, %7 : i64
    %14 = arith.index_cast %12 : i64 to index
    %15 = arith.index_cast %13 : i64 to index
    scf.for %arg15 = %c0_i32 to %arg10 step %c128_i32  : i32 {
      %16 = arith.index_cast %arg15 : i32 to index
      %17 = arith.index_cast %c0_i32 : i32 to index
      %18 = arith.index_cast %arg10 : i32 to index
      %19 = arith.index_cast %c128_i32 : i32 to index
      %20 = arith.index_cast %arg14 : i32 to index
      %21 = arith.index_cast %3 : i32 to index
      %22 = arith.index_cast %1 : i32 to index
      %23 = arith.index_cast %c48_i32 : i32 to index
      %24 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%16, %17, %18, %19, %20, %21, %23]
      %25 = arith.index_cast %24 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %26 = arith.select %25, %c0_i64_0, %c1_i64 : i64
      %27 = hivm.hir.pointer_cast(%c512_i64, %c1024_i64) : memref<128xf16, #hivm.address_space<ub>>
      annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %26]
      hivm.hir.copy ins(%5 : memref<128xf16, #hivm.address_space<ub>>) outs(%27 : memref<128xf16, #hivm.address_space<ub>>)
      %28 = scf.for %arg16 = %c0_i32 to %arg8 step %c1_i32 iter_args(%arg17 = %27) -> (memref<128xf16, #hivm.address_space<ub>>)  : i32 {
        %34 = arith.index_cast %arg16 : i32 to index
        %35 = arith.index_cast %c0_i32 : i32 to index
        %36 = arith.index_cast %arg8 : i32 to index
        %37 = arith.index_cast %c1_i32 : i32 to index
        %38 = arith.index_cast %arg15 : i32 to index
        %39 = arith.index_cast %c0_i32 : i32 to index
        %40 = arith.index_cast %arg10 : i32 to index
        %41 = arith.index_cast %c128_i32 : i32 to index
        %42 = arith.index_cast %arg14 : i32 to index
        %43 = arith.index_cast %3 : i32 to index
        %44 = arith.index_cast %1 : i32 to index
        %45 = arith.index_cast %c48_i32 : i32 to index
        %46 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s7) * ((-s1 + s2 + s3 - 1) floordiv s3) + ((s8 - s9) floordiv s10) * (((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7))) mod 2)>()[%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %45]
        %47 = arith.index_cast %46 : index to i1
        %c1_i64_2 = arith.constant 1 : i64
        %c2_i64 = arith.constant 2 : i64
        %48 = arith.select %47, %c1_i64_2, %c2_i64 : i64
        %49 = hivm.hir.pointer_cast(%c256_i64, %c768_i64) : memref<128xf16, #hivm.address_space<ub>>
        annotation.mark %49 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
        %50 = arith.index_cast %arg16 : i32 to index
        %51 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%14, %50]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%51], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %52 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %53 = arith.cmpi ne, %52, %arg9 : i32
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %48]
        %54 = scf.if %53 -> (memref<128xf16, #hivm.address_space<ub>>) {
          %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%51], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %55 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %56 = arith.extsi %55 : i32 to i64
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%51], sizes: [1], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %57 = memref.load %reinterpret_cast_5[%c0] : memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %58 = arith.muli %56, %7 : i64
          %59 = arith.index_cast %58 : i64 to index
          %60 = arith.index_cast %arg15 : i32 to index
          %61 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%59, %60]
          %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%61], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %62 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%60, %8]
          %63 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%62, %60]
          %64 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%63, %60]
          %65 = arith.cmpi slt, %64, %c128 : index
          %subview_7 = memref.subview %reinterpret_cast_6[0] [%64] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_8 = memref.subview %49[0] [%64] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
          scf.if %65 {
            hivm.hir.pipe_barrier[<PIPE_V>]
            hivm.hir.vbrc ins(%cst : f16) outs(%49 : memref<128xf16, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.load ins(%subview_7 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vmul ins(%49, %57 : memref<128xf16, #hivm.address_space<ub>>, f16) outs(%49 : memref<128xf16, #hivm.address_space<ub>>)
          scf.yield %49 : memref<128xf16, #hivm.address_space<ub>>
        } else {
          scf.yield %5 : memref<128xf16, #hivm.address_space<ub>>
        }
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%arg17, %54 : memref<128xf16, #hivm.address_space<ub>>, memref<128xf16, #hivm.address_space<ub>>) outs(%27 : memref<128xf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %48]
        scf.yield %27 : memref<128xf16, #hivm.address_space<ub>>
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %29 = arith.index_cast %arg15 : i32 to index
      %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%15, %29]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%30], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %31 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%29, %8]
      %32 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%31, %29]
      %33 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%32, %29]
      %subview = memref.subview %28[0] [%33] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %reinterpret_cast[0] [%33] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_1 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %26]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
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
