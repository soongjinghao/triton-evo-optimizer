// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_copy_page_indices_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c3072_i64 = arith.constant 3072 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c256_i32 = arith.constant 256 : i32
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg10 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<256xi32, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<256xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %13 = arith.divsi %arg10, %4 : i32
    %14 = arith.remsi %13, %arg7 : i32
    %15 = arith.muli %14, %arg5 : i32
    %16 = arith.index_cast %14 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %17 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %18 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%16]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%18], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %19 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %20 = arith.subi %19, %17 : i32
    %21 = arith.cmpi slt, %20, %c256_i32 : i32
    scf.if %21 {
      %22 = arith.index_cast %15 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%22], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %23 = arith.index_cast %20 : i32 to index
      %24 = affine.max affine_map<()[s0] -> (0, s0)>()[%23]
      %25 = affine.min affine_map<()[s0] -> (256, s0)>()[%24]
      %26 = arith.cmpi slt, %25, %c256 : index
      %subview = memref.subview %reinterpret_cast_2[0] [%25] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %12[0] [%25] [1] : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %26 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%c0_i32 : i32) outs(%12 : memref<256xi32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
      hivm.hir.load ins(%subview : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %27 = arith.index_cast %17 : i32 to index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %reinterpret_cast_4[0] [%25] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_3 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
    } else {
      %22 = arith.index_cast %15 : i32 to index
      %23 = arith.index_cast %20 : i32 to index
      %24 = arith.index_cast %17 : i32 to index
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
      scf.for %arg11 = %c0_i32 to %20 step %c256_i32  : i32 {
        %25 = arith.index_cast %arg11 : i32 to index
        %26 = arith.index_cast %c0_i32 : i32 to index
        %27 = arith.index_cast %20 : i32 to index
        %28 = arith.index_cast %c256_i32 : i32 to index
        %29 = arith.index_cast %arg10 : i32 to index
        %30 = arith.index_cast %3 : i32 to index
        %31 = arith.index_cast %1 : i32 to index
        %32 = arith.index_cast %c48_i32 : i32 to index
        %33 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%25, %26, %27, %28, %29, %30, %32]
        %34 = arith.index_cast %33 : index to i1
        %c2_i64 = arith.constant 2 : i64
        %c3_i64 = arith.constant 3 : i64
        %35 = arith.select %34, %c2_i64, %c3_i64 : i64
        %36 = hivm.hir.pointer_cast(%c1024_i64, %c3072_i64) : memref<256xi32, #hivm.address_space<ub>>
        annotation.mark %36 {hivm.multi_buffer = 2 : i32} : memref<256xi32, #hivm.address_space<ub>>
        %37 = arith.index_cast %arg11 : i32 to index
        %38 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%22, %37]
        %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %39 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%37, %23]
        %40 = affine.min affine_map<()[s0, s1] -> (s1 + 256, s0)>()[%39, %37]
        %41 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%40, %37]
        %42 = arith.cmpi slt, %41, %c256 : index
        %subview = memref.subview %reinterpret_cast_2[0] [%41] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_3 = memref.subview %36[0] [%41] [1] : memref<256xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %42 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%c0_i32 : i32) outs(%36 : memref<256xi32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %35]
        hivm.hir.load ins(%subview : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c0_i32 : i32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %43 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%24, %37]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [256], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %reinterpret_cast_4[0] [%41] [1] : memref<256xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_3 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %35]
      }
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
