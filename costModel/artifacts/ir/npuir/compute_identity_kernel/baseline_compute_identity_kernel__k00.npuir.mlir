// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @compute_identity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, true, false, true, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c2048_i64 = arith.constant 2048 : i64
  %c5120_i64 = arith.constant 5120 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c256_i32 = arith.constant 256 : i32
  %c0_i32 = arith.constant 0 : i32
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<256xf32, #hivm.address_space<ub>>)
  %6 = arith.divsi %arg8, %c256_i32 : i32
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    %7 = arith.index_cast %arg13 : i32 to index
    %8 = arith.index_cast %3 : i32 to index
    %9 = arith.index_cast %1 : i32 to index
    %10 = arith.index_cast %c48_i32 : i32 to index
    %11 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%7, %8, %10]
    %12 = arith.index_cast %11 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %13 = arith.select %12, %c0_i64_0, %c1_i64 : i64
    %14 = hivm.hir.pointer_cast(%c3072_i64, %c5120_i64) : memref<256xf32, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<256xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c1024_i64, %c4096_i64) : memref<256xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %16 = arith.divsi %arg13, %4 : i32
    %17 = arith.remsi %16, %arg10 : i32
    %18 = arith.divsi %17, %6 : i32
    %19 = arith.remsi %17, %6 : i32
    %20 = arith.muli %19, %c256_i32 : i32
    %21 = arith.cmpi sge, %18, %arg6 : i32
    %22 = arith.cmpi sge, %20, %arg8 : i32
    %23 = arith.ori %21, %22 : i1
    scf.if %23 {
    } else {
      %24 = arith.muli %18, %arg8 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = arith.index_cast %20 : i32 to index
      %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%25, %26]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%27], sizes: [256], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %28 = arith.index_cast %arg8 : i32 to index
      %29 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%26, %28]
      %30 = affine.min affine_map<()[s0, s1] -> (s1 + 256, s0)>()[%29, %26]
      %31 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%30, %26]
      %32 = arith.cmpi slt, %31, %c256 : index
      %subview = memref.subview %reinterpret_cast[0] [%31] [1] : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_1 = memref.subview %15[0] [%31] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %32 {
        hivm.hir.vbrc ins(%cst : f32) outs(%15 : memref<256xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %13]
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %33 = arith.muli %18, %arg9 : i32
      %34 = arith.index_cast %33 : i32 to index
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %13]
      hivm.hir.copy ins(%5 : memref<256xf32, #hivm.address_space<ub>>) outs(%14 : memref<256xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %35 = scf.for %arg14 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg15 = %14) -> (memref<256xf32, #hivm.address_space<ub>>)  : i32 {
        %36 = arith.index_cast %arg14 : i32 to index
        %37 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%34, %36]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %38 = memref.load %reinterpret_cast_5[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %39 = hivm.hir.pointer_cast(%c2048_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%15, %38 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%39 : memref<256xf32, #hivm.address_space<ub>>)
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%arg15, %39 : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%14 : memref<256xf32, #hivm.address_space<ub>>)
        scf.yield %14 : memref<256xf32, #hivm.address_space<ub>>
      }
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %13]
      %reinterpret_cast_2 = memref.reinterpret_cast %arg7 to offset: [%27], sizes: [256], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %35[0] [%31] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %reinterpret_cast_2[0] [%31] [1] : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %13]
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
