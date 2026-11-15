// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @softplus_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c16544_i64 = arith.constant 16544 : i64
  %c16512_i64 = arith.constant 16512 : i64
  %c20672_i64 = arith.constant 20672 : i64
  %c12416_i64 = arith.constant 12416 : i64
  %c8320_i64 = arith.constant 8320 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c16576_i64 = arith.constant 16576 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %c1024_i32 = arith.constant 1024 : i32
  %c1024 = arith.constant 1024 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = arith.index_cast %arg5 : i32 to index
  %6 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%arg7 : f32) outs(%6 : memref<1024xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %7 = arith.index_cast %arg11 : i32 to index
    %8 = arith.index_cast %3 : i32 to index
    %9 = arith.index_cast %1 : i32 to index
    %10 = arith.index_cast %c48_i32 : i32 to index
    %11 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%7, %8, %10]
    %12 = arith.index_cast %11 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %13 = arith.select %12, %c1_i64, %c2_i64 : i64
    %14 = arith.index_cast %11 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64_2 = arith.constant 1 : i64
    %15 = arith.select %14, %c0_i64_1, %c1_i64_2 : i64
    %16 = hivm.hir.pointer_cast(%c12416_i64, %c20672_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c4096_i64, %c16576_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %18 = arith.divsi %arg11, %4 : i32
    %19 = arith.remsi %18, %arg8 : i32
    %20 = arith.muli %19, %c1024_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%21], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %22 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%21, %5]
    %23 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%22, %21]
    %24 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%23, %21]
    %25 = arith.cmpi slt, %24, %c1024 : index
    %subview = memref.subview %reinterpret_cast[0] [%24] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %17[0] [%24] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %25 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%17 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %13]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%17, %arg6 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%17 : memref<1024xf32, #hivm.address_space<ub>>)
    %26 = hivm.hir.pointer_cast(%c8192_i64) : memref<1024xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%17, %6 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%26 : memref<1024xi1, #hivm.address_space<ub>>) compare_mode = <gt>
    %27 = hivm.hir.pointer_cast(%c8320_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%17 : memref<1024xf32, #hivm.address_space<ub>>) outs(%27 : memref<1024xf32, #hivm.address_space<ub>>)
    %28 = hivm.hir.pointer_cast(%c8320_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%27, %cst : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%28 : memref<1024xf32, #hivm.address_space<ub>>)
    %29 = hivm.hir.pointer_cast(%c8320_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%28 : memref<1024xf32, #hivm.address_space<ub>>) outs(%29 : memref<1024xf32, #hivm.address_space<ub>>)
    %30 = hivm.hir.pointer_cast(%c16512_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %15]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%26, %17, %29 : memref<1024xi1, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%16 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%30 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %13]
    %31 = hivm.hir.pointer_cast(%c16544_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%16, %arg6 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%16 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%31 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%21], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_5 = memref.subview %16[0] [%24] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_6 = memref.subview %reinterpret_cast_4[0] [%24] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_5 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %15]
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
