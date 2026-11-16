// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @softplus_kernel_default(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c20800_i64 = arith.constant 20800 : i64
  %c24928_i64 = arith.constant 24928 : i64
  %c16704_i64 = arith.constant 16704 : i64
  %c16576_i64 = arith.constant 16576 : i64
  %c16512_i64 = arith.constant 16512 : i64
  %c12416_i64 = arith.constant 12416 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c20832_i64 = arith.constant 20832 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c1024_i32 = arith.constant 1024 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+01 : f32
  %c1024 = arith.constant 1024 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<1024xf32, #hivm.address_space<ub>>)
  %6 = hivm.hir.pointer_cast(%c4096_i64) : memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%6 : memref<1024xf32, #hivm.address_space<ub>>)
  %7 = arith.index_cast %arg5 : i32 to index
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %8 = arith.index_cast %arg11 : i32 to index
    %9 = arith.index_cast %3 : i32 to index
    %10 = arith.index_cast %1 : i32 to index
    %11 = arith.index_cast %c48_i32 : i32 to index
    %12 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%8, %9, %11]
    %13 = arith.index_cast %12 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %14 = arith.select %13, %c1_i64, %c2_i64 : i64
    %15 = arith.index_cast %12 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64_3 = arith.constant 1 : i64
    %16 = arith.select %15, %c0_i64_2, %c1_i64_3 : i64
    %17 = hivm.hir.pointer_cast(%c16704_i64, %c24928_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c8192_i64, %c20832_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %19 = arith.divsi %arg11, %4 : i32
    %20 = arith.remsi %19, %arg8 : i32
    %21 = arith.muli %20, %c1024_i32 : i32
    %22 = arith.index_cast %21 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%22], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %23 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%22, %7]
    %24 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%23, %22]
    %25 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%24, %22]
    %26 = arith.cmpi slt, %25, %c1024 : index
    %subview = memref.subview %reinterpret_cast[0] [%25] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %18[0] [%25] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %26 {
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%18 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %14]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %27 = hivm.hir.pointer_cast(%c12288_i64) : memref<1024xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%18, %5 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%27 : memref<1024xi1, #hivm.address_space<ub>>) compare_mode = <gt>
    %28 = hivm.hir.pointer_cast(%c12416_i64) : memref<1024xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c16512_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%27, %18, %cst : memref<1024xi1, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%28 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%29 : memref<16xf32, #hivm.address_space<ub>>)
    %30 = hivm.hir.pointer_cast(%c12416_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%28 : memref<1024xf32, #hivm.address_space<ub>>) outs(%30 : memref<1024xf32, #hivm.address_space<ub>>)
    %31 = hivm.hir.pointer_cast(%c12416_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%30, %cst_0 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%31 : memref<1024xf32, #hivm.address_space<ub>>)
    %32 = hivm.hir.pointer_cast(%c12416_i64) : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%31 : memref<1024xf32, #hivm.address_space<ub>>) outs(%32 : memref<1024xf32, #hivm.address_space<ub>>)
    %33 = hivm.hir.pointer_cast(%c16576_i64) : memref<1024xi1, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%18, %6 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%33 : memref<1024xi1, #hivm.address_space<ub>>) compare_mode = <gt>
    %34 = hivm.hir.pointer_cast(%c20800_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %16]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%33, %18, %32 : memref<1024xi1, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%17 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%34 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %14]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%22], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %17[0] [%25] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_7 = memref.subview %reinterpret_cast_5[0] [%25] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %16]
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
