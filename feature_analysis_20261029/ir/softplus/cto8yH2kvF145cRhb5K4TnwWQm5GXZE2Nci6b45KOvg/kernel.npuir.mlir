// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @softplus_kernel_default(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: f32, %arg7: f32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c16384_i64 = arith.constant 16384 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c1024 = arith.constant 1024 : index
  %cst_0 = arith.constant 1.000000e+00 : f32
  %c1024_i32 = arith.constant 1024 : i32
  %cst_1 = arith.constant 2.000000e+01 : f32
  %c0 = arith.constant 0 : index
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg8 : i32
  %7 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%7 : memref<1024xf32, #hivm.address_space<ub>>)
  %8 = arith.muli %6, %c1024_i32 : i32
  %9 = arith.index_cast %8 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %10 = hivm.hir.pointer_cast(%c4096_i64) : memref<1024xf32, #hivm.address_space<ub>>
  %11 = arith.index_cast %arg5 : i32 to index
  %12 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%9, %11]
  %13 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%12, %9]
  %14 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%13, %9]
  %15 = arith.cmpi slt, %14, %c1024 : index
  %subview = memref.subview %reinterpret_cast[0] [%14] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %subview_2 = memref.subview %10[0] [%14] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  scf.if %15 {
    hivm.hir.vbrc ins(%cst : f32) outs(%10 : memref<1024xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  %16 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xi1, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%10, %7 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%16 : memref<1024xi1, #hivm.address_space<ub>>) compare_mode = <gt>
  %17 = hivm.hir.pointer_cast(%c8192_i64) : memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vexp ins(%10 : memref<1024xf32, #hivm.address_space<ub>>) outs(%17 : memref<1024xf32, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c8192_i64) : memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%17, %cst_0 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%18 : memref<1024xf32, #hivm.address_space<ub>>)
  %19 = hivm.hir.pointer_cast(%c8192_i64) : memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vln ins(%18 : memref<1024xf32, #hivm.address_space<ub>>) outs(%19 : memref<1024xf32, #hivm.address_space<ub>>)
  %20 = hivm.hir.pointer_cast(%c12288_i64) : memref<1024xf32, #hivm.address_space<ub>>
  %21 = hivm.hir.pointer_cast(%c16384_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%16, %10, %19 : memref<1024xi1, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%20 : memref<1024xf32, #hivm.address_space<ub>>) temp_buffer(%21 : memref<8xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%9], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %subview_4 = memref.subview %20[0] [%14] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
  %subview_5 = memref.subview %reinterpret_cast_3[0] [%14] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
