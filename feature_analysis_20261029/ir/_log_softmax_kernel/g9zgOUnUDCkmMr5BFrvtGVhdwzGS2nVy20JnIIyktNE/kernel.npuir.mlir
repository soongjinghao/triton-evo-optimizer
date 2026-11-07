// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_log_softmax_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c15008_i64 = arith.constant 15008 : i64
  %c6816_i64 = arith.constant 6816 : i64
  %c6784_i64 = arith.constant 6784 : i64
  %c4736_i64 = arith.constant 4736 : i64
  %c4704_i64 = arith.constant 4704 : i64
  %c4672_i64 = arith.constant 4672 : i64
  %c4640_i64 = arith.constant 4640 : i64
  %c4608_i64 = arith.constant 4608 : i64
  %c4576_i64 = arith.constant 4576 : i64
  %c4544_i64 = arith.constant 4544 : i64
  %c4512_i64 = arith.constant 4512 : i64
  %c4480_i64 = arith.constant 4480 : i64
  %c4448_i64 = arith.constant 4448 : i64
  %c4416_i64 = arith.constant 4416 : i64
  %c4384_i64 = arith.constant 4384 : i64
  %c4128_i64 = arith.constant 4128 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c10912_i64 = arith.constant 10912 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c1024_i32 = arith.constant 1024 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg8 : i32
  %7 = arith.extsi %6 : i32 to i64
  %8 = arith.extsi %arg5 : i32 to i64
  %9 = arith.muli %7, %8 : i64
  %10 = arith.index_cast %9 : i64 to index
  %11 = arith.extsi %arg6 : i32 to i64
  %12 = arith.muli %7, %11 : i64
  %13 = arith.index_cast %12 : i64 to index
  %14 = arith.index_cast %arg7 : i32 to index
  %15:2 = scf.for %arg11 = %c0_i32 to %arg7 step %c1024_i32 iter_args(%arg12 = %cst_0, %arg13 = %cst) -> (f32, f32)  : i32 {
    %20 = hivm.hir.pointer_cast(%c0_i64, %c10912_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %21 = arith.index_cast %arg11 : i32 to index
    %22 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%10, %21]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%22], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %23 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%21, %14]
    %24 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%23, %21]
    %25 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%24, %21]
    %26 = arith.cmpi slt, %25, %c1024 : index
    %subview = memref.subview %reinterpret_cast[0] [%25] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_1 = memref.subview %20[0] [%25] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %26 {
      hivm.hir.vbrc ins(%cst : f32) outs(%20 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %27 = hivm.hir.pointer_cast(%c4096_i64) : memref<1xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c4128_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vreduce <max> ins(%20 : memref<1024xf32, #hivm.address_space<ub>>) outs(%27 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%28 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %29 = memref.load %27[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c4384_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %arg13, %30[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    %31 = hivm.hir.pointer_cast(%c4416_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %29, %31[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %32 = hivm.hir.pointer_cast(%c4448_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.vcmp ins(%30, %30 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%32 : memref<1xi1, #hivm.address_space<ub>>)
    %33 = hivm.hir.pointer_cast(%c4448_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%32 : memref<1xi1, #hivm.address_space<ub>>) outs(%33 : memref<1xi1, #hivm.address_space<ub>>)
    %34 = hivm.hir.pointer_cast(%c4480_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%31, %31 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%34 : memref<1xi1, #hivm.address_space<ub>>)
    %35 = hivm.hir.pointer_cast(%c4480_i64) : memref<1xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%34 : memref<1xi1, #hivm.address_space<ub>>) outs(%35 : memref<1xi1, #hivm.address_space<ub>>)
    %36 = hivm.hir.pointer_cast(%c4512_i64) : memref<1xf32, #hivm.address_space<ub>>
    %37 = arith.maxnumf %arg13, %29 : f32
    memref.store %37, %36[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %38 = hivm.hir.pointer_cast(%c4544_i64) : memref<1xf32, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c4576_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%33, %31, %36 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%38 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%39 : memref<8xf32, #hivm.address_space<ub>>)
    %40 = hivm.hir.pointer_cast(%c4608_i64) : memref<1xf32, #hivm.address_space<ub>>
    %41 = hivm.hir.pointer_cast(%c4640_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%35, %30, %38 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%40 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%41 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %42 = memref.load %40[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %43 = arith.subf %arg13, %42 : f32
    %44 = hivm.hir.pointer_cast(%c4672_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %43, %44[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %45 = hivm.hir.pointer_cast(%c4672_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vexp ins(%44 : memref<1xf32, #hivm.address_space<ub>>) outs(%45 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %46 = memref.load %45[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %47 = arith.mulf %arg12, %46 : f32
    %48 = arith.subf %cst_0, %42 : f32
    hivm.hir.vadd ins(%20, %48 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%20 : memref<1024xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%20 : memref<1024xf32, #hivm.address_space<ub>>) outs(%20 : memref<1024xf32, #hivm.address_space<ub>>)
    %49 = hivm.hir.pointer_cast(%c4704_i64) : memref<1xf32, #hivm.address_space<ub>>
    %50 = hivm.hir.pointer_cast(%c4736_i64) : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%20 : memref<1024xf32, #hivm.address_space<ub>>) outs(%49 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%50 : memref<512xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %51 = memref.load %49[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %52 = arith.addf %47, %51 : f32
    scf.yield %52, %42 : f32, f32
  }
  %16 = hivm.hir.pointer_cast(%c6784_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %15#0, %16[%c0] : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %17 = hivm.hir.pointer_cast(%c6784_i64) : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vln ins(%16 : memref<1xf32, #hivm.address_space<ub>>) outs(%17 : memref<1xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  %18 = memref.load %17[%c0] : memref<1xf32, #hivm.address_space<ub>>
  %19 = arith.addf %18, %15#1 : f32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg11 = %c0_i32 to %arg7 step %c1024_i32  : i32 {
    %20 = arith.index_cast %arg11 : i32 to index
    %21 = arith.index_cast %c0_i32 : i32 to index
    %22 = arith.index_cast %arg7 : i32 to index
    %23 = arith.index_cast %c1024_i32 : i32 to index
    %24 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%20, %21, %23]
    %25 = arith.index_cast %24 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %26 = arith.select %25, %c0_i64_1, %c1_i64 : i64
    %27 = hivm.hir.pointer_cast(%c6816_i64, %c15008_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c6816_i64, %c15008_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    %29 = arith.index_cast %arg11 : i32 to index
    %30 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%10, %29]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%30], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %31 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%29, %14]
    %32 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%31, %29]
    %33 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%32, %29]
    %34 = arith.cmpi slt, %33, %c1024 : index
    %subview = memref.subview %reinterpret_cast[0] [%33] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %28[0] [%33] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %34 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%28 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %26]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %35 = arith.subf %cst_0, %19 : f32
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vadd ins(%28, %35 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%27 : memref<1024xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %36 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%13, %29]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%36], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %27[0] [%33] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_5 = memref.subview %reinterpret_cast_3[0] [%33] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %26]
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
