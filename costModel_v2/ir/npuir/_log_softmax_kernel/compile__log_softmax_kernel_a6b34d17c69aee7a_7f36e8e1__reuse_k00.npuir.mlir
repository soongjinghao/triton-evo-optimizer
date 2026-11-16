// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_log_softmax_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c14976_i64 = arith.constant 14976 : i64
  %c12928_i64 = arith.constant 12928 : i64
  %c12896_i64 = arith.constant 12896 : i64
  %c12864_i64 = arith.constant 12864 : i64
  %c12832_i64 = arith.constant 12832 : i64
  %c12800_i64 = arith.constant 12800 : i64
  %c12768_i64 = arith.constant 12768 : i64
  %c12736_i64 = arith.constant 12736 : i64
  %c12704_i64 = arith.constant 12704 : i64
  %c12672_i64 = arith.constant 12672 : i64
  %c12640_i64 = arith.constant 12640 : i64
  %c12608_i64 = arith.constant 12608 : i64
  %c12576_i64 = arith.constant 12576 : i64
  %c12320_i64 = arith.constant 12320 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c19104_i64 = arith.constant 19104 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c15008_i64 = arith.constant 15008 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1024_i32 = arith.constant 1024 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<1024xf32, #hivm.address_space<ub>>)
  %6 = arith.extsi %arg5 : i32 to i64
  %7 = arith.extsi %arg6 : i32 to i64
  %8 = arith.index_cast %arg7 : i32 to index
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %9 = arith.index_cast %arg11 : i32 to index
    %10 = arith.index_cast %3 : i32 to index
    %11 = arith.index_cast %1 : i32 to index
    %12 = arith.index_cast %c48_i32 : i32 to index
    %13 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%9, %10, %12]
    %14 = arith.index_cast %13 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %15 = arith.select %14, %c0_i64_1, %c1_i64 : i64
    %16 = hivm.hir.pointer_cast(%c4096_i64, %c15008_i64) : memref<1024xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %17 = arith.divsi %arg11, %4 : i32
    %18 = arith.remsi %17, %arg8 : i32
    %19 = arith.extsi %18 : i32 to i64
    %20 = arith.muli %19, %6 : i64
    %21 = arith.muli %19, %7 : i64
    %22 = arith.index_cast %20 : i64 to index
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %15]
    hivm.hir.copy ins(%5 : memref<1024xf32, #hivm.address_space<ub>>) outs(%16 : memref<1024xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    %23:2 = scf.for %arg12 = %c0_i32 to %arg7 step %c1024_i32 iter_args(%arg13 = %cst, %arg14 = %cst_0) -> (f32, f32)  : i32 {
      %30 = hivm.hir.pointer_cast(%c8192_i64, %c19104_i64) : memref<1024xf32, #hivm.address_space<ub>>
      annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<1024xf32, #hivm.address_space<ub>>
      %31 = arith.index_cast %arg12 : i32 to index
      %32 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%22, %31]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%32], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %33 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%31, %8]
      %34 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%33, %31]
      %35 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%34, %31]
      %36 = arith.cmpi slt, %35, %c1024 : index
      %subview = memref.subview %reinterpret_cast[0] [%35] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %30[0] [%35] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %36 {
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%30 : memref<1024xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %subview_3 = memref.subview %16[0] [%35] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.copy ins(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      %37 = hivm.hir.pointer_cast(%c12288_i64) : memref<1xf32, #hivm.address_space<ub>>
      %38 = hivm.hir.pointer_cast(%c12320_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.vreduce <max> ins(%30 : memref<1024xf32, #hivm.address_space<ub>>) outs(%37 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%38 : memref<64xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %39 = memref.load %37[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %40 = hivm.hir.pointer_cast(%c12576_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %arg14, %40[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      %41 = hivm.hir.pointer_cast(%c12608_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %39, %41[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %42 = hivm.hir.pointer_cast(%c12640_i64) : memref<1xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.vcmp ins(%40, %40 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%42 : memref<1xi1, #hivm.address_space<ub>>)
      %43 = hivm.hir.pointer_cast(%c12640_i64) : memref<1xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%42 : memref<1xi1, #hivm.address_space<ub>>) outs(%43 : memref<1xi1, #hivm.address_space<ub>>)
      %44 = hivm.hir.pointer_cast(%c12672_i64) : memref<1xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcmp ins(%41, %41 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%44 : memref<1xi1, #hivm.address_space<ub>>)
      %45 = hivm.hir.pointer_cast(%c12672_i64) : memref<1xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%44 : memref<1xi1, #hivm.address_space<ub>>) outs(%45 : memref<1xi1, #hivm.address_space<ub>>)
      %46 = hivm.hir.pointer_cast(%c12704_i64) : memref<1xf32, #hivm.address_space<ub>>
      %47 = arith.maximumf %arg14, %39 : f32
      memref.store %47, %46[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %48 = hivm.hir.pointer_cast(%c12736_i64) : memref<1xf32, #hivm.address_space<ub>>
      %49 = hivm.hir.pointer_cast(%c12768_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vsel ins(%43, %41, %46 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%48 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%49 : memref<8xf32, #hivm.address_space<ub>>)
      %50 = hivm.hir.pointer_cast(%c12800_i64) : memref<1xf32, #hivm.address_space<ub>>
      %51 = hivm.hir.pointer_cast(%c12832_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%45, %40, %48 : memref<1xi1, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%50 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%51 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %52 = memref.load %50[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %53 = arith.subf %arg14, %52 : f32
      %54 = hivm.hir.pointer_cast(%c12864_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %53, %54[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %55 = hivm.hir.pointer_cast(%c12864_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vexp ins(%54 : memref<1xf32, #hivm.address_space<ub>>) outs(%55 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %56 = memref.load %55[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %57 = arith.mulf %arg13, %56 : f32
      %58 = arith.subf %cst, %52 : f32
      hivm.hir.vadd ins(%30, %58 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%30 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vexp ins(%30 : memref<1024xf32, #hivm.address_space<ub>>) outs(%30 : memref<1024xf32, #hivm.address_space<ub>>)
      %59 = hivm.hir.pointer_cast(%c12896_i64) : memref<1xf32, #hivm.address_space<ub>>
      %60 = hivm.hir.pointer_cast(%c12928_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%30 : memref<1024xf32, #hivm.address_space<ub>>) outs(%59 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%60 : memref<512xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %61 = memref.load %59[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %62 = arith.addf %57, %61 : f32
      scf.yield %62, %52 : f32, f32
    } {pipeline.veconly}
    %24 = hivm.hir.pointer_cast(%c14976_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %23#0, %24[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %25 = hivm.hir.pointer_cast(%c14976_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vln ins(%24 : memref<1xf32, #hivm.address_space<ub>>) outs(%25 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %26 = memref.load %25[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %27 = arith.addf %26, %23#1 : f32
    %28 = arith.subf %cst, %27 : f32
    hivm.hir.vadd ins(%16, %28 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%16 : memref<1024xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %29 = arith.index_cast %21 : i64 to index
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.for %arg12 = %c0_i32 to %arg7 step %c1024_i32  : i32 {
      %30 = arith.index_cast %arg12 : i32 to index
      %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %30]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [1024], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %32 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%30, %8]
      %33 = affine.min affine_map<()[s0, s1] -> (s1 + 1024, s0)>()[%32, %30]
      %34 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%33, %30]
      %subview = memref.subview %16[0] [%34] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %reinterpret_cast[0] [%34] [1] : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %15]
  } {autoblockify.subloop}
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
