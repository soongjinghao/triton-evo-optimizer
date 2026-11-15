// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @rwkv_ka_fusion_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c448_i64 = arith.constant 448 : i64
  %c256_i64 = arith.constant 256 : i64
  %c224_i64 = arith.constant 224 : i64
  %c416_i64 = arith.constant 416 : i64
  %c192_i64 = arith.constant 192 : i64
  %c160_i64 = arith.constant 160 : i64
  %c128_i64 = arith.constant 128 : i64
  %c384_i64 = arith.constant 384 : i64
  %c96_i64 = arith.constant 96 : i64
  %c352_i64 = arith.constant 352 : i64
  %c64_i64 = arith.constant 64 : i64
  %c320_i64 = arith.constant 320 : i64
  %c32_i64 = arith.constant 32 : i64
  %c288_i64 = arith.constant 288 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant -1.000000e+00 : f32
  %cst_0 = arith.constant 9.99999996E-13 : f32
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c8 = arith.constant 8 : index
  %cst_2 = arith.constant 0.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  %5 = arith.muli %arg10, %arg11 : i32
  %6 = arith.index_cast %5 : i32 to index
  %7 = arith.index_cast %arg11 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %8 = arith.divsi %arg17, %4 : i32
    %9 = arith.remsi %8, %arg14 : i32
    %10 = arith.muli %9, %c16_i32 : i32
    scf.for %arg18 = %c0_i32 to %arg12 step %c1_i32  : i32 {
      %11 = arith.index_cast %arg18 : i32 to index
      %12 = arith.index_cast %c0_i32 : i32 to index
      %13 = arith.index_cast %arg12 : i32 to index
      %14 = arith.index_cast %c1_i32 : i32 to index
      %15 = arith.index_cast %arg17 : i32 to index
      %16 = arith.index_cast %3 : i32 to index
      %17 = arith.index_cast %1 : i32 to index
      %18 = arith.index_cast %c48_i32 : i32 to index
      %19 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%11, %12, %13, %14, %15, %16, %18]
      %20 = arith.index_cast %19 : index to i1
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %21 = arith.select %20, %c3_i64, %c4_i64 : i64
      %22 = arith.index_cast %19 : index to i1
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %23 = arith.select %22, %c1_i64, %c2_i64 : i64
      %24 = hivm.hir.pointer_cast(%c192_i64, %c416_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %25 = hivm.hir.pointer_cast(%c256_i64, %c448_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %26 = hivm.hir.pointer_cast(%c192_i64, %c416_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %27 = hivm.hir.pointer_cast(%c96_i64, %c384_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %28 = hivm.hir.pointer_cast(%c64_i64, %c352_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %29 = hivm.hir.pointer_cast(%c32_i64, %c320_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %30 = hivm.hir.pointer_cast(%c0_i64, %c288_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %31 = arith.muli %arg18, %arg13 : i32
      %32 = arith.addi %10, %31 : i32
      %33 = arith.index_cast %32 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%33], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %34 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%33, %6]
      %35 = affine.min affine_map<()[s0, s1] -> (s1 + 8, s0)>()[%34, %33]
      %36 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%35, %33]
      %37 = arith.cmpi slt, %36, %c8 : index
      %subview = memref.subview %reinterpret_cast[0] [%36] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %30[0] [%36] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %37 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_2 : f32) outs(%30 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %23]
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %reinterpret_cast_4[0] [%36] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %29[0] [%36] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %37 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_2 : f32) outs(%29 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %21]
      hivm.hir.load ins(%subview_5 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %38 = arith.index_cast %31 : i32 to index
      %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%38], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %39 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%38, %7]
      %40 = affine.min affine_map<()[s0, s1] -> (s1 + 8, s0)>()[%39, %38]
      %41 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%40, %38]
      %42 = arith.cmpi slt, %41, %c8 : index
      %subview_8 = memref.subview %reinterpret_cast_7[0] [%41] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %28[0] [%41] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %42 {
        hivm.hir.vbrc ins(%cst_2 : f32) outs(%28 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_8 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_9 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_11 = memref.subview %reinterpret_cast_10[0] [%41] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_12 = memref.subview %27[0] [%41] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %42 {
        hivm.hir.vbrc ins(%cst_2 : f32) outs(%27 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%30, %27 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%27 : memref<8xf32, #hivm.address_space<ub>>)
      %43 = hivm.hir.pointer_cast(%c128_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%27, %27 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%43 : memref<8xf32, #hivm.address_space<ub>>)
      %44 = hivm.hir.pointer_cast(%c160_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%43 : memref<8xf32, #hivm.address_space<ub>>) outs(%44 : memref<1xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %45 = memref.load %44[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %46 = arith.addf %45, %cst_0 : f32
      %47 = math.sqrt %46 : f32
      %48 = hivm.hir.pointer_cast(%c224_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %23]
      hivm.hir.vdiv ins(%27, %47 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%26 : memref<8xf32, #hivm.address_space<ub>>) temp_buffer(%48 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_13 = memref.reinterpret_cast %arg8 to offset: [%33], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_14 = memref.subview %26[0] [%36] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_15 = memref.subview %reinterpret_cast_13[0] [%36] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%subview_14 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_15 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vadd ins(%29, %cst : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%25 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%25, %28 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%25 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%25, %cst_1 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%25 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%30, %25 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%25 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %23]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%26, %29 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%24 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %21]
      %reinterpret_cast_16 = memref.reinterpret_cast %arg7 to offset: [%33], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_17 = memref.subview %25[0] [%36] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_18 = memref.subview %reinterpret_cast_16[0] [%36] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_18 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      %reinterpret_cast_19 = memref.reinterpret_cast %arg9 to offset: [%33], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_20 = memref.subview %24[0] [%36] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_21 = memref.subview %reinterpret_cast_19[0] [%36] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%subview_20 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_21 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %23]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
