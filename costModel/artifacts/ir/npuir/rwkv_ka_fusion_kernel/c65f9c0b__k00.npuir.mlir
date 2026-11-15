// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @rwkv_ka_fusion_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c224_i64 = arith.constant 224 : i64
  %c384_i64 = arith.constant 384 : i64
  %c192_i64 = arith.constant 192 : i64
  %c160_i64 = arith.constant 160 : i64
  %c128_i64 = arith.constant 128 : i64
  %c352_i64 = arith.constant 352 : i64
  %c96_i64 = arith.constant 96 : i64
  %c320_i64 = arith.constant 320 : i64
  %c64_i64 = arith.constant 64 : i64
  %c288_i64 = arith.constant 288 : i64
  %c32_i64 = arith.constant 32 : i64
  %c256_i64 = arith.constant 256 : i64
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
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
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
      %c0_i64_3 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %21 = arith.select %20, %c0_i64_3, %c1_i64 : i64
      %22 = arith.index_cast %19 : index to i1
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %23 = arith.select %22, %c3_i64, %c4_i64 : i64
      %24 = arith.index_cast %19 : index to i1
      %c1_i64_4 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %25 = arith.select %24, %c1_i64_4, %c2_i64 : i64
      %26 = hivm.hir.pointer_cast(%c192_i64, %c384_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %27 = hivm.hir.pointer_cast(%c64_i64, %c320_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %28 = hivm.hir.pointer_cast(%c192_i64, %c384_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %29 = hivm.hir.pointer_cast(%c96_i64, %c352_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %30 = hivm.hir.pointer_cast(%c64_i64, %c320_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %31 = hivm.hir.pointer_cast(%c32_i64, %c288_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %32 = hivm.hir.pointer_cast(%c0_i64, %c256_i64) : memref<8xf32, #hivm.address_space<ub>>
      annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
      %33 = arith.muli %arg18, %arg13 : i32
      %34 = arith.addi %10, %33 : i32
      %35 = arith.index_cast %34 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %36 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%35, %6]
      %37 = affine.min affine_map<()[s0, s1] -> (s1 + 8, s0)>()[%36, %35]
      %38 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%37, %35]
      %39 = arith.cmpi slt, %38, %c8 : index
      %subview = memref.subview %reinterpret_cast[0] [%38] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_5 = memref.subview %32[0] [%38] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %39 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_2 : f32) outs(%32 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %25]
      hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%35], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %reinterpret_cast_6[0] [%38] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %31[0] [%38] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %39 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_2 : f32) outs(%31 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %23]
      hivm.hir.load ins(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %40 = arith.index_cast %33 : i32 to index
      %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%40], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %41 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%40, %7]
      %42 = affine.min affine_map<()[s0, s1] -> (s1 + 8, s0)>()[%41, %40]
      %43 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%42, %40]
      %44 = arith.cmpi slt, %43, %c8 : index
      %subview_10 = memref.subview %reinterpret_cast_9[0] [%43] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_11 = memref.subview %30[0] [%43] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %44 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst_2 : f32) outs(%30 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %21]
      hivm.hir.load ins(%subview_10 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_11 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%40], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_13 = memref.subview %reinterpret_cast_12[0] [%43] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_14 = memref.subview %29[0] [%43] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %44 {
        hivm.hir.vbrc ins(%cst_2 : f32) outs(%29 : memref<8xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_13 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_14 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_2 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%32, %29 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%29 : memref<8xf32, #hivm.address_space<ub>>)
      %45 = hivm.hir.pointer_cast(%c128_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%29, %29 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%45 : memref<8xf32, #hivm.address_space<ub>>)
      %46 = hivm.hir.pointer_cast(%c160_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%45 : memref<8xf32, #hivm.address_space<ub>>) outs(%46 : memref<1xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %47 = memref.load %46[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %48 = arith.addf %47, %cst_0 : f32
      %49 = math.sqrt %48 : f32
      %50 = arith.divf %cst_1, %49 : f32
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %25]
      hivm.hir.vmul ins(%29, %50 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%28 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_15 = memref.reinterpret_cast %arg8 to offset: [%35], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_16 = memref.subview %28[0] [%38] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_17 = memref.subview %reinterpret_cast_15[0] [%38] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%subview_16 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_17 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %51 = hivm.hir.pointer_cast(%c224_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%31, %cst : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%51 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%51, %30 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%30 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%30, %cst_1 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%30 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%32, %30 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%27 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %25]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%28, %31 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%26 : memref<8xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %23]
      %reinterpret_cast_18 = memref.reinterpret_cast %arg7 to offset: [%35], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_19 = memref.subview %27[0] [%38] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_20 = memref.subview %reinterpret_cast_18[0] [%38] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_19 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_20 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %21]
      %reinterpret_cast_21 = memref.reinterpret_cast %arg9 to offset: [%35], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_22 = memref.subview %26[0] [%38] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_23 = memref.subview %reinterpret_cast_21[0] [%38] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%subview_22 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_23 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %25]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
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
