// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @reduce_segments(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i64 {tt.divisibility = 16 : i32}, %arg11: i64 {tt.divisibility = 16 : i32}, %arg12: i64 {tt.divisibility = 16 : i32}, %arg13: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, true, false, false, false, true, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c800_i64 = arith.constant 800 : i64
  %c672_i64 = arith.constant 672 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c608_i64 = arith.constant 608 : i64
  %c352_i64 = arith.constant 352 : i64
  %c896_i64 = arith.constant 896 : i64
  %c224_i64 = arith.constant 224 : i64
  %c192_i64 = arith.constant 192 : i64
  %c160_i64 = arith.constant 160 : i64
  %c864_i64 = arith.constant 864 : i64
  %c96_i64 = arith.constant 96 : i64
  %c832_i64 = arith.constant 832 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %c32_i32 = arith.constant 32 : i32
  %c128_i64 = arith.constant 128 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c8_i64 = arith.constant 8 : i64
  %c8_i32 = arith.constant 8 : i32
  %c2_i32 = arith.constant 2 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  %c15_i32 = arith.constant 15 : i32
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %cst_0 = arith.constant 0xFF800000 : f32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<16xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg17 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %12 = arith.select %11, %c1_i64, %c2_i64 : i64
    %13 = arith.index_cast %10 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64_2 = arith.constant 1 : i64
    %14 = arith.select %13, %c0_i64_1, %c1_i64_2 : i64
    %15 = hivm.hir.pointer_cast(%c608_i64, %c1024_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c224_i64, %c896_i64) : memref<2x16xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<2x16xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c128_i64, %c864_i64) : memref<2xf32, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<2xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c64_i64, %c832_i64) : memref<2xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<2xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %19 = arith.divsi %arg17, %arg16 : i32
    %20 = arith.remsi %19, %arg15 : i32
    %21 = arith.divsi %arg17, %4 : i32
    %22 = arith.remsi %21, %arg14 : i32
    %23:2 = scf.while (%arg18 = %c0_i32, %arg19 = %arg8) : (i32, i32) -> (i32, i32) {
      %66 = arith.cmpi slt, %arg18, %arg19 : i32
      scf.condition(%66) %arg18, %arg19 : i32, i32
    } do {
    ^bb0(%arg18: i32, %arg19: i32):
      %66 = arith.addi %arg18, %arg19 : i32
      %67 = arith.divsi %66, %c2_i32 : i32
      %68 = arith.index_cast %67 : i32 to index
      %reinterpret_cast_16 = memref.reinterpret_cast %arg13 to offset: [%68], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %69 = memref.load %reinterpret_cast_16[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %70 = arith.cmpi sle, %69, %22 : i32
      %71 = arith.select %70, %arg19, %67 : i32
      %72 = scf.if %70 -> (i32) {
        %73 = arith.addi %67, %c1_i32 : i32
        scf.yield %73 : i32
      } else {
        scf.yield %arg18 : i32
      }
      scf.yield %72, %71 : i32, i32
    }
    %24 = arith.subi %23#0, %c1_i32 : i32
    %25 = arith.index_cast %24 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%25], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %26 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %27 = arith.addi %26, %c15_i32 : i32
    %28 = arith.divsi %27, %c16_i32 : i32
    %29 = arith.muli %28, %c8_i32 : i32
    %30 = arith.addi %26, %29 : i32
    %31 = arith.subi %30, %c1_i32 : i32
    %32 = arith.divsi %31, %29 : i32
    %33 = arith.extsi %22 : i32 to i64
    %34 = arith.muli %33, %c8_i64 : i64
    %35 = arith.muli %20, %c2_i32 : i32
    %36 = arith.extsi %35 : i32 to i64
    %37 = arith.addi %34, %36 : i64
    %38 = arith.index_cast %37 : i64 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%38], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %39 = arith.index_cast %32 : i32 to index
    %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
    %41 = affine.min affine_map<()[s0] -> (2, s0)>()[%40]
    %42 = arith.cmpi slt, %41, %c2 : index
    %subview = memref.subview %reinterpret_cast_3[0] [%41] [1] : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %18[0] [%41] [1] : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %42 {
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%18 : memref<2xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %12]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %43 = hivm.hir.pointer_cast(%c96_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vreduce <max> ins(%18 : memref<2xf32, #hivm.address_space<ub>>) outs(%43 : memref<1xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %44 = memref.load %43[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%38], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%41] [1] : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %17[0] [%41] [1] : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %42 {
      hivm.hir.vbrc ins(%cst : f32) outs(%17 : memref<2xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %45 = arith.subf %cst, %44 : f32
    hivm.hir.vadd ins(%18, %45 : memref<2xf32, #hivm.address_space<ub>>, f32) outs(%18 : memref<2xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%18 : memref<2xf32, #hivm.address_space<ub>>) outs(%18 : memref<2xf32, #hivm.address_space<ub>>)
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%17, %18 : memref<2xf32, #hivm.address_space<ub>>, memref<2xf32, #hivm.address_space<ub>>) outs(%17 : memref<2xf32, #hivm.address_space<ub>>)
    %46 = hivm.hir.pointer_cast(%c160_i64) : memref<1xf32, #hivm.address_space<ub>>
    %47 = hivm.hir.pointer_cast(%c192_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%17 : memref<2xf32, #hivm.address_space<ub>>) outs(%46 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%47 : memref<1xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %48 = memref.load %46[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    %49 = arith.muli %33, %c128_i64 : i64
    %50 = arith.muli %20, %c32_i32 : i32
    %51 = arith.extsi %50 : i32 to i64
    %52 = arith.addi %49, %51 : i64
    %53 = arith.index_cast %52 : i64 to index
    %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%53], sizes: [2, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<2x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_9 = memref.subview %reinterpret_cast_8[0, 0] [%41, 16] [1, 1] : memref<2x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_10 = memref.subview %16[0, 0] [%41, 16] [1, 1] : memref<2x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
    scf.if %42 {
      %collapse_shape_16 = memref.collapse_shape %16 [[0, 1]] : memref<2x16xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_16 : memref<32xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape = memref.collapse_shape %subview_9 [[0, 1]] : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_11 = memref.collapse_shape %subview_10 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_11 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %expand_shape = memref.expand_shape %18 [[0, 1]] output_shape [2, 1] : memref<2xf32, #hivm.address_space<ub>> into memref<2x1xf32, #hivm.address_space<ub>>
    %54 = hivm.hir.pointer_cast(%c352_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%16, %expand_shape : memref<2x16xf32, #hivm.address_space<ub>>, memref<2x1xf32, #hivm.address_space<ub>>) outs(%16 : memref<2x16xf32, #hivm.address_space<ub>>) temp_buffer(%54 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %12]
    %55 = hivm.hir.pointer_cast(%c608_i64, %c1024_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %56 = hivm.hir.pointer_cast(%c672_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %14]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%16 : memref<2x16xf32, #hivm.address_space<ub>>) outs(%55 : memref<1x16xf32, #hivm.address_space<ub>>) temp_buffer(%56 : memref<32xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_12 = memref.collapse_shape %55 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
    %57 = arith.cmpf oeq, %48, %cst : f32
    %58 = hivm.hir.pointer_cast(%c800_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%collapse_shape_12, %48 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%15 : memref<16xf32, #hivm.address_space<ub>>) temp_buffer(%58 : memref<8xf32, #hivm.address_space<ub>>)
    %59 = arith.index_castui %57 : i1 to index
    %60 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%59]
    %subview_13 = memref.subview %5[0] [%60] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_14 = memref.subview %15[0] [%60] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%subview_13 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_14 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %61 = arith.muli %33, %arg10 : i64
    %62 = arith.extsi %20 : i32 to i64
    %63 = arith.muli %62, %arg11 : i64
    %64 = arith.addi %61, %63 : i64
    %65 = arith.index_cast %64 : i64 to index
    %reinterpret_cast_15 = memref.reinterpret_cast %arg3 to offset: [%65], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%15 : memref<16xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_15 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %14]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
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
