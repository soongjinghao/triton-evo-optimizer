// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @reduce_segments(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i64 {tt.divisibility = 16 : i32}, %arg11: i64, %arg12: i64 {tt.divisibility = 16 : i32}, %arg13: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, true, false, false, false, true, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c800_i64 = arith.constant 800 : i64
  %c704_i64 = arith.constant 704 : i64
  %c1152_i64 = arith.constant 1152 : i64
  %c672_i64 = arith.constant 672 : i64
  %c416_i64 = arith.constant 416 : i64
  %c1056_i64 = arith.constant 1056 : i64
  %c320_i64 = arith.constant 320 : i64
  %c960_i64 = arith.constant 960 : i64
  %c224_i64 = arith.constant 224 : i64
  %c192_i64 = arith.constant 192 : i64
  %c160_i64 = arith.constant 160 : i64
  %c928_i64 = arith.constant 928 : i64
  %c128_i64 = arith.constant 128 : i64
  %c896_i64 = arith.constant 896 : i64
  %c96_i64 = arith.constant 96 : i64
  %c864_i64 = arith.constant 864 : i64
  %c64_i64 = arith.constant 64 : i64
  %c832_i64 = arith.constant 832 : i64
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %c6_i64 = arith.constant 6 : i64
  %c4_i32 = arith.constant 4 : i32
  %c3_i32 = arith.constant 3 : i32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c48_i64 = arith.constant 48 : i64
  %c24_i32 = arith.constant 24 : i32
  %c12_i32 = arith.constant 12 : i32
  %c2_i32 = arith.constant 2 : i32
  %c0_i32 = arith.constant 0 : i32
  %c11_i32 = arith.constant 11 : i32
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%5 : memref<8xf32, #hivm.address_space<ub>>)
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
    %15 = hivm.hir.pointer_cast(%c672_i64, %c1152_i64) : memref<8xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c320_i64, %c1056_i64) : memref<3x8xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<3x8xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c224_i64, %c960_i64) : memref<3x8xf32, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<3x8xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c128_i64, %c928_i64) : memref<3xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<3xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c96_i64, %c896_i64) : memref<3xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<3xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c64_i64, %c864_i64) : memref<3xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<3xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c32_i64, %c832_i64) : memref<3xf32, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<3xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %22 = arith.divsi %arg17, %arg16 : i32
    %23 = arith.remsi %22, %arg15 : i32
    %24 = arith.divsi %arg17, %4 : i32
    %25 = arith.remsi %24, %arg14 : i32
    %26:2 = scf.while (%arg18 = %c0_i32, %arg19 = %arg8) : (i32, i32) -> (i32, i32) {
      %65 = arith.cmpi slt, %arg18, %arg19 : i32
      scf.condition(%65) %arg18, %arg19 : i32, i32
    } do {
    ^bb0(%arg18: i32, %arg19: i32):
      %65 = arith.addi %arg18, %arg19 : i32
      %66 = arith.divsi %65, %c2_i32 : i32
      %67 = arith.index_cast %66 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg13 to offset: [%67], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %68 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %69 = arith.cmpi sle, %68, %25 : i32
      %70 = arith.select %69, %arg19, %66 : i32
      %71 = scf.if %69 -> (i32) {
        %72 = arith.addi %66, %c1_i32 : i32
        scf.yield %72 : i32
      } else {
        scf.yield %arg18 : i32
      }
      scf.yield %71, %70 : i32, i32
    }
    %27 = arith.subi %26#0, %c1_i32 : i32
    %28 = arith.index_cast %27 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%28], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %29 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %30 = arith.addi %29, %c11_i32 : i32
    %31 = arith.divsi %30, %c12_i32 : i32
    %32 = arith.muli %31, %c4_i32 : i32
    %33 = arith.addi %29, %32 : i32
    %34 = arith.subi %33, %c1_i32 : i32
    %35 = arith.divsi %34, %32 : i32
    %36 = arith.extsi %25 : i32 to i64
    %37 = arith.muli %36, %c6_i64 : i64
    %38 = arith.muli %23, %c3_i32 : i32
    %39 = arith.extsi %38 : i32 to i64
    %40 = arith.addi %37, %39 : i64
    %41 = arith.cmpi eq, %35, %c3_i32 : i32
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %12]
    %42:2 = scf.if %41 -> (memref<3xf32, #hivm.address_space<ub>>, memref<3xf32, #hivm.address_space<ub>>) {
      %65 = arith.index_cast %40 : i64 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [3], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<3xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%reinterpret_cast_5 : memref<3xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%21 : memref<3xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%65], sizes: [3], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<3xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%reinterpret_cast_6 : memref<3xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%20 : memref<3xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      scf.yield %20, %21 : memref<3xf32, #hivm.address_space<ub>>, memref<3xf32, #hivm.address_space<ub>>
    } else {
      %65 = arith.index_cast %40 : i64 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [3], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<3xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %66 = arith.index_cast %35 : i32 to index
      %67 = affine.max affine_map<()[s0] -> (0, s0)>()[%66]
      %68 = affine.min affine_map<()[s0] -> (3, s0)>()[%67]
      %69 = arith.cmpi slt, %68, %c3 : index
      %subview_6 = memref.subview %reinterpret_cast_5[0] [%68] [1] : memref<3xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %19[0] [%68] [1] : memref<3xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %69 {
        hivm.hir.vbrc ins(%cst : f32) outs(%19 : memref<3xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%65], sizes: [3], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<3xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_9 = memref.subview %reinterpret_cast_8[0] [%68] [1] : memref<3xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_10 = memref.subview %18[0] [%68] [1] : memref<3xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %69 {
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%18 : memref<3xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_9 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_10 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      scf.yield %18, %19 : memref<3xf32, #hivm.address_space<ub>>, memref<3xf32, #hivm.address_space<ub>>
    }
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %43 = hivm.hir.pointer_cast(%c160_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vreduce <max> ins(%42#1 : memref<3xf32, #hivm.address_space<ub>>) outs(%43 : memref<1xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %44 = memref.load %43[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %45 = arith.subf %cst_0, %44 : f32
    hivm.hir.vadd ins(%42#1, %45 : memref<3xf32, #hivm.address_space<ub>>, f32) outs(%21 : memref<3xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%21 : memref<3xf32, #hivm.address_space<ub>>) outs(%21 : memref<3xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%42#0, %21 : memref<3xf32, #hivm.address_space<ub>>, memref<3xf32, #hivm.address_space<ub>>) outs(%20 : memref<3xf32, #hivm.address_space<ub>>)
    %46 = hivm.hir.pointer_cast(%c192_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%20 : memref<3xf32, #hivm.address_space<ub>>) outs(%46 : memref<1xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %47 = memref.load %46[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    %48 = arith.muli %36, %c48_i64 : i64
    %49 = arith.muli %23, %c24_i32 : i32
    %50 = arith.extsi %49 : i32 to i64
    %51 = arith.addi %48, %50 : i64
    %52 = scf.if %41 -> (memref<3x8xf32, #hivm.address_space<ub>>) {
      %65 = arith.index_cast %51 : i64 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%65], sizes: [3, 8], strides: [8, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<3x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_6 = memref.collapse_shape %reinterpret_cast_5 [[0, 1]] : memref<3x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>> into memref<24xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_7 = memref.collapse_shape %17 [[0, 1]] : memref<3x8xf32, #hivm.address_space<ub>> into memref<24xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%collapse_shape_6 : memref<24xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_7 : memref<24xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      scf.yield %17 : memref<3x8xf32, #hivm.address_space<ub>>
    } else {
      %65 = arith.index_cast %51 : i64 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%65], sizes: [3, 8], strides: [8, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<3x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>
      %66 = arith.index_cast %35 : i32 to index
      %67 = affine.max affine_map<()[s0] -> (0, s0)>()[%66]
      %68 = affine.min affine_map<()[s0] -> (3, s0)>()[%67]
      %69 = arith.cmpi slt, %68, %c3 : index
      %subview_6 = memref.subview %reinterpret_cast_5[0, 0] [%68, 8] [1, 1] : memref<3x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %16[0, 0] [%68, 8] [1, 1] : memref<3x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.if %69 {
        %collapse_shape_10 = memref.collapse_shape %16 [[0, 1]] : memref<3x8xf32, #hivm.address_space<ub>> into memref<24xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_10 : memref<24xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape_8 = memref.collapse_shape %subview_6 [[0, 1]] : memref<?x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_9 = memref.collapse_shape %subview_7 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%collapse_shape_8 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_9 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      scf.yield %16 : memref<3x8xf32, #hivm.address_space<ub>>
    }
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %expand_shape = memref.expand_shape %21 [[0, 1]] output_shape [3, 1] : memref<3xf32, #hivm.address_space<ub>> into memref<3x1xf32, #hivm.address_space<ub>>
    %53 = hivm.hir.pointer_cast(%c416_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%52, %expand_shape : memref<3x8xf32, #hivm.address_space<ub>>, memref<3x1xf32, #hivm.address_space<ub>>) outs(%17 : memref<3x8xf32, #hivm.address_space<ub>>) temp_buffer(%53 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %12]
    %54 = hivm.hir.pointer_cast(%c672_i64, %c1152_i64) : memref<1x8xf32, #hivm.address_space<ub>>
    %55 = hivm.hir.pointer_cast(%c704_i64) : memref<24xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %14]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%17 : memref<3x8xf32, #hivm.address_space<ub>>) outs(%54 : memref<1x8xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<24xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape = memref.collapse_shape %54 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
    %56 = arith.cmpf oeq, %47, %cst_0 : f32
    %57 = hivm.hir.pointer_cast(%c800_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%collapse_shape, %47 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%15 : memref<8xf32, #hivm.address_space<ub>>) temp_buffer(%57 : memref<8xf32, #hivm.address_space<ub>>)
    %58 = arith.index_castui %56 : i1 to index
    %59 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%58]
    %subview = memref.subview %5[0] [%59] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_3 = memref.subview %15[0] [%59] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%subview : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %60 = arith.muli %36, %arg10 : i64
    %61 = arith.extsi %23 : i32 to i64
    %62 = arith.muli %61, %arg11 : i64
    %63 = arith.addi %60, %62 : i64
    %64 = arith.index_cast %63 : i64 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%64], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%15 : memref<8xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
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
