// -----// IR Dump After PlanMemory Failed (hivm-plan-memory) //----- //
func.func @layer_norm_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: f32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 4.8828125E-4 : f32
  %c2048_i32 = arith.constant 2048 : i32
  %c8_i32 = arith.constant 8 : i32
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  %5 = arith.index_cast %arg9 : i32 to index
  %6 = arith.index_cast %arg10 : i32 to index
  %7 = arith.index_cast %arg12 : i32 to index
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %8 = arith.divsi %arg17, %arg16 : i32
    %9 = arith.remsi %8, %arg15 : i32
    %10 = arith.divsi %arg17, %4 : i32
    %11 = arith.remsi %10, %arg14 : i32
    %12 = arith.muli %11, %c8_i32 : i32
    %13 = arith.muli %9, %c2048_i32 : i32
    %14 = arith.index_cast %12 : i32 to index
    %15 = arith.index_cast %13 : i32 to index
    %16 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%15, %14, %5]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [8, 2048], strides: [%5, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %17 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%15, %14, %6]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%17], sizes: [8, 2048], strides: [%6, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<8x2048xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<8x2048xf32, #hivm.address_space<ub>>
    %18 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%14, %7]
    %19 = affine.min affine_map<()[s0, s1] -> (s1 + 8, s0)>()[%18, %14]
    %20 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%19, %14]
    %21 = affine.min affine_map<()[s0, s1] -> (8, s0 - s1)>()[%19, %14]
    %22 = affine.max affine_map<()[s0] -> (0, s0)>()[%21]
    %23 = arith.cmpi slt, %22, %c8 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%22, 2048] [1, 1] : memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %alloc[0, 0] [%22, 2048] [1, 1] : memref<8x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    scf.if %23 {
      %collapse_shape_42 = memref.collapse_shape %alloc [[0, 1]] : memref<8x2048xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape_42 : memref<16384xf32, #hivm.address_space<ub>>)
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x1xf32, #hivm.address_space<ub>>
    %alloc_5 = memref.alloc() : memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vreduce <sum> ins(%alloc : memref<8x2048xf32, #hivm.address_space<ub>>) outs(%alloc_4 : memref<8x1xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_5 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape = memref.collapse_shape %alloc_4 [[0, 1]] : memref<8x1xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<8x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_7 = memref.collapse_shape %alloc [[0, 1]] : memref<8x2048xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
    %collapse_shape_8 = memref.collapse_shape %alloc_6 [[0, 1]] : memref<8x2048xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_7, %collapse_shape_7 : memref<16384xf32, #hivm.address_space<ub>>, memref<16384xf32, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<16384xf32, #hivm.address_space<ub>>)
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<8x1xf32, #hivm.address_space<ub>>
    %alloc_10 = memref.alloc() : memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vreduce <sum> ins(%alloc_6 : memref<8x2048xf32, #hivm.address_space<ub>>) outs(%alloc_9 : memref<8x1xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_10 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_11 = memref.collapse_shape %alloc_9 [[0, 1]] : memref<8x1xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_12 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape, %cst_0 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_12 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_11, %cst_0 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_13 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc_12, %alloc_12 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vsub ins(%alloc_13, %alloc_14 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%alloc_15 : memref<8xf32, #hivm.address_space<ub>>)
    %24 = arith.muli %9, %arg12 : i32
    %25 = arith.index_cast %24 : i32 to index
    %26 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%25, %14]
    %reinterpret_cast_16 = memref.reinterpret_cast %arg7 to offset: [%26], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_17 = memref.subview %alloc_12[0] [%20] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_18 = memref.subview %reinterpret_cast_16[0] [%20] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_18 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc_15, %arg13 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_19 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vsqrt ins(%alloc_19 : memref<8xf32, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_21 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc_21 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.vdiv ins(%alloc_21, %alloc_20 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_22 = memref.reinterpret_cast %arg8 to offset: [%26], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_23 = memref.subview %alloc_21[0] [%20] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_24 = memref.subview %reinterpret_cast_22[0] [%20] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_23 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_24 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %reinterpret_cast_25 = memref.reinterpret_cast %arg5 to offset: [%15], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_26 = memref.alloc() : memref<1x2048xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_26 {hivm.multi_buffer = 2 : i32} : memref<1x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %reinterpret_cast_25 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_28 = memref.collapse_shape %alloc_26 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_27 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_28 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    %reinterpret_cast_29 = memref.reinterpret_cast %arg6 to offset: [%15], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_30 = memref.alloc() : memref<1x2048xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_30 {hivm.multi_buffer = 2 : i32} : memref<1x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %reinterpret_cast_29 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_32 = memref.collapse_shape %alloc_30 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_31 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_32 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    %expand_shape = memref.expand_shape %alloc_12 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<8x2048xf32, #hivm.address_space<ub>>
    %alloc_34 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vsub ins(%alloc, %expand_shape : memref<8x2048xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%alloc_33 : memref<8x2048xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_34 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %expand_shape_35 = memref.expand_shape %alloc_21 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
    %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<8x2048xf32, #hivm.address_space<ub>>
    %alloc_37 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc_33, %expand_shape_35 : memref<8x2048xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%alloc_36 : memref<8x2048xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_37 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %alloc_38 = memref.alloc() {alignment = 64 : i64} : memref<8x2048xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc_36, %alloc_26 : memref<8x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%alloc_38 : memref<8x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<8x2048xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_39 {hivm.multi_buffer = 2 : i32} : memref<8x2048xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc_38, %alloc_30 : memref<8x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%alloc_39 : memref<8x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    %subview_40 = memref.subview %alloc_39[0, 0] [%22, 2048] [1, 1] : memref<8x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    %subview_41 = memref.subview %reinterpret_cast_2[0, 0] [%22, 2048] [1, 1] : memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_40 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) outs(%subview_41 : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  return
}

[NOTE] Ub overflow detected; automatically set enable-code-motion to false and retrying compilation.
// -----// IR Dump After PlanMemory Failed (hivm-plan-memory) //----- //
func.func @layer_norm_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: f32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 4.8828125E-4 : f32
  %c2048_i32 = arith.constant 2048 : i32
  %c8_i32 = arith.constant 8 : i32
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %4 = arith.divsi %arg17, %arg16 : i32
    %5 = arith.remsi %4, %arg15 : i32
    %6 = arith.muli %arg16, %arg15 : i32
    %7 = arith.divsi %arg17, %6 : i32
    %8 = arith.remsi %7, %arg14 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.muli %5, %c2048_i32 : i32
    %11 = arith.index_cast %9 : i32 to index
    %12 = arith.index_cast %arg9 : i32 to index
    %13 = arith.index_cast %10 : i32 to index
    %14 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%13, %11, %12]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [8, 2048], strides: [%12, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %15 = arith.index_cast %arg10 : i32 to index
    %16 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%13, %11, %15]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%16], sizes: [8, 2048], strides: [%15, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<8x2048xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<8x2048xf32, #hivm.address_space<ub>>
    %17 = arith.index_cast %arg12 : i32 to index
    %18 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%11, %17]
    %19 = affine.min affine_map<()[s0, s1] -> (s1 + 8, s0)>()[%18, %11]
    %20 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%19, %11]
    %21 = affine.min affine_map<()[s0, s1] -> (8, s0 - s1)>()[%19, %11]
    %22 = affine.max affine_map<()[s0] -> (0, s0)>()[%21]
    %23 = arith.cmpi slt, %22, %c8 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%22, 2048] [1, 1] : memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %alloc[0, 0] [%22, 2048] [1, 1] : memref<8x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    scf.if %23 {
      %collapse_shape_42 = memref.collapse_shape %alloc [[0, 1]] : memref<8x2048xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape_42 : memref<16384xf32, #hivm.address_space<ub>>)
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x1xf32, #hivm.address_space<ub>>
    %alloc_5 = memref.alloc() : memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vreduce <sum> ins(%alloc : memref<8x2048xf32, #hivm.address_space<ub>>) outs(%alloc_4 : memref<8x1xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_5 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape = memref.collapse_shape %alloc_4 [[0, 1]] : memref<8x1xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<8x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_7 = memref.collapse_shape %alloc [[0, 1]] : memref<8x2048xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
    %collapse_shape_8 = memref.collapse_shape %alloc_6 [[0, 1]] : memref<8x2048xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_7, %collapse_shape_7 : memref<16384xf32, #hivm.address_space<ub>>, memref<16384xf32, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<16384xf32, #hivm.address_space<ub>>)
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<8x1xf32, #hivm.address_space<ub>>
    %alloc_10 = memref.alloc() : memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vreduce <sum> ins(%alloc_6 : memref<8x2048xf32, #hivm.address_space<ub>>) outs(%alloc_9 : memref<8x1xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_10 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_11 = memref.collapse_shape %alloc_9 [[0, 1]] : memref<8x1xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_12 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape, %cst_0 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_12 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_11, %cst_0 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_13 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc_12, %alloc_12 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%alloc_14 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vsub ins(%alloc_13, %alloc_14 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%alloc_15 : memref<8xf32, #hivm.address_space<ub>>)
    %24 = arith.muli %5, %arg12 : i32
    %25 = arith.index_cast %24 : i32 to index
    %26 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%25, %11]
    %reinterpret_cast_16 = memref.reinterpret_cast %arg7 to offset: [%26], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_17 = memref.subview %alloc_12[0] [%20] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_18 = memref.subview %reinterpret_cast_16[0] [%20] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_17 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_18 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc_15, %arg13 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%alloc_19 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vsqrt ins(%alloc_19 : memref<8xf32, #hivm.address_space<ub>>) outs(%alloc_20 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<8xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_21 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc_21 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.vdiv ins(%alloc_21, %alloc_20 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%alloc_21 : memref<8xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_22 = memref.reinterpret_cast %arg8 to offset: [%26], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_23 = memref.subview %alloc_21[0] [%20] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_24 = memref.subview %reinterpret_cast_22[0] [%20] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_23 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_24 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %reinterpret_cast_25 = memref.reinterpret_cast %arg5 to offset: [%13], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_26 = memref.alloc() : memref<1x2048xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_26 {hivm.multi_buffer = 2 : i32} : memref<1x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %reinterpret_cast_25 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_28 = memref.collapse_shape %alloc_26 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_27 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_28 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    %reinterpret_cast_29 = memref.reinterpret_cast %arg6 to offset: [%13], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_30 = memref.alloc() : memref<1x2048xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_30 {hivm.multi_buffer = 2 : i32} : memref<1x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %reinterpret_cast_29 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_32 = memref.collapse_shape %alloc_30 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_31 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_32 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    %expand_shape = memref.expand_shape %alloc_12 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<8x2048xf32, #hivm.address_space<ub>>
    %alloc_34 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vsub ins(%alloc, %expand_shape : memref<8x2048xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%alloc_33 : memref<8x2048xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_34 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %expand_shape_35 = memref.expand_shape %alloc_21 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
    %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<8x2048xf32, #hivm.address_space<ub>>
    %alloc_37 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc_33, %expand_shape_35 : memref<8x2048xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%alloc_36 : memref<8x2048xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_37 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %alloc_38 = memref.alloc() {alignment = 64 : i64} : memref<8x2048xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc_36, %alloc_26 : memref<8x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%alloc_38 : memref<8x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<8x2048xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_39 {hivm.multi_buffer = 2 : i32} : memref<8x2048xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc_38, %alloc_30 : memref<8x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%alloc_39 : memref<8x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    %subview_40 = memref.subview %alloc_39[0, 0] [%22, 2048] [1, 1] : memref<8x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    %subview_41 = memref.subview %reinterpret_cast_2[0, 0] [%22, 2048] [1, 1] : memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_40 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) outs(%subview_41 : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  return
}

[NOTE] Ub overflow detected; automatically set enable-auto-multi-buffer to false and retrying compilation.
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @layer_norm_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: f32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c73792_i64 = arith.constant 73792 : i64
  %c65600_i64 = arith.constant 65600 : i64
  %c65568_i64 = arith.constant 65568 : i64
  %c147552_i64 = arith.constant 147552 : i64
  %c147520_i64 = arith.constant 147520 : i64
  %c81984_i64 = arith.constant 81984 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 4.8828125E-4 : f32
  %c2048_i32 = arith.constant 2048 : i32
  %c8_i32 = arith.constant 8 : i32
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %4 = arith.divsi %arg17, %arg16 : i32
    %5 = arith.remsi %4, %arg15 : i32
    %6 = arith.muli %arg16, %arg15 : i32
    %7 = arith.divsi %arg17, %6 : i32
    %8 = arith.remsi %7, %arg14 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.muli %5, %c2048_i32 : i32
    %11 = arith.index_cast %9 : i32 to index
    %12 = arith.index_cast %arg9 : i32 to index
    %13 = arith.index_cast %10 : i32 to index
    %14 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%13, %11, %12]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [8, 2048], strides: [%12, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %15 = arith.index_cast %arg10 : i32 to index
    %16 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%13, %11, %15]
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%16], sizes: [8, 2048], strides: [%15, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %17 = hivm.hir.pointer_cast(%c0_i64) : memref<8x2048xf32, #hivm.address_space<ub>>
    %18 = arith.index_cast %arg12 : i32 to index
    %19 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%11, %18]
    %20 = affine.min affine_map<()[s0, s1] -> (s1 + 8, s0)>()[%19, %11]
    %21 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%20, %11]
    %22 = affine.min affine_map<()[s0, s1] -> (8, s0 - s1)>()[%20, %11]
    %23 = affine.max affine_map<()[s0] -> (0, s0)>()[%22]
    %24 = arith.cmpi slt, %23, %c8 : index
    %subview = memref.subview %reinterpret_cast[0, 0] [%23, 2048] [1, 1] : memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %17[0, 0] [%23, 2048] [1, 1] : memref<8x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    scf.if %24 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %collapse_shape_22 = memref.collapse_shape %17 [[0, 1]] : memref<8x2048xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape_22 : memref<16384xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%subview : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %25 = hivm.hir.pointer_cast(%c65536_i64) : memref<8x1xf32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c81984_i64) : memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vreduce <sum> ins(%17 : memref<8x2048xf32, #hivm.address_space<ub>>) outs(%25 : memref<8x1xf32, #hivm.address_space<ub>>) temp_buffer(%26 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape = memref.collapse_shape %25 [[0, 1]] : memref<8x1xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c81984_i64) : memref<8x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_4 = memref.collapse_shape %17 [[0, 1]] : memref<8x2048xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
    %collapse_shape_5 = memref.collapse_shape %27 [[0, 1]] : memref<8x2048xf32, #hivm.address_space<ub>> into memref<16384xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_4, %collapse_shape_4 : memref<16384xf32, #hivm.address_space<ub>>, memref<16384xf32, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<16384xf32, #hivm.address_space<ub>>)
    %28 = hivm.hir.pointer_cast(%c147520_i64) : memref<8x1xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c147552_i64) : memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%27 : memref<8x2048xf32, #hivm.address_space<ub>>) outs(%28 : memref<8x1xf32, #hivm.address_space<ub>>) temp_buffer(%29 : memref<8192xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_6 = memref.collapse_shape %28 [[0, 1]] : memref<8x1xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c65536_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape, %cst_0 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%30 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %31 = hivm.hir.pointer_cast(%c147520_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_6, %cst_0 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%31 : memref<8xf32, #hivm.address_space<ub>>)
    %32 = hivm.hir.pointer_cast(%c81984_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%30, %30 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%32 : memref<8xf32, #hivm.address_space<ub>>)
    %33 = hivm.hir.pointer_cast(%c147520_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%31, %32 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%33 : memref<8xf32, #hivm.address_space<ub>>)
    %34 = arith.muli %5, %arg12 : i32
    %35 = arith.index_cast %34 : i32 to index
    %36 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%35, %11]
    %reinterpret_cast_7 = memref.reinterpret_cast %arg7 to offset: [%36], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_8 = memref.subview %30[0] [%21] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_9 = memref.subview %reinterpret_cast_7[0] [%21] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %37 = hivm.hir.pointer_cast(%c147520_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%33, %arg13 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%37 : memref<8xf32, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c147520_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsqrt ins(%37 : memref<8xf32, #hivm.address_space<ub>>) outs(%38 : memref<8xf32, #hivm.address_space<ub>>)
    %39 = hivm.hir.pointer_cast(%c65568_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%39 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%39, %38 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%39 : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_10 = memref.reinterpret_cast %arg8 to offset: [%36], sizes: [8], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_11 = memref.subview %39[0] [%21] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_12 = memref.subview %reinterpret_cast_10[0] [%21] [1] : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_11 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_12 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %reinterpret_cast_13 = memref.reinterpret_cast %arg5 to offset: [%13], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %40 = hivm.hir.pointer_cast(%c65600_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_14 = memref.collapse_shape %reinterpret_cast_13 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_15 = memref.collapse_shape %40 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_14 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_15 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %reinterpret_cast_16 = memref.reinterpret_cast %arg6 to offset: [%13], sizes: [1, 2048], strides: [2048, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>>
    %41 = hivm.hir.pointer_cast(%c73792_i64) : memref<1x2048xf32, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %reinterpret_cast_16 [[0, 1]] : memref<1x2048xf32, strided<[2048, 1], offset: ?>, #hivm.address_space<gm>> into memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_18 = memref.collapse_shape %41 [[0, 1]] : memref<1x2048xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_17 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_18 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %expand_shape = memref.expand_shape %30 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
    %42 = hivm.hir.pointer_cast(%c0_i64) : memref<8x2048xf32, #hivm.address_space<ub>>
    %43 = hivm.hir.pointer_cast(%c81984_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vsub ins(%17, %expand_shape : memref<8x2048xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%42 : memref<8x2048xf32, #hivm.address_space<ub>>) temp_buffer(%43 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %expand_shape_19 = memref.expand_shape %39 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
    %44 = hivm.hir.pointer_cast(%c0_i64) : memref<8x2048xf32, #hivm.address_space<ub>>
    %45 = hivm.hir.pointer_cast(%c81984_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%42, %expand_shape_19 : memref<8x2048xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%44 : memref<8x2048xf32, #hivm.address_space<ub>>) temp_buffer(%45 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %46 = hivm.hir.pointer_cast(%c0_i64) : memref<8x2048xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%44, %40 : memref<8x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%46 : memref<8x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    %47 = hivm.hir.pointer_cast(%c0_i64) : memref<8x2048xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%46, %41 : memref<8x2048xf32, #hivm.address_space<ub>>, memref<1x2048xf32, #hivm.address_space<ub>>) outs(%47 : memref<8x2048xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_20 = memref.subview %47[0, 0] [%23, 2048] [1, 1] : memref<8x2048xf32, #hivm.address_space<ub>> to memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>
    %subview_21 = memref.subview %reinterpret_cast_2[0, 0] [%23, 2048] [1, 1] : memref<8x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_20 : memref<?x2048xf32, strided<[2048, 1]>, #hivm.address_space<ub>>) outs(%subview_21 : memref<?x2048xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

[NOTE] Due to Ub overflow, automatically set enable-code-motion to false, set enable-auto-multi-buffer to false; compilation then succeeded.
bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
