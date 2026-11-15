// -----// IR Dump After PlanMemory Failed (hivm-plan-memory) //----- //
func.func @fused_gdn_gating_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+01 : f32
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_2 = arith.constant -1.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %alloc [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
  %5 = arith.index_cast %arg9 : i32 to index
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %alloc_3 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_4 : memref<8192xf32, #hivm.address_space<ub>>)
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %6 = arith.divsi %arg13, %arg12 : i32
    %7 = arith.remsi %6, %arg11 : i32
    %8 = arith.divsi %arg13, %4 : i32
    %9 = arith.remsi %8, %arg10 : i32
    %10 = arith.muli %9, %c128_i32 : i32
    %11 = arith.muli %7, %c64_i32 : i32
    %12 = arith.index_cast %10 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %alloc_5 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_5 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
    %13 = affine.max affine_map<()[s0] -> (256, s0)>()[%12]
    %14 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%13, %12]
    %15 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%14, %12]
    %16 = arith.cmpi slt, %15, %c128 : index
    %subview = memref.subview %reinterpret_cast[0] [%15] [1] : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %alloc_5[0] [%15] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %16 {
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<128xf32, #hivm.address_space<ub>>)
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %reinterpret_cast_7 = memref.reinterpret_cast %arg8 to offset: [%12], sizes: [1, 128], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_8 = memref.alloc() : memref<1x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_8 {hivm.multi_buffer = 2 : i32} : memref<1x128xf32, #hivm.address_space<ub>>
    %subview_9 = memref.subview %reinterpret_cast_7[0, 0] [1, %15] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_10 = memref.subview %alloc_8[0, 0] [1, %15] [1, 1] : memref<1x128xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %16 {
      %collapse_shape_57 = memref.collapse_shape %alloc_8 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_57 : memref<128xf32, #hivm.address_space<ub>>)
    } {hivm.unlikely_condition}
    %collapse_shape_11 = memref.collapse_shape %subview_9 [[0, 1]] : memref<1x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_12 = memref.collapse_shape %subview_10 [[0, 1]] : memref<1x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_12 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %17 = arith.index_cast %11 : i32 to index
    %18 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%12, %17]
    %reinterpret_cast_13 = memref.reinterpret_cast %arg6 to offset: [%18], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_14 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_14 {hivm.multi_buffer = 2 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    %19 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%17, %5]
    %20 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%19, %17]
    %21 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%20, %17]
    %22 = affine.max affine_map<()[s0] -> (0, s0)>()[%21]
    %23 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%14, %12]
    %24 = affine.max affine_map<()[s0] -> (0, s0)>()[%23]
    %25 = arith.cmpi slt, %22, %c64 : index
    %26 = arith.cmpi slt, %24, %c128 : index
    %27 = arith.ori %25, %26 : i1
    %subview_15 = memref.subview %reinterpret_cast_13[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_16 = memref.subview %alloc_14[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %27 {
      %collapse_shape_57 = memref.collapse_shape %alloc_14 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_57 : memref<8192xf32, #hivm.address_space<ub>>)
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_15 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_16 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %reinterpret_cast_17 = memref.reinterpret_cast %arg7 to offset: [%18], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_18 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_18 {hivm.multi_buffer = 2 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    %subview_19 = memref.subview %reinterpret_cast_17[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_20 = memref.subview %alloc_18[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %27 {
      %collapse_shape_57 = memref.collapse_shape %alloc_18 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_57 : memref<8192xf32, #hivm.address_space<ub>>)
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_19 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_20 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc_14, %alloc_8 : memref<64x128xf32, #hivm.address_space<ub>>, memref<1x128xf32, #hivm.address_space<ub>>) outs(%alloc_21 : memref<64x128xf32, #hivm.address_space<ub>>) broadcast = [0]
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %alloc_21 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %alloc_22 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%collapse_shape_23 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_24 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %alloc_25 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_24, %cst_0 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_26 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_28 = memref.collapse_shape %alloc_27 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vln ins(%collapse_shape_26 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_28 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<64x128xi1, #hivm.address_space<ub>>
    %collapse_shape_30 = memref.collapse_shape %alloc_29 [[0, 1]] : memref<64x128xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%collapse_shape_23, %collapse_shape : memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_30 : memref<8192xi1, #hivm.address_space<ub>>) compare_mode = <gt>
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_32 = memref.collapse_shape %alloc_31 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %alloc_33 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vsel ins(%collapse_shape_30, %collapse_shape_23, %collapse_shape_28 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_32 : memref<8192xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_33 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%alloc_5 : memref<128xf32, #hivm.address_space<ub>>) outs(%alloc_34 : memref<128xf32, #hivm.address_space<ub>>)
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32, #hivm.address_space<ub>>
    %collapse_shape_36 = memref.collapse_shape %alloc_35 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc_34, %cst_2 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_36 : memref<128xf32, #hivm.address_space<ub>>)
    %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32, #hivm.address_space<ub>>
    %collapse_shape_38 = memref.collapse_shape %alloc_37 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_36, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_38 : memref<128xf32, #hivm.address_space<ub>>)
    %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_39 {hivm.multi_buffer = 2 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc_37, %alloc_31 : memref<1x128xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>) outs(%alloc_39 : memref<64x128xf32, #hivm.address_space<ub>>) broadcast = [0]
    %reinterpret_cast_40 = memref.reinterpret_cast %arg3 to offset: [%18], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_41 = memref.subview %alloc_39[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_42 = memref.subview %reinterpret_cast_40[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_41 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_42 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
    %alloc_43 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_44 = memref.collapse_shape %alloc_18 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %collapse_shape_45 = memref.collapse_shape %alloc_43 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_44, %cst_2 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_45 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_46 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_47 = memref.collapse_shape %alloc_46 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_45, %cst : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_47 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_48 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_49 = memref.collapse_shape %alloc_48 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%collapse_shape_47 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_49 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_50 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_51 = memref.collapse_shape %alloc_50 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_49, %cst_0 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_51 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_52 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_52 {hivm.multi_buffer = 2 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_53 = memref.collapse_shape %alloc_52 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vdiv ins(%collapse_shape_4, %collapse_shape_51 : memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_53 : memref<8192xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_54 = memref.reinterpret_cast %arg4 to offset: [%18], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_55 = memref.subview %alloc_52[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_56 = memref.subview %reinterpret_cast_54[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_55 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_56 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  return
}

[NOTE] Ub overflow detected; automatically set enable-code-motion to false and retrying compilation.
// -----// IR Dump After PlanMemory Failed (hivm-plan-memory) //----- //
func.func @fused_gdn_gating_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+01 : f32
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_2 = arith.constant -1.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %4 = arith.divsi %arg13, %arg12 : i32
    %5 = arith.remsi %4, %arg11 : i32
    %6 = arith.muli %arg12, %arg11 : i32
    %7 = arith.divsi %arg13, %6 : i32
    %8 = arith.remsi %7, %arg10 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %alloc [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
    %9 = arith.muli %8, %c128_i32 : i32
    %10 = arith.muli %5, %c64_i32 : i32
    %11 = arith.index_cast %9 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%11], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %alloc_3 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_3 {hivm.multi_buffer = 2 : i32} : memref<128xf32, #hivm.address_space<ub>>
    %12 = affine.max affine_map<()[s0] -> (256, s0)>()[%11]
    %13 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%12, %11]
    %14 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%13, %11]
    %15 = arith.cmpi slt, %14, %c128 : index
    %subview = memref.subview %reinterpret_cast[0] [%14] [1] : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %alloc_3[0] [%14] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %15 {
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_3 : memref<128xf32, #hivm.address_space<ub>>)
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %reinterpret_cast_5 = memref.reinterpret_cast %arg8 to offset: [%11], sizes: [1, 128], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_6 = memref.alloc() : memref<1x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_6 {hivm.multi_buffer = 2 : i32} : memref<1x128xf32, #hivm.address_space<ub>>
    %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [1, %14] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_8 = memref.subview %alloc_6[0, 0] [1, %14] [1, 1] : memref<1x128xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %15 {
      %collapse_shape_57 = memref.collapse_shape %alloc_6 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_57 : memref<128xf32, #hivm.address_space<ub>>)
    } {hivm.unlikely_condition}
    %collapse_shape_9 = memref.collapse_shape %subview_7 [[0, 1]] : memref<1x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_10 = memref.collapse_shape %subview_8 [[0, 1]] : memref<1x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_9 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_10 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %16 = arith.index_cast %10 : i32 to index
    %17 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%11, %16]
    %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%17], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_12 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_12 {hivm.multi_buffer = 2 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    %18 = arith.index_cast %arg9 : i32 to index
    %19 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%16, %18]
    %20 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%19, %16]
    %21 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%20, %16]
    %22 = affine.max affine_map<()[s0] -> (0, s0)>()[%21]
    %23 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%13, %11]
    %24 = affine.max affine_map<()[s0] -> (0, s0)>()[%23]
    %25 = arith.cmpi slt, %22, %c64 : index
    %26 = arith.cmpi slt, %24, %c128 : index
    %27 = arith.ori %25, %26 : i1
    %subview_13 = memref.subview %reinterpret_cast_11[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_14 = memref.subview %alloc_12[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %27 {
      %collapse_shape_57 = memref.collapse_shape %alloc_12 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_57 : memref<8192xf32, #hivm.address_space<ub>>)
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_13 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_14 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %reinterpret_cast_15 = memref.reinterpret_cast %arg7 to offset: [%17], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %alloc_16 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_16 {hivm.multi_buffer = 2 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    %subview_17 = memref.subview %reinterpret_cast_15[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_18 = memref.subview %alloc_16[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %27 {
      %collapse_shape_57 = memref.collapse_shape %alloc_16 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_57 : memref<8192xf32, #hivm.address_space<ub>>)
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_17 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_18 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc_12, %alloc_6 : memref<64x128xf32, #hivm.address_space<ub>>, memref<1x128xf32, #hivm.address_space<ub>>) outs(%alloc_19 : memref<64x128xf32, #hivm.address_space<ub>>) broadcast = [0]
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %alloc_19 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %alloc_20 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%collapse_shape_21 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_22 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_24 = memref.collapse_shape %alloc_23 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_22, %cst_0 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_24 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %alloc_25 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vln ins(%collapse_shape_24 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<64x128xi1, #hivm.address_space<ub>>
    %collapse_shape_28 = memref.collapse_shape %alloc_27 [[0, 1]] : memref<64x128xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%collapse_shape_21, %collapse_shape : memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_28 : memref<8192xi1, #hivm.address_space<ub>>) compare_mode = <gt>
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_30 = memref.collapse_shape %alloc_29 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %alloc_31 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.vsel ins(%collapse_shape_28, %collapse_shape_21, %collapse_shape_26 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_30 : memref<8192xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_31 : memref<8xf32, #hivm.address_space<ub>>)
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%alloc_3 : memref<128xf32, #hivm.address_space<ub>>) outs(%alloc_32 : memref<128xf32, #hivm.address_space<ub>>)
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32, #hivm.address_space<ub>>
    %collapse_shape_34 = memref.collapse_shape %alloc_33 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc_32, %cst_2 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_34 : memref<128xf32, #hivm.address_space<ub>>)
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<1x128xf32, #hivm.address_space<ub>>
    %collapse_shape_36 = memref.collapse_shape %alloc_35 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_34, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_36 : memref<128xf32, #hivm.address_space<ub>>)
    %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_37 {hivm.multi_buffer = 2 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc_35, %alloc_29 : memref<1x128xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>) outs(%alloc_37 : memref<64x128xf32, #hivm.address_space<ub>>) broadcast = [0]
    %reinterpret_cast_38 = memref.reinterpret_cast %arg3 to offset: [%17], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_39 = memref.subview %alloc_37[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_40 = memref.subview %reinterpret_cast_38[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_39 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_40 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
    %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_42 = memref.collapse_shape %alloc_16 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %collapse_shape_43 = memref.collapse_shape %alloc_41 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_42, %cst_2 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_43 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_44 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_45 = memref.collapse_shape %alloc_44 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_43, %cst : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_45 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_46 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_47 = memref.collapse_shape %alloc_46 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%collapse_shape_45 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_47 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_48 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_49 = memref.collapse_shape %alloc_48 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%collapse_shape_47, %cst_0 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_49 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_50 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_51 = memref.collapse_shape %alloc_50 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_51 : memref<8192xf32, #hivm.address_space<ub>>)
    %alloc_52 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_52 {hivm.multi_buffer = 2 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_53 = memref.collapse_shape %alloc_52 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vdiv ins(%collapse_shape_51, %collapse_shape_49 : memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_53 : memref<8192xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_54 = memref.reinterpret_cast %arg4 to offset: [%17], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_55 = memref.subview %alloc_52[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_56 = memref.subview %reinterpret_cast_54[0, 0] [%22, %24] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%subview_55 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_56 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  return
}

[NOTE] Ub overflow detected; automatically set enable-auto-multi-buffer to false and retrying compilation.
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @fused_gdn_gating_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c164864_i64 = arith.constant 164864 : i64
  %c66560_i64 = arith.constant 66560 : i64
  %c99328_i64 = arith.constant 99328 : i64
  %c33792_i64 = arith.constant 33792 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c512_i64 = arith.constant 512 : i64
  %c0_i64 = arith.constant 0 : i64
  %c132096_i64 = arith.constant 132096 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+01 : f32
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_2 = arith.constant -1.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %4 = arith.divsi %arg13, %arg12 : i32
    %5 = arith.remsi %4, %arg11 : i32
    %6 = arith.muli %arg12, %arg11 : i32
    %7 = arith.divsi %arg13, %6 : i32
    %8 = arith.remsi %7, %arg10 : i32
    %9 = hivm.hir.pointer_cast(%c132096_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %9 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
    %10 = arith.muli %8, %c128_i32 : i32
    %11 = arith.muli %5, %c64_i32 : i32
    %12 = arith.index_cast %10 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%12], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %13 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf32, #hivm.address_space<ub>>
    %14 = affine.max affine_map<()[s0] -> (256, s0)>()[%12]
    %15 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%14, %12]
    %16 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%15, %12]
    %17 = arith.cmpi slt, %16, %c128 : index
    %subview = memref.subview %reinterpret_cast[0] [%16] [1] : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_3 = memref.subview %13[0] [%16] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %17 {
      hivm.hir.vbrc ins(%cst : f32) outs(%13 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %reinterpret_cast_4 = memref.reinterpret_cast %arg8 to offset: [%12], sizes: [1, 128], strides: [128, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %18 = hivm.hir.pointer_cast(%c512_i64) : memref<1x128xf32, #hivm.address_space<ub>>
    %subview_5 = memref.subview %reinterpret_cast_4[0, 0] [1, %16] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_6 = memref.subview %18[0, 0] [1, %16] [1, 1] : memref<1x128xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %17 {
      %collapse_shape_36 = memref.collapse_shape %18 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_36 : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    %collapse_shape_7 = memref.collapse_shape %subview_5 [[0, 1]] : memref<1x?xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_8 = memref.collapse_shape %subview_6 [[0, 1]] : memref<1x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %19 = arith.index_cast %11 : i32 to index
    %20 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 256)>()[%12, %19]
    %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%20], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %21 = hivm.hir.pointer_cast(%c1024_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %22 = arith.index_cast %arg9 : i32 to index
    %23 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%19, %22]
    %24 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%23, %19]
    %25 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%24, %19]
    %26 = affine.max affine_map<()[s0] -> (0, s0)>()[%25]
    %27 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%15, %12]
    %28 = affine.max affine_map<()[s0] -> (0, s0)>()[%27]
    %29 = arith.cmpi slt, %26, %c64 : index
    %30 = arith.cmpi slt, %28, %c128 : index
    %31 = arith.ori %29, %30 : i1
    %subview_10 = memref.subview %reinterpret_cast_9[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_11 = memref.subview %21[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %31 {
      %collapse_shape_36 = memref.collapse_shape %21 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_36 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_10 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_11 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    %reinterpret_cast_12 = memref.reinterpret_cast %arg7 to offset: [%20], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %32 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %subview_13 = memref.subview %reinterpret_cast_12[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_14 = memref.subview %32[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    scf.if %31 {
      %collapse_shape_36 = memref.collapse_shape %32 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_36 : memref<8192xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.load ins(%subview_13 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_14 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %33 = hivm.hir.pointer_cast(%c1024_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.vadd ins(%21, %18 : memref<64x128xf32, #hivm.address_space<ub>>, memref<1x128xf32, #hivm.address_space<ub>>) outs(%33 : memref<64x128xf32, #hivm.address_space<ub>>) broadcast = [0]
    %34 = hivm.hir.pointer_cast(%c99328_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_15 = memref.collapse_shape %33 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %34 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%collapse_shape_15 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_16 : memref<8192xf32, #hivm.address_space<ub>>)
    %35 = hivm.hir.pointer_cast(%c99328_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_17 = memref.collapse_shape %35 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_16, %cst_0 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_17 : memref<8192xf32, #hivm.address_space<ub>>)
    %36 = hivm.hir.pointer_cast(%c99328_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_18 = memref.collapse_shape %36 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vln ins(%collapse_shape_17 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_18 : memref<8192xf32, #hivm.address_space<ub>>)
    %37 = hivm.hir.pointer_cast(%c132096_i64) : memref<64x128xi1, #hivm.address_space<ub>>
    %collapse_shape_19 = memref.collapse_shape %37 [[0, 1]] : memref<64x128xi1, #hivm.address_space<ub>> into memref<8192xi1, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%collapse_shape_15, %collapse_shape : memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<8192xi1, #hivm.address_space<ub>>) compare_mode = <gt>
    %38 = hivm.hir.pointer_cast(%c66560_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %38 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c164864_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%collapse_shape_19, %collapse_shape_15, %collapse_shape_18 : memref<8192xi1, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<8192xf32, #hivm.address_space<ub>>) temp_buffer(%39 : memref<8xf32, #hivm.address_space<ub>>)
    %40 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vexp ins(%13 : memref<128xf32, #hivm.address_space<ub>>) outs(%40 : memref<128xf32, #hivm.address_space<ub>>)
    %41 = hivm.hir.pointer_cast(%c0_i64) : memref<1x128xf32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %41 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%40, %cst_2 : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_21 : memref<128xf32, #hivm.address_space<ub>>)
    %42 = hivm.hir.pointer_cast(%c0_i64) : memref<1x128xf32, #hivm.address_space<ub>>
    %collapse_shape_22 = memref.collapse_shape %42 [[0, 1]] : memref<1x128xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_21, %cst : memref<128xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_22 : memref<128xf32, #hivm.address_space<ub>>)
    %43 = hivm.hir.pointer_cast(%c66560_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%42, %38 : memref<1x128xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>) outs(%43 : memref<64x128xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    %reinterpret_cast_23 = memref.reinterpret_cast %arg3 to offset: [%20], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_24 = memref.subview %43[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_25 = memref.subview %reinterpret_cast_23[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_24 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_25 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
    %44 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_26 = memref.collapse_shape %32 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %44 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_26, %cst_2 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_27 : memref<8192xf32, #hivm.address_space<ub>>)
    %45 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_28 = memref.collapse_shape %45 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_27, %cst : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_28 : memref<8192xf32, #hivm.address_space<ub>>)
    %46 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_29 = memref.collapse_shape %46 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%collapse_shape_28 : memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_29 : memref<8192xf32, #hivm.address_space<ub>>)
    %47 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_30 = memref.collapse_shape %47 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_29, %cst_0 : memref<8192xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_30 : memref<8192xf32, #hivm.address_space<ub>>)
    %48 = hivm.hir.pointer_cast(%c99328_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %48 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_0 : f32) outs(%collapse_shape_31 : memref<8192xf32, #hivm.address_space<ub>>)
    %49 = hivm.hir.pointer_cast(%c99328_i64) : memref<64x128xf32, #hivm.address_space<ub>>
    %collapse_shape_32 = memref.collapse_shape %49 [[0, 1]] : memref<64x128xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%collapse_shape_31, %collapse_shape_30 : memref<8192xf32, #hivm.address_space<ub>>, memref<8192xf32, #hivm.address_space<ub>>) outs(%collapse_shape_32 : memref<8192xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %reinterpret_cast_33 = memref.reinterpret_cast %arg4 to offset: [%20], sizes: [64, 128], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_34 = memref.subview %49[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %subview_35 = memref.subview %reinterpret_cast_33[0, 0] [%26, %28] [1, 1] : memref<64x128xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_34 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>) outs(%subview_35 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

[NOTE] Due to Ub overflow, automatically set enable-code-motion to false, set enable-auto-multi-buffer to false; compilation then succeeded.
bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
