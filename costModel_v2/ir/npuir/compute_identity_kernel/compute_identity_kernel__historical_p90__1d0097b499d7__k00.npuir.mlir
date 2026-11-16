// -----// IR Dump After PlanMemory Failed (hivm-plan-memory) //----- //
func.func @compute_identity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c512_i32 = arith.constant 512 : i32
  %c256_i32 = arith.constant 256 : i32
  %c8_i32 = arith.constant 8 : i32
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<32xf32, #hivm.address_space<ub>>)
  %5 = arith.index_cast %arg9 : i32 to index
  %6 = arith.remsi %arg8, %c512_i32 : i32
  %7 = arith.cmpi eq, %6, %c0_i32 : i32
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %8 = arith.divsi %arg13, %4 : i32
    %9 = arith.remsi %8, %arg10 : i32
    %10 = arith.muli %9, %c32_i32 : i32
    %11 = arith.subi %arg7, %10 : i32
    %12 = arith.muli %10, %arg9 : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.index_cast %11 : i32 to index
    %15 = affine.max affine_map<()[s0] -> (0, s0)>()[%14]
    %16 = affine.min affine_map<()[s0] -> (32, s0)>()[%15]
    %17 = affine.max affine_map<()[s0] -> (0, s0)>()[%16]
    %18 = arith.cmpi slt, %17, %c32 : index
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%alloc : memref<32xf32, #hivm.address_space<ub>>) outs(%alloc_0 : memref<32xf32, #hivm.address_space<ub>>)
    %19 = scf.for %arg14 = %c0_i32 to %arg3 step %c8_i32 iter_args(%arg15 = %alloc_0) -> (memref<32xf32, #hivm.address_space<ub>>)  : i32 {
      %20 = arith.subi %arg3, %arg14 : i32
      %21 = arith.index_cast %arg14 : i32 to index
      %22 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%13, %21]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%22], sizes: [32, 8], strides: [%5, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_1 = memref.alloc() : memref<32x8xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_1 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
      %23 = arith.index_cast %20 : i32 to index
      %24 = affine.max affine_map<()[s0] -> (0, s0)>()[%23]
      %25 = affine.min affine_map<()[s0] -> (8, s0)>()[%24]
      %26 = affine.max affine_map<()[s0] -> (0, s0)>()[%25]
      %27 = arith.cmpi slt, %26, %c8 : index
      %28 = arith.ori %18, %27 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%17, %26] [1, 1] : memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %alloc_1[0, 0] [%17, %26] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.if %28 {
        %collapse_shape_8 = memref.collapse_shape %alloc_1 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_8 : memref<256xf32, #hivm.address_space<ub>>)
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x1xf32, #hivm.address_space<ub>>
      hivm.hir.vreduce <sum> ins(%alloc_1 : memref<32x8xf32, #hivm.address_space<ub>>) outs(%alloc_3 : memref<32x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
      %collapse_shape = memref.collapse_shape %alloc_3 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      %subview_4 = memref.subview %collapse_shape[0] [%16] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%alloc : memref<32xf32, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32xf32, #hivm.address_space<ub>>)
      %subview_6 = memref.subview %alloc_5[0] [%16] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%arg15, %alloc_5 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32xf32, #hivm.address_space<ub>>)
      scf.yield %alloc_7 : memref<32xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    scf.if %7 {
      %20 = arith.muli %10, %arg8 : i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %arg8 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%21], sizes: [32, 256], strides: [%22, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_1 = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_1 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %23 = affine.max affine_map<()[s0] -> (0, s0)>()[%22]
      %24 = affine.min affine_map<()[s0] -> (256, s0)>()[%23]
      %25 = affine.max affine_map<()[s0] -> (0, s0)>()[%24]
      %26 = arith.cmpi slt, %25, %c256 : index
      %27 = arith.ori %18, %26 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%17, %25] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %alloc_1[0, 0] [%17, %25] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      scf.if %27 {
        %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %expand_shape = memref.expand_shape %19 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
      %28 = scf.for %arg14 = %c0_i32 to %arg8 step %c512_i32 iter_args(%arg15 = %alloc_1) -> (memref<32x256xf32, #hivm.address_space<ub>>)  : i32 {
        %29 = arith.addi %arg14, %c256_i32 : i32
        %30 = arith.cmpi slt, %29, %arg8 : i32
        %31 = scf.if %30 -> (memref<32x256xf32, #hivm.address_space<ub>>) {
          %41 = arith.subi %arg8, %29 : i32
          %42 = arith.index_cast %29 : i32 to index
          %43 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%21, %42]
          %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [32, 256], strides: [%22, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_13 = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_13 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
          %44 = arith.index_cast %41 : i32 to index
          %45 = affine.max affine_map<()[s0] -> (0, s0)>()[%44]
          %46 = affine.min affine_map<()[s0] -> (256, s0)>()[%45]
          %47 = affine.max affine_map<()[s0] -> (0, s0)>()[%46]
          %48 = arith.cmpi slt, %47, %c256 : index
          %49 = arith.ori %18, %48 : i1
          %subview_14 = memref.subview %reinterpret_cast_12[0, 0] [%17, %47] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_15 = memref.subview %alloc_13[0, 0] [%17, %47] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          scf.if %49 {
            %collapse_shape = memref.collapse_shape %alloc_13 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
          } {hivm.unlikely_condition}
          hivm.hir.load ins(%subview_14 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_15 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
          scf.yield %alloc_13 : memref<32x256xf32, #hivm.address_space<ub>>
        } else {
          scf.yield %arg15 : memref<32x256xf32, #hivm.address_space<ub>>
        }
        %32 = arith.subi %arg8, %arg14 : i32
        %33 = arith.index_cast %arg14 : i32 to index
        %34 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%21, %33]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%34], sizes: [32, 256], strides: [%22, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_4 = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_4 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
        %35 = arith.index_cast %32 : i32 to index
        %36 = affine.max affine_map<()[s0] -> (0, s0)>()[%35]
        %37 = affine.min affine_map<()[s0] -> (256, s0)>()[%36]
        %38 = affine.max affine_map<()[s0] -> (0, s0)>()[%37]
        %39 = arith.cmpi slt, %38, %c256 : index
        %40 = arith.ori %18, %39 : i1
        %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%17, %38] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_6 = memref.subview %alloc_4[0, 0] [%17, %38] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        scf.if %40 {
          %collapse_shape = memref.collapse_shape %alloc_4 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_5 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32x256xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_7 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
        %alloc_8 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.vmul ins(%alloc_4, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_8 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
        %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%34], sizes: [32, 256], strides: [%22, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_10 = memref.subview %alloc_7[0, 0] [%17, %38] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_11 = memref.subview %reinterpret_cast_9[0, 0] [%17, %38] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.store ins(%subview_10 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_11 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
        scf.if %30 {
          %41 = arith.subi %arg8, %29 : i32
          %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<32x256xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_12 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
          %alloc_13 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%31, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%alloc_12 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_13 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
          %42 = arith.index_cast %29 : i32 to index
          %43 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%21, %42]
          %reinterpret_cast_14 = memref.reinterpret_cast %arg6 to offset: [%43], sizes: [32, 256], strides: [%22, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %44 = arith.index_cast %41 : i32 to index
          %45 = affine.max affine_map<()[s0] -> (0, s0)>()[%44]
          %46 = affine.min affine_map<()[s0] -> (256, s0)>()[%45]
          %47 = affine.max affine_map<()[s0] -> (0, s0)>()[%46]
          %subview_15 = memref.subview %alloc_12[0, 0] [%17, %47] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%17, %47] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.store ins(%subview_15 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_16 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
        } {pipeline.veconly}
        hivm.hir.copy ins(%31 : memref<32x256xf32, #hivm.address_space<ub>>) outs(%arg15 : memref<32x256xf32, #hivm.address_space<ub>>)
        scf.yield %arg15 : memref<32x256xf32, #hivm.address_space<ub>>
      }
    } else {
      %20 = arith.muli %10, %arg8 : i32
      %expand_shape = memref.expand_shape %19 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.index_cast %arg8 : i32 to index
      scf.for %arg14 = %c0_i32 to %arg8 step %c256_i32  : i32 {
        %23 = arith.subi %arg8, %arg14 : i32
        %24 = arith.index_cast %arg14 : i32 to index
        %25 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%21, %24]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%25], sizes: [32, 256], strides: [%22, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_1 = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_1 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
        %26 = arith.index_cast %23 : i32 to index
        %27 = affine.max affine_map<()[s0] -> (0, s0)>()[%26]
        %28 = affine.min affine_map<()[s0] -> (256, s0)>()[%27]
        %29 = affine.max affine_map<()[s0] -> (0, s0)>()[%28]
        %30 = arith.cmpi slt, %29, %c256 : index
        %31 = arith.ori %18, %30 : i1
        %subview = memref.subview %reinterpret_cast[0, 0] [%17, %29] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %alloc_1[0, 0] [%17, %29] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        scf.if %31 {
          %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x256xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_3 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
        %alloc_4 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.vmul ins(%alloc_1, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%alloc_3 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_4 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%25], sizes: [32, 256], strides: [%22, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_6 = memref.subview %alloc_3[0, 0] [%17, %29] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%17, %29] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.store ins(%subview_6 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      }
    } {pipeline.veconly}
  } {autoblockify.subloop}
  return
}

[NOTE] Ub overflow detected; automatically set enable-code-motion to false and retrying compilation.
// -----// IR Dump After PlanMemory Failed (hivm-plan-memory) //----- //
func.func @compute_identity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c512_i32 = arith.constant 512 : i32
  %c256_i32 = arith.constant 256 : i32
  %c8_i32 = arith.constant 8 : i32
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %4 = arith.muli %arg12, %arg11 : i32
    %5 = arith.divsi %arg13, %4 : i32
    %6 = arith.remsi %5, %arg10 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<32xf32, #hivm.address_space<ub>>)
    %7 = arith.muli %6, %c32_i32 : i32
    %8 = arith.subi %arg7, %7 : i32
    %9 = arith.muli %7, %arg9 : i32
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%alloc : memref<32xf32, #hivm.address_space<ub>>) outs(%alloc_0 : memref<32xf32, #hivm.address_space<ub>>)
    %10 = scf.for %arg14 = %c0_i32 to %arg3 step %c8_i32 iter_args(%arg15 = %alloc_0) -> (memref<32xf32, #hivm.address_space<ub>>)  : i32 {
      %13 = arith.subi %arg3, %arg14 : i32
      %14 = arith.index_cast %9 : i32 to index
      %15 = arith.index_cast %arg14 : i32 to index
      %16 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%14, %15]
      %17 = arith.index_cast %arg9 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%16], sizes: [32, 8], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_1 = memref.alloc() : memref<32x8xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_1 {hivm.multi_buffer = 2 : i32} : memref<32x8xf32, #hivm.address_space<ub>>
      %18 = arith.index_cast %8 : i32 to index
      %19 = affine.max affine_map<()[s0] -> (0, s0)>()[%18]
      %20 = affine.min affine_map<()[s0] -> (32, s0)>()[%19]
      %21 = arith.index_cast %13 : i32 to index
      %22 = affine.max affine_map<()[s0] -> (0, s0)>()[%21]
      %23 = affine.max affine_map<()[s0] -> (0, s0)>()[%20]
      %24 = affine.min affine_map<()[s0] -> (8, s0)>()[%22]
      %25 = affine.max affine_map<()[s0] -> (0, s0)>()[%24]
      %26 = arith.cmpi slt, %23, %c32 : index
      %27 = arith.cmpi slt, %25, %c8 : index
      %28 = arith.ori %26, %27 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%23, %25] [1, 1] : memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %alloc_1[0, 0] [%23, %25] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.if %28 {
        %collapse_shape_8 = memref.collapse_shape %alloc_1 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_8 : memref<256xf32, #hivm.address_space<ub>>)
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x1xf32, #hivm.address_space<ub>>
      hivm.hir.vreduce <sum> ins(%alloc_1 : memref<32x8xf32, #hivm.address_space<ub>>) outs(%alloc_3 : memref<32x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
      %collapse_shape = memref.collapse_shape %alloc_3 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      %subview_4 = memref.subview %collapse_shape[0] [%20] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%alloc : memref<32xf32, #hivm.address_space<ub>>) outs(%alloc_5 : memref<32xf32, #hivm.address_space<ub>>)
      %subview_6 = memref.subview %alloc_5[0] [%20] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%arg15, %alloc_5 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32xf32, #hivm.address_space<ub>>)
      scf.yield %alloc_7 : memref<32xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    %11 = arith.remsi %arg8, %c512_i32 : i32
    %12 = arith.cmpi eq, %11, %c0_i32 : i32
    scf.if %12 {
      %13 = arith.muli %7, %arg8 : i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = arith.index_cast %arg8 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [32, 256], strides: [%15, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %alloc_1 = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_1 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
      %16 = arith.index_cast %8 : i32 to index
      %17 = affine.max affine_map<()[s0] -> (0, s0)>()[%16]
      %18 = affine.max affine_map<()[s0] -> (0, s0)>()[%15]
      %19 = affine.min affine_map<()[s0] -> (32, s0)>()[%17]
      %20 = affine.max affine_map<()[s0] -> (0, s0)>()[%19]
      %21 = affine.min affine_map<()[s0] -> (256, s0)>()[%18]
      %22 = affine.max affine_map<()[s0] -> (0, s0)>()[%21]
      %23 = arith.cmpi slt, %20, %c32 : index
      %24 = arith.cmpi slt, %22, %c256 : index
      %25 = arith.ori %23, %24 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%20, %22] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %alloc_1[0, 0] [%20, %22] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      scf.if %25 {
        %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %expand_shape = memref.expand_shape %10 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
      %26 = scf.for %arg14 = %c0_i32 to %arg8 step %c512_i32 iter_args(%arg15 = %alloc_1) -> (memref<32x256xf32, #hivm.address_space<ub>>)  : i32 {
        %27 = arith.addi %arg14, %c256_i32 : i32
        %28 = arith.cmpi slt, %27, %arg8 : i32
        %29 = scf.if %28 -> (memref<32x256xf32, #hivm.address_space<ub>>) {
          %39 = arith.subi %arg8, %27 : i32
          %40 = arith.index_cast %27 : i32 to index
          %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%14, %40]
          %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%41], sizes: [32, 256], strides: [%15, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_13 = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_13 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
          %42 = arith.index_cast %39 : i32 to index
          %43 = affine.max affine_map<()[s0] -> (0, s0)>()[%42]
          %44 = affine.min affine_map<()[s0] -> (256, s0)>()[%43]
          %45 = affine.max affine_map<()[s0] -> (0, s0)>()[%44]
          %46 = arith.cmpi slt, %45, %c256 : index
          %47 = arith.ori %23, %46 : i1
          %subview_14 = memref.subview %reinterpret_cast_12[0, 0] [%20, %45] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_15 = memref.subview %alloc_13[0, 0] [%20, %45] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          scf.if %47 {
            %collapse_shape = memref.collapse_shape %alloc_13 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
          } {hivm.unlikely_condition}
          hivm.hir.load ins(%subview_14 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_15 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
          scf.yield %alloc_13 : memref<32x256xf32, #hivm.address_space<ub>>
        } else {
          scf.yield %arg15 : memref<32x256xf32, #hivm.address_space<ub>>
        }
        %30 = arith.subi %arg8, %arg14 : i32
        %31 = arith.index_cast %arg14 : i32 to index
        %32 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%14, %31]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%32], sizes: [32, 256], strides: [%15, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_4 = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_4 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
        %33 = arith.index_cast %30 : i32 to index
        %34 = affine.max affine_map<()[s0] -> (0, s0)>()[%33]
        %35 = affine.min affine_map<()[s0] -> (256, s0)>()[%34]
        %36 = affine.max affine_map<()[s0] -> (0, s0)>()[%35]
        %37 = arith.cmpi slt, %36, %c256 : index
        %38 = arith.ori %23, %37 : i1
        %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%20, %36] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_6 = memref.subview %alloc_4[0, 0] [%20, %36] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        scf.if %38 {
          %collapse_shape = memref.collapse_shape %alloc_4 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_5 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_6 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32x256xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_7 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
        %alloc_8 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.vmul ins(%alloc_4, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%alloc_7 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_8 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
        %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%32], sizes: [32, 256], strides: [%15, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_10 = memref.subview %alloc_7[0, 0] [%20, %36] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_11 = memref.subview %reinterpret_cast_9[0, 0] [%20, %36] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.store ins(%subview_10 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_11 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
        scf.if %28 {
          %39 = arith.subi %arg8, %27 : i32
          %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<32x256xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_12 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
          %alloc_13 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
          hivm.hir.vmul ins(%29, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%alloc_12 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_13 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
          %40 = arith.index_cast %27 : i32 to index
          %41 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%14, %40]
          %reinterpret_cast_14 = memref.reinterpret_cast %arg6 to offset: [%41], sizes: [32, 256], strides: [%15, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %42 = arith.index_cast %39 : i32 to index
          %43 = affine.max affine_map<()[s0] -> (0, s0)>()[%42]
          %44 = affine.min affine_map<()[s0] -> (256, s0)>()[%43]
          %45 = affine.max affine_map<()[s0] -> (0, s0)>()[%44]
          %subview_15 = memref.subview %alloc_12[0, 0] [%20, %45] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %subview_16 = memref.subview %reinterpret_cast_14[0, 0] [%20, %45] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.store ins(%subview_15 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_16 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
        } {pipeline.veconly}
        hivm.hir.copy ins(%29 : memref<32x256xf32, #hivm.address_space<ub>>) outs(%arg15 : memref<32x256xf32, #hivm.address_space<ub>>)
        scf.yield %arg15 : memref<32x256xf32, #hivm.address_space<ub>>
      }
    } else {
      %13 = arith.muli %7, %arg8 : i32
      %expand_shape = memref.expand_shape %10 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
      scf.for %arg14 = %c0_i32 to %arg8 step %c256_i32  : i32 {
        %14 = arith.subi %arg8, %arg14 : i32
        %15 = arith.index_cast %13 : i32 to index
        %16 = arith.index_cast %arg14 : i32 to index
        %17 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%15, %16]
        %18 = arith.index_cast %arg8 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%17], sizes: [32, 256], strides: [%18, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc_1 = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_1 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
        %19 = arith.index_cast %8 : i32 to index
        %20 = affine.max affine_map<()[s0] -> (0, s0)>()[%19]
        %21 = arith.index_cast %14 : i32 to index
        %22 = affine.max affine_map<()[s0] -> (0, s0)>()[%21]
        %23 = affine.min affine_map<()[s0] -> (32, s0)>()[%20]
        %24 = affine.max affine_map<()[s0] -> (0, s0)>()[%23]
        %25 = affine.min affine_map<()[s0] -> (256, s0)>()[%22]
        %26 = affine.max affine_map<()[s0] -> (0, s0)>()[%25]
        %27 = arith.cmpi slt, %24, %c32 : index
        %28 = arith.cmpi slt, %26, %c256 : index
        %29 = arith.ori %27, %28 : i1
        %subview = memref.subview %reinterpret_cast[0, 0] [%24, %26] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %alloc_1[0, 0] [%24, %26] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        scf.if %29 {
          %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x256xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_3 {hivm.multi_buffer = 2 : i32} : memref<32x256xf32, #hivm.address_space<ub>>
        %alloc_4 = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.vmul ins(%alloc_1, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%alloc_3 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_4 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%17], sizes: [32, 256], strides: [%18, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_6 = memref.subview %alloc_3[0, 0] [%24, %26] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_7 = memref.subview %reinterpret_cast_5[0, 0] [%24, %26] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.store ins(%subview_6 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
      }
    } {pipeline.veconly}
  } {autoblockify.subloop}
  return
}

[NOTE] Ub overflow detected; automatically set enable-auto-multi-buffer to false and retrying compilation.
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @compute_identity_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, true, true, false, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c167424_i64 = arith.constant 167424 : i64
  %c134656_i64 = arith.constant 134656 : i64
  %c133632_i64 = arith.constant 133632 : i64
  %c100864_i64 = arith.constant 100864 : i64
  %c99840_i64 = arith.constant 99840 : i64
  %c67072_i64 = arith.constant 67072 : i64
  %c34304_i64 = arith.constant 34304 : i64
  %c1536_i64 = arith.constant 1536 : i64
  %c1280_i64 = arith.constant 1280 : i64
  %c1152_i64 = arith.constant 1152 : i64
  %c128_i64 = arith.constant 128 : i64
  %c1408_i64 = arith.constant 1408 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c512_i32 = arith.constant 512 : i32
  %c256_i32 = arith.constant 256 : i32
  %c8_i32 = arith.constant 8 : i32
  %c8 = arith.constant 8 : index
  %c256 = arith.constant 256 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %4 = arith.muli %arg12, %arg11 : i32
    %5 = arith.divsi %arg13, %4 : i32
    %6 = arith.remsi %5, %arg10 : i32
    %7 = hivm.hir.pointer_cast(%c0_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%cst : f32) outs(%7 : memref<32xf32, #hivm.address_space<ub>>)
    %8 = arith.muli %6, %c32_i32 : i32
    %9 = arith.subi %arg7, %8 : i32
    %10 = arith.muli %8, %arg9 : i32
    %11 = hivm.hir.pointer_cast(%c1408_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%7 : memref<32xf32, #hivm.address_space<ub>>) outs(%11 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %12 = scf.for %arg14 = %c0_i32 to %arg3 step %c8_i32 iter_args(%arg15 = %11) -> (memref<32xf32, #hivm.address_space<ub>>)  : i32 {
      %15 = arith.subi %arg3, %arg14 : i32
      %16 = arith.index_cast %10 : i32 to index
      %17 = arith.index_cast %arg14 : i32 to index
      %18 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%16, %17]
      %19 = arith.index_cast %arg9 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%18], sizes: [32, 8], strides: [%19, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %20 = hivm.hir.pointer_cast(%c128_i64) : memref<32x8xf32, #hivm.address_space<ub>>
      %21 = arith.index_cast %9 : i32 to index
      %22 = affine.max affine_map<()[s0] -> (0, s0)>()[%21]
      %23 = affine.min affine_map<()[s0] -> (32, s0)>()[%22]
      %24 = arith.index_cast %15 : i32 to index
      %25 = affine.max affine_map<()[s0] -> (0, s0)>()[%24]
      %26 = affine.max affine_map<()[s0] -> (0, s0)>()[%23]
      %27 = affine.min affine_map<()[s0] -> (8, s0)>()[%25]
      %28 = affine.max affine_map<()[s0] -> (0, s0)>()[%27]
      %29 = arith.cmpi slt, %26, %c32 : index
      %30 = arith.cmpi slt, %28, %c8 : index
      %31 = arith.ori %29, %30 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%26, %28] [1, 1] : memref<32x8xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %20[0, 0] [%26, %28] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>
      scf.if %31 {
        %collapse_shape_3 = memref.collapse_shape %20 [[0, 1]] : memref<32x8xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_3 : memref<256xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xf32, strided<[8, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %32 = hivm.hir.pointer_cast(%c1152_i64) : memref<32x1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vreduce <sum> ins(%20 : memref<32x8xf32, #hivm.address_space<ub>>) outs(%32 : memref<32x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      %collapse_shape = memref.collapse_shape %32 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      %subview_1 = memref.subview %collapse_shape[0] [%23] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %33 = hivm.hir.pointer_cast(%c1280_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%7 : memref<32xf32, #hivm.address_space<ub>>) outs(%33 : memref<32xf32, #hivm.address_space<ub>>)
      %subview_2 = memref.subview %33[0] [%23] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%subview_1 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
      %34 = hivm.hir.pointer_cast(%c1408_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%arg15, %33 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%34 : memref<32xf32, #hivm.address_space<ub>>)
      scf.yield %34 : memref<32xf32, #hivm.address_space<ub>>
    } {pipeline.veconly}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    %13 = arith.remsi %arg8, %c512_i32 : i32
    %14 = arith.cmpi eq, %13, %c0_i32 : i32
    scf.if %14 {
      %15 = arith.muli %8, %arg8 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = arith.index_cast %arg8 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%16], sizes: [32, 256], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %18 = hivm.hir.pointer_cast(%c1536_i64) : memref<32x256xf32, #hivm.address_space<ub>>
      %19 = arith.index_cast %9 : i32 to index
      %20 = affine.max affine_map<()[s0] -> (0, s0)>()[%19]
      %21 = affine.max affine_map<()[s0] -> (0, s0)>()[%17]
      %22 = affine.min affine_map<()[s0] -> (32, s0)>()[%20]
      %23 = affine.max affine_map<()[s0] -> (0, s0)>()[%22]
      %24 = affine.min affine_map<()[s0] -> (256, s0)>()[%21]
      %25 = affine.max affine_map<()[s0] -> (0, s0)>()[%24]
      %26 = arith.cmpi slt, %23, %c32 : index
      %27 = arith.cmpi slt, %25, %c256 : index
      %28 = arith.ori %26, %27 : i1
      %subview = memref.subview %reinterpret_cast[0, 0] [%23, %25] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %18[0, 0] [%23, %25] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
      scf.if %28 {
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %collapse_shape = memref.collapse_shape %18 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
      hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %expand_shape = memref.expand_shape %12 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      %29 = scf.for %arg14 = %c0_i32 to %arg8 step %c512_i32 iter_args(%arg15 = %18) -> (memref<32x256xf32, #hivm.address_space<ub>>)  : i32 {
        %30 = arith.addi %arg14, %c256_i32 : i32
        %31 = arith.cmpi slt, %30, %arg8 : i32
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        %32 = scf.if %31 -> (memref<32x256xf32, #hivm.address_space<ub>>) {
          %45 = arith.subi %arg8, %30 : i32
          %46 = arith.index_cast %30 : i32 to index
          %47 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%16, %46]
          %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [32, 256], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %48 = hivm.hir.pointer_cast(%c34304_i64) : memref<32x256xf32, #hivm.address_space<ub>>
          %49 = arith.index_cast %45 : i32 to index
          %50 = affine.max affine_map<()[s0] -> (0, s0)>()[%49]
          %51 = affine.min affine_map<()[s0] -> (256, s0)>()[%50]
          %52 = affine.max affine_map<()[s0] -> (0, s0)>()[%51]
          %53 = arith.cmpi slt, %52, %c256 : index
          %54 = arith.ori %26, %53 : i1
          %subview_8 = memref.subview %reinterpret_cast_7[0, 0] [%23, %52] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_9 = memref.subview %48[0, 0] [%23, %52] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          scf.if %54 {
            %collapse_shape = memref.collapse_shape %48 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          hivm.hir.load ins(%subview_8 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_9 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
          scf.yield %48 : memref<32x256xf32, #hivm.address_space<ub>>
        } else {
          scf.yield %arg15 : memref<32x256xf32, #hivm.address_space<ub>>
        }
        %33 = arith.subi %arg8, %arg14 : i32
        %34 = arith.index_cast %arg14 : i32 to index
        %35 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%16, %34]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [32, 256], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %36 = hivm.hir.pointer_cast(%c67072_i64) : memref<32x256xf32, #hivm.address_space<ub>>
        %37 = arith.index_cast %33 : i32 to index
        %38 = affine.max affine_map<()[s0] -> (0, s0)>()[%37]
        %39 = affine.min affine_map<()[s0] -> (256, s0)>()[%38]
        %40 = affine.max affine_map<()[s0] -> (0, s0)>()[%39]
        %41 = arith.cmpi slt, %40, %c256 : index
        %42 = arith.ori %26, %41 : i1
        %subview_2 = memref.subview %reinterpret_cast_1[0, 0] [%23, %40] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_3 = memref.subview %36[0, 0] [%23, %40] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        scf.if %42 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          %collapse_shape = memref.collapse_shape %36 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.load ins(%subview_2 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %43 = hivm.hir.pointer_cast(%c67072_i64) : memref<32x256xf32, #hivm.address_space<ub>>
        %44 = hivm.hir.pointer_cast(%c99840_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%36, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%43 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%44 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%35], sizes: [32, 256], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %43[0, 0] [%23, %40] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_6 = memref.subview %reinterpret_cast_4[0, 0] [%23, %40] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_5 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        scf.if %31 {
          %45 = arith.subi %arg8, %30 : i32
          %46 = hivm.hir.pointer_cast(%c100864_i64) : memref<32x256xf32, #hivm.address_space<ub>>
          %47 = hivm.hir.pointer_cast(%c133632_i64) : memref<256xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vmul ins(%32, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%46 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%47 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %48 = arith.index_cast %30 : i32 to index
          %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%16, %48]
          %reinterpret_cast_7 = memref.reinterpret_cast %arg6 to offset: [%49], sizes: [32, 256], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %50 = arith.index_cast %45 : i32 to index
          %51 = affine.max affine_map<()[s0] -> (0, s0)>()[%50]
          %52 = affine.min affine_map<()[s0] -> (256, s0)>()[%51]
          %53 = affine.max affine_map<()[s0] -> (0, s0)>()[%52]
          %subview_8 = memref.subview %46[0, 0] [%23, %53] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
          %subview_9 = memref.subview %reinterpret_cast_7[0, 0] [%23, %53] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          hivm.hir.store ins(%subview_8 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        } {pipeline.veconly}
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.copy ins(%32 : memref<32x256xf32, #hivm.address_space<ub>>) outs(%arg15 : memref<32x256xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        scf.yield %arg15 : memref<32x256xf32, #hivm.address_space<ub>>
      }
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } else {
      %15 = arith.muli %8, %arg8 : i32
      %expand_shape = memref.expand_shape %12 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      scf.for %arg14 = %c0_i32 to %arg8 step %c256_i32  : i32 {
        %16 = arith.subi %arg8, %arg14 : i32
        %17 = arith.index_cast %15 : i32 to index
        %18 = arith.index_cast %arg14 : i32 to index
        %19 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%17, %18]
        %20 = arith.index_cast %arg8 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%19], sizes: [32, 256], strides: [%20, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %21 = hivm.hir.pointer_cast(%c134656_i64) : memref<32x256xf32, #hivm.address_space<ub>>
        %22 = arith.index_cast %9 : i32 to index
        %23 = affine.max affine_map<()[s0] -> (0, s0)>()[%22]
        %24 = arith.index_cast %16 : i32 to index
        %25 = affine.max affine_map<()[s0] -> (0, s0)>()[%24]
        %26 = affine.min affine_map<()[s0] -> (32, s0)>()[%23]
        %27 = affine.max affine_map<()[s0] -> (0, s0)>()[%26]
        %28 = affine.min affine_map<()[s0] -> (256, s0)>()[%25]
        %29 = affine.max affine_map<()[s0] -> (0, s0)>()[%28]
        %30 = arith.cmpi slt, %27, %c32 : index
        %31 = arith.cmpi slt, %29, %c256 : index
        %32 = arith.ori %30, %31 : i1
        %subview = memref.subview %reinterpret_cast[0, 0] [%27, %29] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_0 = memref.subview %21[0, 0] [%27, %29] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        scf.if %32 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %collapse_shape = memref.collapse_shape %21 [[0, 1]] : memref<32x256xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<8192xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.load ins(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %33 = hivm.hir.pointer_cast(%c134656_i64) : memref<32x256xf32, #hivm.address_space<ub>>
        %34 = hivm.hir.pointer_cast(%c167424_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%21, %expand_shape : memref<32x256xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%33 : memref<32x256xf32, #hivm.address_space<ub>>) temp_buffer(%34 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg6 to offset: [%19], sizes: [32, 256], strides: [%20, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %33[0, 0] [%27, %29] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>
        %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%27, %29] [1, 1] : memref<32x256xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%subview_2 : memref<?x?xf32, strided<[256, 1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      }
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

[NOTE] Due to Ub overflow, automatically set enable-code-motion to false, set enable-auto-multi-buffer to false; compilation then succeeded.
bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
