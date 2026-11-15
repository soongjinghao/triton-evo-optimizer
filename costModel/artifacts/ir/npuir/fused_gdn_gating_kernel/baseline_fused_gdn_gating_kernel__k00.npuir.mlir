// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @fused_gdn_gating_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c544_i64 = arith.constant 544 : i64
  %c832_i64 = arith.constant 832 : i64
  %c480_i64 = arith.constant 480 : i64
  %c416_i64 = arith.constant 416 : i64
  %c384_i64 = arith.constant 384 : i64
  %c768_i64 = arith.constant 768 : i64
  %c320_i64 = arith.constant 320 : i64
  %c704_i64 = arith.constant 704 : i64
  %c256_i64 = arith.constant 256 : i64
  %c640_i64 = arith.constant 640 : i64
  %c192_i64 = arith.constant 192 : i64
  %c576_i64 = arith.constant 576 : i64
  %c128_i64 = arith.constant 128 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c256_i32 = arith.constant 256 : i32
  %c16_i32 = arith.constant 16 : i32
  %c52_i32 = arith.constant 52 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+01 : f32
  %c0_i32 = arith.constant 0 : i32
  %c16 = arith.constant 16 : index
  %cst_2 = arith.constant -1.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg10, %arg11 : i32
  %1 = arith.muli %0, %arg12 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg12, %arg11 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%5 : memref<16xf32, #hivm.address_space<ub>>)
  scf.for %arg13 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %6 = arith.divsi %arg13, %4 : i32
    %7 = arith.remsi %6, %arg10 : i32
    %8 = arith.muli %7, %c52_i32 : i32
    %9 = arith.cmpi sge, %8, %arg9 : i32
    scf.if %9 {
    } else {
      %10 = arith.addi %8, %c52_i32 : i32
      %11 = arith.minsi %10, %arg9 : i32
      %12 = arith.subi %11, %8 : i32
      %13 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst_0 : f32) outs(%13 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      scf.for %arg14 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
        %14 = arith.index_cast %arg14 : i32 to index
        %15 = arith.index_cast %c0_i32 : i32 to index
        %16 = arith.index_cast %c16_i32 : i32 to index
        %17 = arith.index_cast %c1_i32 : i32 to index
        %18 = arith.index_cast %arg13 : i32 to index
        %19 = arith.index_cast %3 : i32 to index
        %20 = arith.index_cast %1 : i32 to index
        %21 = arith.index_cast %c48_i32 : i32 to index
        %22 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%14, %15, %16, %17, %18, %19, %21]
        %23 = arith.index_cast %22 : index to i1
        %c0_i64_3 = arith.constant 0 : i64
        %c1_i64 = arith.constant 1 : i64
        %24 = arith.select %23, %c0_i64_3, %c1_i64 : i64
        %25 = hivm.hir.pointer_cast(%c192_i64, %c640_i64) : memref<16xf32, #hivm.address_space<ub>>
        annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
        %26 = hivm.hir.pointer_cast(%c128_i64, %c576_i64) : memref<16xf32, #hivm.address_space<ub>>
        annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
        %27 = arith.muli %arg14, %c16_i32 : i32
        %28 = arith.index_cast %27 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%28], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %29 = affine.max affine_map<()[s0] -> (256, s0)>()[%28]
        %30 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%29, %28]
        %31 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%30, %28]
        %32 = arith.cmpi slt, %31, %c16 : index
        %subview = memref.subview %reinterpret_cast[0] [%31] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %26[0] [%31] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %32 {
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vbrc ins(%cst : f32) outs(%26 : memref<16xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %24]
        hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg8 to offset: [%28], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_6 = memref.subview %reinterpret_cast_5[0] [%31] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %25[0] [%31] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %32 {
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f32) outs(%25 : memref<16xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
        } {hivm.unlikely_condition}
        hivm.hir.pipe_barrier[<PIPE_MTE2>]
        hivm.hir.load ins(%subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
        hivm.hir.vexp ins(%26 : memref<16xf32, #hivm.address_space<ub>>) outs(%26 : memref<16xf32, #hivm.address_space<ub>>)
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%26, %cst_2 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%26 : memref<16xf32, #hivm.address_space<ub>>)
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%26, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%26 : memref<16xf32, #hivm.address_space<ub>>)
        scf.for %arg15 = %c0_i32 to %12 step %c1_i32  : i32 {
          %33 = arith.index_cast %arg15 : i32 to index
          %34 = arith.index_cast %c0_i32 : i32 to index
          %35 = arith.index_cast %12 : i32 to index
          %36 = arith.index_cast %c1_i32 : i32 to index
          %37 = arith.index_cast %arg14 : i32 to index
          %38 = arith.index_cast %c0_i32 : i32 to index
          %39 = arith.index_cast %c16_i32 : i32 to index
          %40 = arith.index_cast %c1_i32 : i32 to index
          %41 = arith.index_cast %arg13 : i32 to index
          %42 = arith.index_cast %3 : i32 to index
          %43 = arith.index_cast %1 : i32 to index
          %44 = arith.index_cast %c48_i32 : i32 to index
          %45 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s7) * ((-s1 + s2 + s3 - 1) floordiv s3) + ((s8 - s9) floordiv s10) * (((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7))) mod 2)>()[%33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %44]
          %46 = arith.index_cast %45 : index to i1
          %c0_i64_8 = arith.constant 0 : i64
          %c1_i64_9 = arith.constant 1 : i64
          %47 = arith.select %46, %c0_i64_8, %c1_i64_9 : i64
          %48 = arith.index_cast %45 : index to i1
          %c3_i64 = arith.constant 3 : i64
          %c4_i64 = arith.constant 4 : i64
          %49 = arith.select %48, %c3_i64, %c4_i64 : i64
          %50 = arith.index_cast %45 : index to i1
          %c1_i64_10 = arith.constant 1 : i64
          %c2_i64 = arith.constant 2 : i64
          %51 = arith.select %50, %c1_i64_10, %c2_i64 : i64
          %52 = hivm.hir.pointer_cast(%c320_i64, %c768_i64) : memref<16xf32, #hivm.address_space<ub>>
          annotation.mark %52 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
          %53 = hivm.hir.pointer_cast(%c480_i64, %c832_i64) : memref<16xf32, #hivm.address_space<ub>>
          annotation.mark %53 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
          %54 = hivm.hir.pointer_cast(%c320_i64, %c768_i64) : memref<16xf32, #hivm.address_space<ub>>
          annotation.mark %54 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
          %55 = hivm.hir.pointer_cast(%c256_i64, %c704_i64) : memref<16xf32, #hivm.address_space<ub>>
          annotation.mark %55 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
          %56 = arith.addi %8, %arg15 : i32
          %57 = arith.muli %56, %c256_i32 : i32
          %58 = arith.addi %57, %27 : i32
          %59 = arith.index_cast %58 : i32 to index
          %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%59], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_12 = memref.subview %reinterpret_cast_11[0] [%31] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_13 = memref.subview %55[0] [%31] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %32 {
            hivm.hir.vbrc ins(%cst : f32) outs(%55 : memref<16xf32, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %49]
          hivm.hir.load ins(%subview_12 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_13 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
          %reinterpret_cast_14 = memref.reinterpret_cast %arg7 to offset: [%59], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_15 = memref.subview %reinterpret_cast_14[0] [%31] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_16 = memref.subview %54[0] [%31] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %32 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.vbrc ins(%cst : f32) outs(%54 : memref<16xf32, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
          } {hivm.unlikely_condition}
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %47]
          hivm.hir.load ins(%subview_15 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_16 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
          hivm.hir.vadd ins(%55, %25 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%55 : memref<16xf32, #hivm.address_space<ub>>)
          %60 = hivm.hir.pointer_cast(%c384_i64) : memref<16xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vcmp ins(%55, %5 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%60 : memref<16xi1, #hivm.address_space<ub>>) compare_mode = <le>
          %61 = hivm.hir.pointer_cast(%c416_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.vexp ins(%55 : memref<16xf32, #hivm.address_space<ub>>) outs(%61 : memref<16xf32, #hivm.address_space<ub>>)
          %62 = hivm.hir.pointer_cast(%c416_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vadd ins(%61, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%62 : memref<16xf32, #hivm.address_space<ub>>)
          %63 = hivm.hir.pointer_cast(%c416_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vln ins(%62 : memref<16xf32, #hivm.address_space<ub>>) outs(%63 : memref<16xf32, #hivm.address_space<ub>>)
          %64 = hivm.hir.pointer_cast(%c544_i64) : memref<8xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %51]
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vsel ins(%60, %63, %55 : memref<16xi1, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%53 : memref<16xf32, #hivm.address_space<ub>>) temp_buffer(%64 : memref<8xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %49]
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vmul ins(%26, %53 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%53 : memref<16xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vmul ins(%54, %cst_2 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%54 : memref<16xf32, #hivm.address_space<ub>>)
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vadd ins(%54, %cst : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%54 : memref<16xf32, #hivm.address_space<ub>>)
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vexp ins(%54 : memref<16xf32, #hivm.address_space<ub>>) outs(%54 : memref<16xf32, #hivm.address_space<ub>>)
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vadd ins(%54, %cst_0 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%54 : memref<16xf32, #hivm.address_space<ub>>)
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.vdiv ins(%13, %54 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%52 : memref<16xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %reinterpret_cast_17 = memref.reinterpret_cast %arg3 to offset: [%59], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_18 = memref.subview %53[0] [%31] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_19 = memref.subview %reinterpret_cast_17[0] [%31] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          hivm.hir.store ins(%subview_18 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_19 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %51]
          %reinterpret_cast_20 = memref.reinterpret_cast %arg4 to offset: [%59], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_21 = memref.subview %52[0] [%31] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %subview_22 = memref.subview %reinterpret_cast_20[0] [%31] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.store ins(%subview_21 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_22 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %47]
        } {pipeline.veconly}
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %24]
      }
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
