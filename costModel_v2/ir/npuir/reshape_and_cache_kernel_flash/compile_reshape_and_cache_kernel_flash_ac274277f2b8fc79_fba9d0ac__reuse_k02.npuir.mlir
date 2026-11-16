// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @reshape_and_cache_kernel_flash(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i64 {tt.divisibility = 16 : i32}, %arg11: i64 {tt.divisibility = 16 : i32}, %arg12: i64 {tt.divisibility = 16 : i32}, %arg13: i64 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c3840_i64 = arith.constant 3840 : i64
  %c1792_i64 = arith.constant 1792 : i64
  %c3584_i64 = arith.constant 3584 : i64
  %c1536_i64 = arith.constant 1536 : i64
  %c3328_i64 = arith.constant 3328 : i64
  %c1280_i64 = arith.constant 1280 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c2816_i64 = arith.constant 2816 : i64
  %c768_i64 = arith.constant 768 : i64
  %c2560_i64 = arith.constant 2560 : i64
  %c512_i64 = arith.constant 512 : i64
  %c2304_i64 = arith.constant 2304 : i64
  %c256_i64 = arith.constant 256 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c3_i32 = arith.constant 3 : i32
  %c2_i32 = arith.constant 2 : i32
  %c8_i64 = arith.constant 8 : i64
  %c128_i32 = arith.constant 128 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID7>]
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg17 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = arith.index_cast %9 : index to i1
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %13 = arith.select %12, %c2_i64, %c3_i64 : i64
    %14 = arith.index_cast %9 : index to i1
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %15 = arith.select %14, %c4_i64, %c5_i64 : i64
    %16 = arith.index_cast %9 : index to i1
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %17 = arith.select %16, %c6_i64, %c7_i64 : i64
    %18 = hivm.hir.pointer_cast(%c1792_i64, %c3840_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c1536_i64, %c3584_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c1280_i64, %c3328_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c1024_i64, %c3072_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c768_i64, %c2816_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c512_i64, %c2560_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c256_i64, %c2304_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c0_i64, %c2048_i64) : memref<128xf16, #hivm.address_space<ub>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %26 = arith.divsi %arg17, %arg16 : i32
    %27 = arith.remsi %26, %arg15 : i32
    %28 = arith.divsi %arg17, %4 : i32
    %29 = arith.remsi %28, %arg14 : i32
    %30 = arith.muli %29, %c4_i32 : i32
    %31 = arith.muli %27, %c128_i32 : i32
    %32 = arith.cmpi slt, %30, %c1_i32 : i32
    scf.if %32 {
      %39 = arith.index_cast %30 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%39], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %40 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %41 = arith.cmpi sge, %40, %c0_i64 : i64
      scf.if %41 {
        %42 = arith.divsi %40, %c8_i64 : i64
        %43 = arith.remsi %40, %c8_i64 : i64
        %44 = arith.extsi %30 : i32 to i64
        %45 = arith.muli %44, %arg10 : i64
        %46 = arith.muli %44, %arg11 : i64
        %47 = arith.muli %42, %arg12 : i64
        %48 = arith.muli %43, %arg13 : i64
        %49 = arith.addi %47, %48 : i64
        %50 = arith.index_cast %45 : i64 to index
        %51 = arith.index_cast %31 : i32 to index
        %52 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%50, %51]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %53 = affine.max affine_map<()[s0] -> (128, s0)>()[%51]
        %54 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%53, %51]
        %55 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%54, %51]
        %56 = arith.cmpi slt, %55, %c128 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %25[0] [%55] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %56 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%25 : memref<128xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %17]
        hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %57 = arith.index_cast %46 : i64 to index
        %58 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%57, %51]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%58], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %reinterpret_cast_3[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %24[0] [%55] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %56 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%24 : memref<128xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_4 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %59 = arith.index_cast %49 : i64 to index
        %60 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%59, %51]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %reinterpret_cast_6[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%60], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %17]
      }
    } {pipeline.veconly}
    %33 = arith.addi %30, %c1_i32 : i32
    %34 = arith.cmpi slt, %33, %c1_i32 : i32
    scf.if %34 {
      %39 = arith.index_cast %33 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%39], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %40 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %41 = arith.cmpi sge, %40, %c0_i64 : i64
      scf.if %41 {
        %42 = arith.divsi %40, %c8_i64 : i64
        %43 = arith.remsi %40, %c8_i64 : i64
        %44 = arith.extsi %33 : i32 to i64
        %45 = arith.muli %44, %arg10 : i64
        %46 = arith.muli %44, %arg11 : i64
        %47 = arith.muli %42, %arg12 : i64
        %48 = arith.muli %43, %arg13 : i64
        %49 = arith.addi %47, %48 : i64
        %50 = arith.index_cast %45 : i64 to index
        %51 = arith.index_cast %31 : i32 to index
        %52 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%50, %51]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %53 = affine.max affine_map<()[s0] -> (128, s0)>()[%51]
        %54 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%53, %51]
        %55 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%54, %51]
        %56 = arith.cmpi slt, %55, %c128 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %23[0] [%55] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %56 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%23 : memref<128xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
        hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %57 = arith.index_cast %46 : i64 to index
        %58 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%57, %51]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%58], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %reinterpret_cast_3[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %22[0] [%55] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %56 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%22 : memref<128xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_4 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %59 = arith.index_cast %49 : i64 to index
        %60 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%59, %51]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %reinterpret_cast_6[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%60], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
      }
    } {pipeline.veconly}
    %35 = arith.addi %30, %c2_i32 : i32
    %36 = arith.cmpi slt, %35, %c1_i32 : i32
    scf.if %36 {
      %39 = arith.index_cast %35 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%39], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %40 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %41 = arith.cmpi sge, %40, %c0_i64 : i64
      scf.if %41 {
        %42 = arith.divsi %40, %c8_i64 : i64
        %43 = arith.remsi %40, %c8_i64 : i64
        %44 = arith.extsi %35 : i32 to i64
        %45 = arith.muli %44, %arg10 : i64
        %46 = arith.muli %44, %arg11 : i64
        %47 = arith.muli %42, %arg12 : i64
        %48 = arith.muli %43, %arg13 : i64
        %49 = arith.addi %47, %48 : i64
        %50 = arith.index_cast %45 : i64 to index
        %51 = arith.index_cast %31 : i32 to index
        %52 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%50, %51]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %53 = affine.max affine_map<()[s0] -> (128, s0)>()[%51]
        %54 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%53, %51]
        %55 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%54, %51]
        %56 = arith.cmpi slt, %55, %c128 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %21[0] [%55] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %56 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%21 : memref<128xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
        hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %57 = arith.index_cast %46 : i64 to index
        %58 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%57, %51]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%58], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %reinterpret_cast_3[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %20[0] [%55] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %56 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%20 : memref<128xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_4 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %59 = arith.index_cast %49 : i64 to index
        %60 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%59, %51]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %reinterpret_cast_6[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%60], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
      }
    } {pipeline.veconly}
    %37 = arith.addi %30, %c3_i32 : i32
    %38 = arith.cmpi slt, %37, %c1_i32 : i32
    scf.if %38 {
      %39 = arith.index_cast %37 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%39], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %40 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %41 = arith.cmpi sge, %40, %c0_i64 : i64
      scf.if %41 {
        %42 = arith.divsi %40, %c8_i64 : i64
        %43 = arith.remsi %40, %c8_i64 : i64
        %44 = arith.extsi %37 : i32 to i64
        %45 = arith.muli %44, %arg10 : i64
        %46 = arith.muli %44, %arg11 : i64
        %47 = arith.muli %42, %arg12 : i64
        %48 = arith.muli %43, %arg13 : i64
        %49 = arith.addi %47, %48 : i64
        %50 = arith.index_cast %45 : i64 to index
        %51 = arith.index_cast %31 : i32 to index
        %52 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%50, %51]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %53 = affine.max affine_map<()[s0] -> (128, s0)>()[%51]
        %54 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%53, %51]
        %55 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%54, %51]
        %56 = arith.cmpi slt, %55, %c128 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %19[0] [%55] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %56 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%19 : memref<128xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
        hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %57 = arith.index_cast %46 : i64 to index
        %58 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%57, %51]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%58], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %reinterpret_cast_3[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %18[0] [%55] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %56 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%18 : memref<128xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_4 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %59 = arith.index_cast %49 : i64 to index
        %60 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%59, %51]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %reinterpret_cast_6[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%60], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%55] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
      }
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID7>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
