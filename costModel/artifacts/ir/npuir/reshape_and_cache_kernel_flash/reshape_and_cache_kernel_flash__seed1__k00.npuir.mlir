// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @reshape_and_cache_kernel_flash(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i64 {tt.divisibility = 16 : i32}, %arg11: i64 {tt.divisibility = 16 : i32}, %arg12: i64 {tt.divisibility = 16 : i32}, %arg13: i64 {tt.divisibility = 16 : i32}, %arg14: i64 {tt.divisibility = 16 : i32}, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<18xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c15360_i64 = arith.constant 15360 : i64
  %c7168_i64 = arith.constant 7168 : i64
  %c14336_i64 = arith.constant 14336 : i64
  %c6144_i64 = arith.constant 6144 : i64
  %c13312_i64 = arith.constant 13312 : i64
  %c5120_i64 = arith.constant 5120 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c11264_i64 = arith.constant 11264 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c9216_i64 = arith.constant 9216 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c3_i32 = arith.constant 3 : i32
  %c2_i32 = arith.constant 2 : i32
  %c8_i64 = arith.constant 8 : i64
  %c512_i32 = arith.constant 512 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg15, %arg16 : i32
  %1 = arith.muli %0, %arg17 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg17, %arg16 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID7>]
  scf.for %arg18 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg18 : i32 to index
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
    %18 = hivm.hir.pointer_cast(%c7168_i64, %c15360_i64) : memref<512xf16, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c6144_i64, %c14336_i64) : memref<512xf16, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c5120_i64, %c13312_i64) : memref<512xf16, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c4096_i64, %c12288_i64) : memref<512xf16, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c3072_i64, %c11264_i64) : memref<512xf16, #hivm.address_space<ub>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c2048_i64, %c10240_i64) : memref<512xf16, #hivm.address_space<ub>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c1024_i64, %c9216_i64) : memref<512xf16, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c0_i64, %c8192_i64) : memref<512xf16, #hivm.address_space<ub>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<512xf16, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %26 = arith.divsi %arg18, %arg17 : i32
    %27 = arith.remsi %26, %arg16 : i32
    %28 = arith.divsi %arg18, %4 : i32
    %29 = arith.remsi %28, %arg15 : i32
    %30 = arith.muli %29, %c4_i32 : i32
    %31 = arith.muli %27, %c512_i32 : i32
    %32 = arith.extsi %30 : i32 to i64
    %33 = arith.cmpi slt, %32, %arg14 : i64
    scf.if %33 {
      %43 = arith.index_cast %30 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%43], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %44 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %45 = arith.cmpi sge, %44, %c0_i64 : i64
      scf.if %45 {
        %46 = arith.divsi %44, %c8_i64 : i64
        %47 = arith.remsi %44, %c8_i64 : i64
        %48 = arith.muli %32, %arg10 : i64
        %49 = arith.muli %32, %arg11 : i64
        %50 = arith.muli %46, %arg12 : i64
        %51 = arith.muli %47, %arg13 : i64
        %52 = arith.addi %50, %51 : i64
        %53 = arith.index_cast %48 : i64 to index
        %54 = arith.index_cast %31 : i32 to index
        %55 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%53, %54]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%55], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %56 = affine.max affine_map<()[s0] -> (512, s0)>()[%54]
        %57 = affine.min affine_map<()[s0, s1] -> (s1 + 512, s0)>()[%56, %54]
        %58 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%57, %54]
        %59 = arith.cmpi slt, %58, %c512 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %25[0] [%58] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %59 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%25 : memref<512xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %17]
        hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %60 = arith.index_cast %49 : i64 to index
        %61 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%60, %54]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%61], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %reinterpret_cast_3[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %24[0] [%58] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %59 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%24 : memref<512xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_4 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %62 = arith.index_cast %52 : i64 to index
        %63 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%62, %54]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%63], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %reinterpret_cast_6[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%63], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %17]
      }
    } {pipeline.veconly}
    %34 = arith.addi %30, %c1_i32 : i32
    %35 = arith.extsi %34 : i32 to i64
    %36 = arith.cmpi slt, %35, %arg14 : i64
    scf.if %36 {
      %43 = arith.index_cast %34 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%43], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %44 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %45 = arith.cmpi sge, %44, %c0_i64 : i64
      scf.if %45 {
        %46 = arith.divsi %44, %c8_i64 : i64
        %47 = arith.remsi %44, %c8_i64 : i64
        %48 = arith.muli %35, %arg10 : i64
        %49 = arith.muli %35, %arg11 : i64
        %50 = arith.muli %46, %arg12 : i64
        %51 = arith.muli %47, %arg13 : i64
        %52 = arith.addi %50, %51 : i64
        %53 = arith.index_cast %48 : i64 to index
        %54 = arith.index_cast %31 : i32 to index
        %55 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%53, %54]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%55], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %56 = affine.max affine_map<()[s0] -> (512, s0)>()[%54]
        %57 = affine.min affine_map<()[s0, s1] -> (s1 + 512, s0)>()[%56, %54]
        %58 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%57, %54]
        %59 = arith.cmpi slt, %58, %c512 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %23[0] [%58] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %59 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%23 : memref<512xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
        hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %60 = arith.index_cast %49 : i64 to index
        %61 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%60, %54]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%61], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %reinterpret_cast_3[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %22[0] [%58] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %59 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%22 : memref<512xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_4 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %62 = arith.index_cast %52 : i64 to index
        %63 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%62, %54]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%63], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %reinterpret_cast_6[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%63], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %15]
      }
    } {pipeline.veconly}
    %37 = arith.addi %30, %c2_i32 : i32
    %38 = arith.extsi %37 : i32 to i64
    %39 = arith.cmpi slt, %38, %arg14 : i64
    scf.if %39 {
      %43 = arith.index_cast %37 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%43], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %44 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %45 = arith.cmpi sge, %44, %c0_i64 : i64
      scf.if %45 {
        %46 = arith.divsi %44, %c8_i64 : i64
        %47 = arith.remsi %44, %c8_i64 : i64
        %48 = arith.muli %38, %arg10 : i64
        %49 = arith.muli %38, %arg11 : i64
        %50 = arith.muli %46, %arg12 : i64
        %51 = arith.muli %47, %arg13 : i64
        %52 = arith.addi %50, %51 : i64
        %53 = arith.index_cast %48 : i64 to index
        %54 = arith.index_cast %31 : i32 to index
        %55 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%53, %54]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%55], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %56 = affine.max affine_map<()[s0] -> (512, s0)>()[%54]
        %57 = affine.min affine_map<()[s0, s1] -> (s1 + 512, s0)>()[%56, %54]
        %58 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%57, %54]
        %59 = arith.cmpi slt, %58, %c512 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %21[0] [%58] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %59 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%21 : memref<512xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
        hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %60 = arith.index_cast %49 : i64 to index
        %61 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%60, %54]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%61], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %reinterpret_cast_3[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %20[0] [%58] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %59 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%20 : memref<512xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_4 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %62 = arith.index_cast %52 : i64 to index
        %63 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%62, %54]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%63], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %reinterpret_cast_6[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%63], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_9 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %13]
      }
    } {pipeline.veconly}
    %40 = arith.addi %30, %c3_i32 : i32
    %41 = arith.extsi %40 : i32 to i64
    %42 = arith.cmpi slt, %41, %arg14 : i64
    scf.if %42 {
      %43 = arith.index_cast %40 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%43], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %44 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %45 = arith.cmpi sge, %44, %c0_i64 : i64
      scf.if %45 {
        %46 = arith.divsi %44, %c8_i64 : i64
        %47 = arith.remsi %44, %c8_i64 : i64
        %48 = arith.muli %41, %arg10 : i64
        %49 = arith.muli %41, %arg11 : i64
        %50 = arith.muli %46, %arg12 : i64
        %51 = arith.muli %47, %arg13 : i64
        %52 = arith.addi %50, %51 : i64
        %53 = arith.index_cast %48 : i64 to index
        %54 = arith.index_cast %31 : i32 to index
        %55 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%53, %54]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%55], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %56 = affine.max affine_map<()[s0] -> (512, s0)>()[%54]
        %57 = affine.min affine_map<()[s0, s1] -> (s1 + 512, s0)>()[%56, %54]
        %58 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%57, %54]
        %59 = arith.cmpi slt, %58, %c512 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %19[0] [%58] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %59 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%19 : memref<512xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %11]
        hivm.hir.load ins(%subview : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        %60 = arith.index_cast %49 : i64 to index
        %61 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%60, %54]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%61], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_4 = memref.subview %reinterpret_cast_3[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %18[0] [%58] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        scf.if %59 {
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vbrc ins(%cst : f16) outs(%18 : memref<512xf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_4 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_5 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        %62 = arith.index_cast %52 : i64 to index
        %63 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%62, %54]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%63], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_7 = memref.subview %reinterpret_cast_6[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_2 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%63], sizes: [512], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%58] [1] : memref<512xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
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
