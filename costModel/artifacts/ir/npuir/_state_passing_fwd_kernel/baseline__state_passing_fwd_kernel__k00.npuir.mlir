// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_state_passing_fwd_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg9: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32, %arg30: i32, %arg31: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<32xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c96_i64 = arith.constant 96 : i64
  %c64_i64 = arith.constant 64 : i64
  %c224_i64 = arith.constant 224 : i64
  %c0_i64 = arith.constant 0 : i64
  %c288_i64 = arith.constant 288 : i64
  %c160_i64 = arith.constant 160 : i64
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg29, %arg30 : i32
  %1 = arith.muli %0, %arg31 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg31, %arg30 : i32
  %5 = arith.subi %arg14, %c1_i32 : i32
  %6 = arith.index_cast %arg11 : i32 to index
  %7 = arith.subi %arg12, %c1_i32 : i32
  %8 = arith.index_cast %arg19 : i32 to index
  %9 = arith.extsi %arg24 : i32 to i64
  %10 = arith.index_cast %arg16 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg32 = %3 to %1 step %c48_i32  : i32 {
    %11 = hivm.hir.pointer_cast(%c160_i64, %c288_i64) : memref<16xf32, #hivm.address_space<ub>>
    annotation.mark %11 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %12 = arith.remsi %arg32, %arg31 : i32
    %13 = arith.divsi %arg32, %arg31 : i32
    %14 = arith.remsi %13, %arg30 : i32
    %15 = arith.divsi %arg32, %4 : i32
    %16 = arith.remsi %15, %arg29 : i32
    %17 = arith.muli %14, %arg15 : i32
    %18 = arith.muli %12, %arg17 : i32
    %19 = arith.muli %14, %arg23 : i32
    %20 = arith.muli %12, %arg25 : i32
    %21 = arith.addi %19, %20 : i32
    %22 = arith.addi %21, %5 : i32
    %23 = arith.extsi %22 : i32 to i64
    %24 = arith.muli %14, %arg18 : i32
    %25 = arith.muli %12, %arg20 : i32
    %26 = arith.muli %14, %arg21 : i32
    %27 = arith.muli %12, %arg22 : i32
    %28 = arith.addi %26, %27 : i32
    %29 = arith.muli %12, %arg27 : i32
    %30 = arith.muli %14, %arg26 : i32
    %31 = arith.muli %16, %c16_i32 : i32
    %32 = arith.index_cast %24 : i32 to index
    %33 = arith.index_cast %25 : i32 to index
    %34 = arith.index_cast %31 : i32 to index
    %35 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%34, %32, %33]
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %36 = arith.index_cast %28 : i32 to index
    %37 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%36, %34]
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %38 = arith.index_cast %29 : i32 to index
    %39 = arith.index_cast %30 : i32 to index
    %40 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%34, %38, %39]
    %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%40], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %41 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%34, %6]
    %42 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%41, %34]
    %43 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%42, %34]
    %44 = arith.cmpi slt, %43, %c16 : index
    %subview = memref.subview %reinterpret_cast_1[0] [%43] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %11[0] [%43] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %44 {
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%cst : f32) outs(%11 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %subview_3 = memref.subview %reinterpret_cast[0] [%43] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %45 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%34, %8]
    %46 = arith.index_cast %17 : i32 to index
    %47 = arith.index_cast %18 : i32 to index
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    %48:4 = scf.for %arg33 = %c0_i32 to %arg12 step %c1_i32 iter_args(%arg34 = %11, %arg35 = %23, %arg36 = %34, %arg37 = %45) -> (memref<16xf32, #hivm.address_space<ub>>, i64, index, index)  : i32 {
      %49 = arith.index_cast %arg33 : i32 to index
      %50 = arith.index_cast %c0_i32 : i32 to index
      %51 = arith.index_cast %arg12 : i32 to index
      %52 = arith.index_cast %c1_i32 : i32 to index
      %53 = arith.index_cast %arg32 : i32 to index
      %54 = arith.index_cast %3 : i32 to index
      %55 = arith.index_cast %1 : i32 to index
      %56 = arith.index_cast %c48_i32 : i32 to index
      %57 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%49, %50, %51, %52, %53, %54, %56]
      %58 = arith.index_cast %57 : index to i1
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %59 = arith.select %58, %c1_i64, %c2_i64 : i64
      %60 = hivm.hir.pointer_cast(%c160_i64, %c288_i64) : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %60 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
      %61 = hivm.hir.pointer_cast(%c0_i64, %c224_i64) : memref<16xf32, #hivm.address_space<ub>>
      annotation.mark %61 {hivm.multi_buffer = 2 : i32} : memref<16xf32, #hivm.address_space<ub>>
      %62 = arith.index_cast %arg35 : i64 to index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%62], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %63 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%arg36, %46, %47]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%63], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %64 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%arg37, %32, %33]
      %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%64], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %reinterpret_cast_5[0] [%43] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_8 = memref.subview %61[0] [%43] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %44 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst : f32) outs(%61 : memref<16xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %59]
      hivm.hir.load ins(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_8 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %65 = memref.load %reinterpret_cast_4[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %66 = hivm.hir.pointer_cast(%c64_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %65, %66[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %67 = hivm.hir.pointer_cast(%c64_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vexp ins(%66 : memref<1xf32, #hivm.address_space<ub>>) outs(%67 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %68 = memref.load %67[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %69 = hivm.hir.pointer_cast(%c96_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%arg34, %68 : memref<16xf32, #hivm.address_space<ub>>, f32) outs(%69 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %59]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%69, %61 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%60 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %59]
      %70 = arith.cmpi slt, %arg33, %7 : i32
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %70 {
        %subview_9 = memref.subview %60[0] [%43] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %subview_10 = memref.subview %reinterpret_cast_6[0] [%43] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_9 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_10 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      } else {
        %subview_9 = memref.subview %60[0] [%43] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %subview_10 = memref.subview %reinterpret_cast_0[0] [%43] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_9 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_10 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %59]
      %71 = arith.addi %arg35, %9 : i64
      %72 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg36, %10]
      %73 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg37, %8]
      scf.yield %60, %71, %72, %73 : memref<16xf32, #hivm.address_space<ub>>, i64, index, index
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
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
