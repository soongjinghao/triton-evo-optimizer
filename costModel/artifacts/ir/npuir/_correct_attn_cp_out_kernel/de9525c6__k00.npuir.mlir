// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_correct_attn_cp_out_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1472_i64 = arith.constant 1472 : i64
  %c800_i64 = arith.constant 800 : i64
  %c1216_i64 = arith.constant 1216 : i64
  %c544_i64 = arith.constant 544 : i64
  %c512_i64 = arith.constant 512 : i64
  %c1184_i64 = arith.constant 1184 : i64
  %c480_i64 = arith.constant 480 : i64
  %c448_i64 = arith.constant 448 : i64
  %c320_i64 = arith.constant 320 : i64
  %c288_i64 = arith.constant 288 : i64
  %c160_i64 = arith.constant 160 : i64
  %c128_i64 = arith.constant 128 : i64
  %c1056_i64 = arith.constant 1056 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg14, %arg13 : i32
  %5 = arith.extsi %arg10 : i32 to i64
  %6 = arith.index_cast %arg9 : i32 to index
  %7 = arith.extsi %arg7 : i32 to i64
  %8 = arith.extsi %arg8 : i32 to i64
  %9 = arith.muli %arg11, %arg9 : i32
  %10 = arith.extsi %9 : i32 to i64
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg15 = %3 to %1 step %c48_i32  : i32 {
    %11 = arith.index_cast %arg15 : i32 to index
    %12 = arith.index_cast %3 : i32 to index
    %13 = arith.index_cast %1 : i32 to index
    %14 = arith.index_cast %c48_i32 : i32 to index
    %15 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%11, %12, %14]
    %16 = arith.index_cast %15 : index to i1
    %c0_i64_1 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %17 = arith.select %16, %c0_i64_1, %c1_i64 : i64
    %18 = hivm.hir.pointer_cast(%c800_i64, %c1472_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c544_i64, %c1216_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c480_i64, %c1184_i64) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c0_i64, %c1056_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<4x8xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %22 = arith.divsi %arg15, %arg14 : i32
    %23 = arith.remsi %22, %arg13 : i32
    %24 = arith.divsi %arg15, %4 : i32
    %25 = arith.remsi %24, %arg12 : i32
    %26 = arith.extsi %25 : i32 to i64
    %27 = arith.extsi %23 : i32 to i64
    %28 = arith.muli %26, %5 : i64
    %29 = arith.index_cast %28 : i64 to index
    %30 = arith.index_cast %23 : i32 to index
    %31 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %30]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%31], sizes: [4], strides: [%6] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %subview = memref.subview %21[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%reinterpret_cast : memref<4xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview : memref<4xf32, strided<[8]>, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %32 = hivm.hir.pointer_cast(%c128_i64) : memref<1xf32, #hivm.address_space<ub>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_2 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %32 : memref<1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_7 = memref.reinterpret_cast %base_buffer_3 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %33 = hivm.hir.pointer_cast(%c160_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vreduce <max> ins(%reinterpret_cast_2 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_7 : memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) temp_buffer(%33 : memref<32xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %34 = memref.load %32[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %35 = arith.cmpf oeq, %34, %cst_0 : f32
    %36 = arith.select %35, %cst, %34 : f32
    %subview_8 = memref.subview %21[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    %37 = arith.subf %cst, %36 : f32
    %base_buffer_9, %offset_10, %sizes_11, %strides_12 = memref.extract_strided_metadata %subview_8 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_13 = memref.reinterpret_cast %base_buffer_9 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%reinterpret_cast_2, %37 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>, f32) outs(%reinterpret_cast_13 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %subview_14 = memref.subview %21[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_15, %offset_16, %sizes_17, %strides_18 = memref.extract_strided_metadata %subview_14 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_19 = memref.reinterpret_cast %base_buffer_15 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%reinterpret_cast_13 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_19 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c288_i64) : memref<1xf32, #hivm.address_space<ub>>
    %base_buffer_20, %offset_21, %sizes_22, %strides_23 = memref.extract_strided_metadata %38 : memref<1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_24 = memref.reinterpret_cast %base_buffer_20 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
    %39 = hivm.hir.pointer_cast(%c320_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%reinterpret_cast_19 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>) outs(%reinterpret_cast_24 : memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>) temp_buffer(%39 : memref<32xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %40 = memref.load %38[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %41 = hivm.hir.pointer_cast(%c448_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %40, %41[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %42 = hivm.hir.pointer_cast(%c448_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vln ins(%41 : memref<1xf32, #hivm.address_space<ub>>) outs(%42 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %43 = memref.load %42[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %44 = arith.addf %43, %36 : f32
    %45 = arith.addi %28, %27 : i64
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %17]
    memref.store %44, %20[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %46 = arith.index_cast %45 : i64 to index
    %reinterpret_cast_25 = memref.reinterpret_cast %arg6 to offset: [%46], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%20 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_25 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %17]
    %47 = arith.muli %26, %7 : i64
    %48 = arith.muli %27, %8 : i64
    %49 = arith.addi %47, %48 : i64
    %50 = arith.addi %10, %28 : i64
    %51 = arith.addi %50, %27 : i64
    %52 = arith.index_cast %51 : i64 to index
    %reinterpret_cast_26 = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %53 = memref.load %reinterpret_cast_26[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %54 = arith.subf %53, %44 : f32
    %55 = hivm.hir.pointer_cast(%c512_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %54, %55[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %56 = hivm.hir.pointer_cast(%c512_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vexp ins(%55 : memref<1xf32, #hivm.address_space<ub>>) outs(%56 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %57 = memref.load %56[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %58 = arith.index_cast %49 : i64 to index
    %reinterpret_cast_27 = memref.reinterpret_cast %arg3 to offset: [%58], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_27 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%19 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %17]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%19, %57 : memref<64xf32, #hivm.address_space<ub>>, f32) outs(%18 : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_28 = memref.reinterpret_cast %arg4 to offset: [%58], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%18 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_28 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %17]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
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
