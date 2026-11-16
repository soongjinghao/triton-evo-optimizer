// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_correct_attn_cp_out_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1248_i64 = arith.constant 1248 : i64
  %c672_i64 = arith.constant 672 : i64
  %c992_i64 = arith.constant 992 : i64
  %c416_i64 = arith.constant 416 : i64
  %c384_i64 = arith.constant 384 : i64
  %c960_i64 = arith.constant 960 : i64
  %c352_i64 = arith.constant 352 : i64
  %c320_i64 = arith.constant 320 : i64
  %c288_i64 = arith.constant 288 : i64
  %c256_i64 = arith.constant 256 : i64
  %c224_i64 = arith.constant 224 : i64
  %c160_i64 = arith.constant 160 : i64
  %c128_i64 = arith.constant 128 : i64
  %c96_i64 = arith.constant 96 : i64
  %c64_i64 = arith.constant 64 : i64
  %c928_i64 = arith.constant 928 : i64
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0x7F800000 : f32
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0xFF800000 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg14, %arg13 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<4xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<4xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg15 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg15 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c0_i64_2 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_2, %c1_i64 : i64
    %13 = hivm.hir.pointer_cast(%c672_i64, %c1248_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c416_i64, %c992_i64) : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c352_i64, %c960_i64) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c32_i64, %c928_i64) : memref<4xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %17 = arith.divsi %arg15, %arg14 : i32
    %18 = arith.remsi %17, %arg13 : i32
    %19 = arith.divsi %arg15, %4 : i32
    %20 = arith.remsi %19, %arg12 : i32
    %21 = arith.muli %20, %arg9 : i32
    %22 = arith.muli %18, %arg10 : i32
    %23 = arith.index_cast %21 : i32 to index
    %24 = arith.index_cast %22 : i32 to index
    %25 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%23, %24]
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%25], sizes: [4], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast : memref<4xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%16 : memref<4xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %26 = hivm.hir.pointer_cast(%c64_i64) : memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcmp ins(%16, %16 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%26 : memref<4xi1, #hivm.address_space<ub>>)
    %27 = hivm.hir.pointer_cast(%c64_i64) : memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%26 : memref<4xi1, #hivm.address_space<ub>>) outs(%27 : memref<4xi1, #hivm.address_space<ub>>)
    %28 = hivm.hir.pointer_cast(%c96_i64) : memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%16, %5 : memref<4xf32, #hivm.address_space<ub>>, memref<4xf32, #hivm.address_space<ub>>) outs(%28 : memref<4xi1, #hivm.address_space<ub>>)
    %29 = hivm.hir.pointer_cast(%c96_i64) : memref<4xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vor ins(%27, %28 : memref<4xi1, #hivm.address_space<ub>>, memref<4xi1, #hivm.address_space<ub>>) outs(%29 : memref<4xi1, #hivm.address_space<ub>>)
    %30 = hivm.hir.pointer_cast(%c128_i64) : memref<4xf32, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c160_i64) : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsel ins(%29, %cst_0, %16 : memref<4xi1, #hivm.address_space<ub>>, f32, memref<4xf32, #hivm.address_space<ub>>) outs(%30 : memref<4xf32, #hivm.address_space<ub>>) temp_buffer(%31 : memref<16xf32, #hivm.address_space<ub>>)
    %32 = hivm.hir.pointer_cast(%c224_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <max> ins(%30 : memref<4xf32, #hivm.address_space<ub>>) outs(%32 : memref<1xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %33 = memref.load %32[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %34 = arith.cmpf oeq, %33, %cst_0 : f32
    %35 = arith.select %34, %cst_1, %33 : f32
    %36 = hivm.hir.pointer_cast(%c128_i64) : memref<4xf32, #hivm.address_space<ub>>
    %37 = arith.subf %cst_1, %35 : f32
    hivm.hir.vadd ins(%30, %37 : memref<4xf32, #hivm.address_space<ub>>, f32) outs(%36 : memref<4xf32, #hivm.address_space<ub>>)
    %38 = hivm.hir.pointer_cast(%c128_i64) : memref<4xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%36 : memref<4xf32, #hivm.address_space<ub>>) outs(%38 : memref<4xf32, #hivm.address_space<ub>>)
    %39 = hivm.hir.pointer_cast(%c256_i64) : memref<1xf32, #hivm.address_space<ub>>
    %40 = hivm.hir.pointer_cast(%c288_i64) : memref<2xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%38 : memref<4xf32, #hivm.address_space<ub>>) outs(%39 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%40 : memref<2xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %41 = memref.load %39[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %42 = hivm.hir.pointer_cast(%c320_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %41, %42[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %43 = hivm.hir.pointer_cast(%c320_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vln ins(%42 : memref<1xf32, #hivm.address_space<ub>>) outs(%43 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %44 = memref.load %43[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %45 = arith.addf %44, %35 : f32
    %46 = arith.addi %21, %22 : i32
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %12]
    memref.store %45, %15[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    %47 = arith.index_cast %46 : i32 to index
    %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%47], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%15 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_3 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %12]
    %48 = arith.muli %20, %arg7 : i32
    %49 = arith.muli %18, %arg8 : i32
    %50 = arith.addi %48, %49 : i32
    %51 = arith.addi %arg11, %21 : i32
    %52 = arith.addi %51, %22 : i32
    %53 = arith.index_cast %52 : i32 to index
    %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%53], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %54 = memref.load %reinterpret_cast_4[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %55 = arith.subf %54, %45 : f32
    %56 = arith.cmpf une, %55, %55 : f32
    %57 = arith.cmpf oeq, %55, %cst : f32
    %58 = arith.ori %56, %57 : i1
    %59 = arith.select %58, %cst_0, %55 : f32
    %60 = hivm.hir.pointer_cast(%c384_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %59, %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %61 = hivm.hir.pointer_cast(%c384_i64) : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vexp ins(%60 : memref<1xf32, #hivm.address_space<ub>>) outs(%61 : memref<1xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %62 = memref.load %61[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %63 = arith.index_cast %50 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%63], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_5 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%14 : memref<64xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %12]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%14, %62 : memref<64xf32, #hivm.address_space<ub>>, f32) outs(%13 : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%63], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%13 : memref<64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_6 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %12]
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
