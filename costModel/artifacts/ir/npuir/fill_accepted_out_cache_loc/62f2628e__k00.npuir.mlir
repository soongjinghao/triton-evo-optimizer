// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @fill_accepted_out_cache_loc(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c5280_i64 = arith.constant 5280 : i64
  %c4224_i64 = arith.constant 4224 : i64
  %c4192_i64 = arith.constant 4192 : i64
  %c3168_i64 = arith.constant 3168 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c2560_i64 = arith.constant 2560 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c2304_i64 = arith.constant 2304 : i64
  %c4256_i64 = arith.constant 4256 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f16
  %c-1_i64 = arith.constant -1 : i64
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<128xi64, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%c-1_i64 : i64) outs(%5 : memref<128xi64, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [128], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<128xi64, strided<[1]>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %6 = arith.index_cast %arg9 : i32 to index
    %7 = arith.index_cast %3 : i32 to index
    %8 = arith.index_cast %1 : i32 to index
    %9 = arith.index_cast %c48_i32 : i32 to index
    %10 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%6, %7, %9]
    %11 = arith.index_cast %10 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %12 = arith.select %11, %c0_i64_0, %c1_i64 : i64
    %13 = hivm.hir.pointer_cast(%c4224_i64, %c5280_i64) : memref<1xi64, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<1xi64, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c1024_i64, %c4256_i64) : memref<128xi64, #hivm.address_space<ub>>
    annotation.mark %14 {hivm.multi_buffer = 2 : i32} : memref<128xi64, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %15 = arith.divsi %arg9, %4 : i32
    %16 = arith.remsi %15, %arg6 : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = affine.max affine_map<()[s0] -> (0, s0)>()[%17]
    %19 = affine.min affine_map<()[s0] -> (128, s0)>()[%18]
    %20 = arith.cmpi slt, %19, %c128 : index
    %subview = memref.subview %reinterpret_cast[0] [%19] [1] : memref<128xi64, strided<[1]>, #hivm.address_space<gm>> to memref<?xi64, strided<[1]>, #hivm.address_space<gm>>
    %subview_1 = memref.subview %14[0] [%19] [1] : memref<128xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
    scf.if %20 {
      hivm.hir.vbrc ins(%c-1_i64 : i64) outs(%14 : memref<128xi64, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?xi64, strided<[1]>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c-1_i64 : i64 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    %21 = hivm.hir.pointer_cast(%c2304_i64) : memref<128xi1, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c1024_i64, %c4256_i64) : memref<128xi8, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg10 = %c0 to %c128 step %c1 {
      %33 = memref.load %14[%arg10] : memref<128xi64, #hivm.address_space<ub>>
      %34 = memref.load %5[%arg10] : memref<128xi64, #hivm.address_space<ub>>
      %35 = arith.cmpi eq, %33, %34 : i64
      %36 = arith.extui %35 : i1 to i8
      memref.store %36, %22[%arg10] : memref<128xi8, #hivm.address_space<ub>>
    }
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    %23 = hivm.hir.pointer_cast(%c2048_i64) : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcast ins(%22 : memref<128xi8, #hivm.address_space<ub>>) outs(%23 : memref<128xf16, #hivm.address_space<ub>>)
    %24 = hivm.hir.pointer_cast(%c2304_i64) : memref<128xf16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f16) outs(%24 : memref<128xf16, #hivm.address_space<ub>>)
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcmp ins(%23, %24 : memref<128xf16, #hivm.address_space<ub>>, memref<128xf16, #hivm.address_space<ub>>) outs(%21 : memref<128xi1, #hivm.address_space<ub>>) compare_mode = <ne>
    %25 = hivm.hir.pointer_cast(%c2304_i64) : memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vnot ins(%21 : memref<128xi1, #hivm.address_space<ub>>) outs(%25 : memref<128xi1, #hivm.address_space<ub>>)
    %26 = hivm.hir.pointer_cast(%c2560_i64) : memref<128xf32, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c3072_i64) : memref<24xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%25 : memref<128xi1, #hivm.address_space<ub>>) outs(%26 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%27 : memref<24xf32, #hivm.address_space<ub>>)
    %28 = hivm.hir.pointer_cast(%c3168_i64) : memref<128xi64, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%26 : memref<128xf32, #hivm.address_space<ub>>) outs(%28 : memref<128xi64, #hivm.address_space<ub>>) round_mode = <trunc>
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %29 = hivm.hir.pointer_cast(%c4192_i64) : memref<1xi64, #hivm.address_space<ub>>
    memref.store %c0_i64, %29[%c0] : memref<1xi64, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg10 = %c0 to %c128 step %c1 {
      %33 = memref.load %28[%arg10] : memref<128xi64, #hivm.address_space<ub>>
      %34 = arith.cmpi eq, %arg10, %c0 : index
      scf.if %34 {
        memref.store %c0_i64, %29[%c0] : memref<1xi64, #hivm.address_space<ub>>
      }
      %35 = memref.load %29[%c0] : memref<1xi64, #hivm.address_space<ub>>
      %36 = arith.addi %33, %35 : i64
      memref.store %36, %29[%c0] : memref<1xi64, #hivm.address_space<ub>>
    }
    %30 = memref.load %29[%c0] : memref<1xi64, #hivm.address_space<ub>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%17], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %31 = memref.load %reinterpret_cast_2[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %32 = arith.cmpi sgt, %31, %c-1_i64 : i64
    scf.if %32 {
      %33 = arith.index_cast %31 : i64 to index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%33], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %34 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %12]
      memref.store %34, %13[%c0] : memref<1xi64, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %35 = arith.index_cast %30 : i64 to index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%35], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%13 : memref<1xi64, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %12]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
