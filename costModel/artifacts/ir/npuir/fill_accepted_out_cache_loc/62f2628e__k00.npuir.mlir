// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @fill_accepted_out_cache_loc(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1 = arith.constant 1 : index
  %c4672_i64 = arith.constant 4672 : i64
  %c3616_i64 = arith.constant 3616 : i64
  %c3584_i64 = arith.constant 3584 : i64
  %c3552_i64 = arith.constant 3552 : i64
  %c3520_i64 = arith.constant 3520 : i64
  %c3488_i64 = arith.constant 3488 : i64
  %c3200_i64 = arith.constant 3200 : i64
  %c3168_i64 = arith.constant 3168 : i64
  %c3072_i64 = arith.constant 3072 : i64
  %c2560_i64 = arith.constant 2560 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c2304_i64 = arith.constant 2304 : i64
  %c3648_i64 = arith.constant 3648 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %cst = arith.constant 0.000000e+00 : f16
  %c-1_i64 = arith.constant -1 : i64
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1_i32 = arith.constant 1 : i32
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
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %6 = hivm.hir.pointer_cast(%c1024_i64, %c3648_i64) : memref<128xi64, #hivm.address_space<ub>>
    annotation.mark %6 {hivm.multi_buffer = 2 : i32} : memref<128xi64, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %7 = arith.divsi %arg9, %4 : i32
    %8 = arith.remsi %7, %arg6 : i32
    %9 = arith.divsi %8, %c16_i32 : i32
    %10 = arith.remsi %8, %c16_i32 : i32
    %11 = arith.cmpi ne, %10, %c0_i32 : i32
    scf.if %11 {
    } else {
      %12 = arith.muli %9, %c16_i32 : i32
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [128], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<128xi64, strided<[1]>, #hivm.address_space<gm>>
      %13 = arith.index_cast %12 : i32 to index
      %14 = affine.max affine_map<()[s0] -> (0, s0)>()[%13]
      %15 = affine.min affine_map<()[s0] -> (128, s0)>()[%14]
      %16 = arith.cmpi slt, %15, %c128 : index
      %subview = memref.subview %reinterpret_cast[0] [%15] [1] : memref<128xi64, strided<[1]>, #hivm.address_space<gm>> to memref<?xi64, strided<[1]>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %6[0] [%15] [1] : memref<128xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
      scf.if %16 {
        hivm.hir.vbrc ins(%c-1_i64 : i64) outs(%6 : memref<128xi64, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview : memref<?xi64, strided<[1]>, #hivm.address_space<gm>>) outs(%subview_0 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %c-1_i64 : i64 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %17 = hivm.hir.pointer_cast(%c2304_i64) : memref<128xi1, #hivm.address_space<ub>>
      %18 = hivm.hir.pointer_cast(%c1024_i64, %c3648_i64) : memref<128xi8, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg10 = %c0 to %c128 step %c1 {
        %29 = memref.load %6[%arg10] : memref<128xi64, #hivm.address_space<ub>>
        %30 = memref.load %5[%arg10] : memref<128xi64, #hivm.address_space<ub>>
        %31 = arith.cmpi eq, %29, %30 : i64
        %32 = arith.extui %31 : i1 to i8
        memref.store %32, %18[%arg10] : memref<128xi8, #hivm.address_space<ub>>
      }
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %19 = hivm.hir.pointer_cast(%c2048_i64) : memref<128xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%18 : memref<128xi8, #hivm.address_space<ub>>) outs(%19 : memref<128xf16, #hivm.address_space<ub>>)
      %20 = hivm.hir.pointer_cast(%c2304_i64) : memref<128xf16, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f16) outs(%20 : memref<128xf16, #hivm.address_space<ub>>)
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%19, %20 : memref<128xf16, #hivm.address_space<ub>>, memref<128xf16, #hivm.address_space<ub>>) outs(%17 : memref<128xi1, #hivm.address_space<ub>>) compare_mode = <ne>
      %21 = hivm.hir.pointer_cast(%c2304_i64) : memref<128xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%17 : memref<128xi1, #hivm.address_space<ub>>) outs(%21 : memref<128xi1, #hivm.address_space<ub>>)
      %22 = hivm.hir.pointer_cast(%c2560_i64) : memref<128xi32, #hivm.address_space<ub>>
      %23 = hivm.hir.pointer_cast(%c3072_i64) : memref<24xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%21 : memref<128xi1, #hivm.address_space<ub>>) outs(%22 : memref<128xi32, #hivm.address_space<ub>>) temp_buffer(%23 : memref<24xi32, #hivm.address_space<ub>>)
      %24 = hivm.hir.pointer_cast(%c3168_i64) : memref<1xi32, #hivm.address_space<ub>>
      %25 = hivm.hir.pointer_cast(%c3200_i64) : memref<72xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%22 : memref<128xi32, #hivm.address_space<ub>>) outs(%24 : memref<1xi32, #hivm.address_space<ub>>) temp_buffer(%25 : memref<72xi32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %26 = memref.load %24[%c0] : memref<1xi32, #hivm.address_space<ub>>
      %27 = arith.index_cast %26 : i32 to index
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
      %28 = scf.for %arg10 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg11 = %c0) -> (index)  : i32 {
        %29 = arith.index_cast %arg10 : i32 to index
        %30 = arith.index_cast %c0_i32 : i32 to index
        %31 = arith.index_cast %c16_i32 : i32 to index
        %32 = arith.index_cast %c1_i32 : i32 to index
        %33 = arith.index_cast %arg9 : i32 to index
        %34 = arith.index_cast %3 : i32 to index
        %35 = arith.index_cast %1 : i32 to index
        %36 = arith.index_cast %c48_i32 : i32 to index
        %37 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%29, %30, %31, %32, %33, %34, %36]
        %38 = arith.index_cast %37 : index to i1
        %c0_i64_1 = arith.constant 0 : i64
        %c1_i64 = arith.constant 1 : i64
        %39 = arith.select %38, %c0_i64_1, %c1_i64 : i64
        %40 = hivm.hir.pointer_cast(%c3616_i64, %c4672_i64) : memref<1xi64, #hivm.address_space<ub>>
        annotation.mark %40 {hivm.multi_buffer = 2 : i32} : memref<1xi64, #hivm.address_space<ub>>
        %41 = arith.addi %12, %arg10 : i32
        %42 = arith.cmpi slt, %41, %c128_i32 : i32
        %43 = arith.index_cast %41 : i32 to index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %44 = memref.load %reinterpret_cast_2[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %45 = hivm.hir.pointer_cast(%c3488_i64) : memref<1xi1, #hivm.address_space<ub>>
        memref.store %42, %45[%c0] : memref<1xi1, #hivm.address_space<ub>>
        %46 = hivm.hir.pointer_cast(%c3520_i64) : memref<1xi64, #hivm.address_space<ub>>
        memref.store %44, %46[%c0] : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %47 = hivm.hir.pointer_cast(%c3520_i64) : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vsel ins(%45, %46, %c-1_i64 : memref<1xi1, #hivm.address_space<ub>>, memref<1xi64, #hivm.address_space<ub>>, i64) outs(%47 : memref<1xi64, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %48 = memref.load %47[%c0] : memref<1xi64, #hivm.address_space<ub>>
        %49 = arith.cmpi sgt, %48, %c-1_i64 : i64
        %50 = arith.index_cast %48 : i64 to index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%50], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %51 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %52 = hivm.hir.pointer_cast(%c3552_i64) : memref<1xi1, #hivm.address_space<ub>>
        memref.store %49, %52[%c0] : memref<1xi1, #hivm.address_space<ub>>
        %53 = hivm.hir.pointer_cast(%c3584_i64) : memref<1xi64, #hivm.address_space<ub>>
        memref.store %51, %53[%c0] : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %54 = hivm.hir.pointer_cast(%c3584_i64) : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vsel ins(%52, %53, %c0_i64 : memref<1xi1, #hivm.address_space<ub>>, memref<1xi64, #hivm.address_space<ub>>, i64) outs(%54 : memref<1xi64, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %55 = memref.load %54[%c0] : memref<1xi64, #hivm.address_space<ub>>
        %56 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%27, %arg11]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%56], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %39]
        memref.store %55, %40[%c0] : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %57 = arith.index_castui %49 : i1 to index
        %subview_5 = memref.subview %40[0] [%57] [1] : memref<1xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
        %subview_6 = memref.subview %reinterpret_cast_4[0] [%57] [1] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        hivm.hir.store ins(%subview_5 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_6 : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %39]
        %58 = arith.extui %49 : i1 to i32
        %59 = arith.index_cast %58 : i32 to index
        %60 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg11, %59]
        scf.yield %60 : index
      }
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
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
