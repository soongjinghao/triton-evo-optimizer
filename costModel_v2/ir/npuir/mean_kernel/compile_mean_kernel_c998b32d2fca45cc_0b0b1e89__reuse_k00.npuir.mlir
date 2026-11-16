// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @mean_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c82016_i64 = arith.constant 82016 : i64
  %c49216_i64 = arith.constant 49216 : i64
  %c32832_i64 = arith.constant 32832 : i64
  %c32800_i64 = arith.constant 32800 : i64
  %c49248_i64 = arith.constant 49248 : i64
  %c32_i64 = arith.constant 32 : i64
  %c0_i64 = arith.constant 0 : i64
  %c8191_i32 = arith.constant 8191 : i32
  %c0_i32 = arith.constant 0 : i32
  %c8192_i32 = arith.constant 8192 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c8192 = arith.constant 8192 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = arith.sitofp %arg7 : i32 to f32
  %6 = arith.divf %cst_0, %5 : f32
  %7 = arith.addi %arg7, %c8191_i32 : i32
  %8 = arith.divsi %7, %c8192_i32 : i32
  %9 = arith.index_cast %arg7 : i32 to index
  %10 = hivm.hir.pointer_cast(%c0_i64) : memref<1xf32, #hivm.address_space<ub>>
  memref.store %6, %10[%c0] : memref<1xf32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %11 = arith.divsi %arg11, %4 : i32
    %12 = arith.remsi %11, %arg8 : i32
    %13 = arith.muli %12, %c2_i32 : i32
    %14 = arith.addi %13, %c2_i32 : i32
    %15 = arith.minsi %14, %arg6 : i32
    %16 = arith.subi %15, %13 : i32
    scf.for %arg12 = %c0_i32 to %16 step %c1_i32  : i32 {
      %17 = arith.index_cast %arg12 : i32 to index
      %18 = arith.index_cast %c0_i32 : i32 to index
      %19 = arith.index_cast %16 : i32 to index
      %20 = arith.index_cast %c1_i32 : i32 to index
      %21 = arith.index_cast %arg11 : i32 to index
      %22 = arith.index_cast %3 : i32 to index
      %23 = arith.index_cast %1 : i32 to index
      %24 = arith.index_cast %c48_i32 : i32 to index
      %25 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%17, %18, %19, %20, %21, %22, %24]
      %26 = arith.index_cast %25 : index to i1
      %c0_i64_1 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %27 = arith.select %26, %c0_i64_1, %c1_i64 : i64
      %28 = hivm.hir.pointer_cast(%c49216_i64, %c82016_i64) : memref<1xf32, #hivm.address_space<ub>>
      annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
      %29 = arith.addi %13, %arg12 : i32
      %30 = arith.muli %29, %arg5 : i32
      %31 = arith.index_cast %30 : i32 to index
      %32 = scf.for %arg13 = %c0_i32 to %8 step %c1_i32 iter_args(%arg14 = %cst) -> (f32)  : i32 {
        %36 = hivm.hir.pointer_cast(%c32_i64, %c49248_i64) : memref<8192xf32, #hivm.address_space<ub>>
        annotation.mark %36 {hivm.multi_buffer = 2 : i32} : memref<8192xf32, #hivm.address_space<ub>>
        %37 = arith.muli %arg13, %c8192_i32 : i32
        %38 = arith.index_cast %37 : i32 to index
        %39 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%31, %38]
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [8192], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %40 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%38, %9]
        %41 = affine.min affine_map<()[s0, s1] -> (s1 + 8192, s0)>()[%40, %38]
        %42 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%41, %38]
        %43 = arith.cmpi slt, %42, %c8192 : index
        %subview = memref.subview %reinterpret_cast_2[0] [%42] [1] : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_3 = memref.subview %36[0] [%42] [1] : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %43 {
          hivm.hir.vbrc ins(%cst : f32) outs(%36 : memref<8192xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %44 = hivm.hir.pointer_cast(%c32800_i64) : memref<1xf32, #hivm.address_space<ub>>
        %45 = hivm.hir.pointer_cast(%c32832_i64) : memref<4096xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vreduce <sum> ins(%36 : memref<8192xf32, #hivm.address_space<ub>>) outs(%44 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%45 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %46 = memref.load %44[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %47 = arith.addf %arg14, %46 : f32
        scf.yield %47 : f32
      } {pipeline.veconly}
      %33 = memref.load %10[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %34 = arith.mulf %32, %33 : f32
      %35 = arith.index_cast %29 : i32 to index
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %27]
      memref.store %34, %28[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%35], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%28 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %27]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
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
