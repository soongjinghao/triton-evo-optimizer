// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @mean_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c81984_i64 = arith.constant 81984 : i64
  %c49184_i64 = arith.constant 49184 : i64
  %c32800_i64 = arith.constant 32800 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c49216_i64 = arith.constant 49216 : i64
  %c0_i64 = arith.constant 0 : i64
  %c8191_i32 = arith.constant 8191 : i32
  %c0_i32 = arith.constant 0 : i32
  %c8192_i32 = arith.constant 8192 : i32
  %cst = arith.constant 0.000000e+00 : f32
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
  %5 = arith.addi %arg7, %c8191_i32 : i32
  %6 = arith.divsi %5, %c8192_i32 : i32
  %7 = arith.sitofp %arg7 : i32 to f32
  %8 = arith.index_cast %arg7 : i32 to index
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %9 = arith.divsi %arg11, %4 : i32
    %10 = arith.remsi %9, %arg8 : i32
    %11 = arith.muli %10, %c2_i32 : i32
    %12 = arith.addi %11, %c2_i32 : i32
    %13 = arith.minsi %12, %arg6 : i32
    %14 = arith.subi %13, %11 : i32
    scf.for %arg12 = %c0_i32 to %14 step %c1_i32  : i32 {
      %15 = arith.index_cast %arg12 : i32 to index
      %16 = arith.index_cast %c0_i32 : i32 to index
      %17 = arith.index_cast %14 : i32 to index
      %18 = arith.index_cast %c1_i32 : i32 to index
      %19 = arith.index_cast %arg11 : i32 to index
      %20 = arith.index_cast %3 : i32 to index
      %21 = arith.index_cast %1 : i32 to index
      %22 = arith.index_cast %c48_i32 : i32 to index
      %23 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%15, %16, %17, %18, %19, %20, %22]
      %24 = arith.index_cast %23 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %25 = arith.select %24, %c0_i64_0, %c1_i64 : i64
      %26 = hivm.hir.pointer_cast(%c49184_i64, %c81984_i64) : memref<1xf32, #hivm.address_space<ub>>
      annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
      %27 = arith.addi %11, %arg12 : i32
      %28 = arith.muli %27, %arg5 : i32
      %29 = arith.index_cast %28 : i32 to index
      %30 = scf.for %arg13 = %c0_i32 to %6 step %c1_i32 iter_args(%arg14 = %cst) -> (f32)  : i32 {
        %33 = hivm.hir.pointer_cast(%c0_i64, %c49216_i64) : memref<8192xf32, #hivm.address_space<ub>>
        annotation.mark %33 {hivm.multi_buffer = 2 : i32} : memref<8192xf32, #hivm.address_space<ub>>
        %34 = arith.muli %arg13, %c8192_i32 : i32
        %35 = arith.index_cast %34 : i32 to index
        %36 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%29, %35]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%36], sizes: [8192], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %37 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%35, %8]
        %38 = affine.min affine_map<()[s0, s1] -> (s1 + 8192, s0)>()[%37, %35]
        %39 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%38, %35]
        %40 = arith.cmpi slt, %39, %c8192 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%39] [1] : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %33[0] [%39] [1] : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %40 {
          hivm.hir.vbrc ins(%cst : f32) outs(%33 : memref<8192xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %41 = hivm.hir.pointer_cast(%c32768_i64) : memref<1xf32, #hivm.address_space<ub>>
        %42 = hivm.hir.pointer_cast(%c32800_i64) : memref<4096xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vreduce <sum> ins(%33 : memref<8192xf32, #hivm.address_space<ub>>) outs(%41 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%42 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %43 = memref.load %41[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %44 = arith.addf %arg14, %43 : f32
        scf.yield %44 : f32
      } {pipeline.veconly}
      %31 = arith.divf %30, %7 : f32
      %32 = arith.index_cast %27 : i32 to index
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %25]
      memref.store %31, %26[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%32], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%26 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %25]
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
