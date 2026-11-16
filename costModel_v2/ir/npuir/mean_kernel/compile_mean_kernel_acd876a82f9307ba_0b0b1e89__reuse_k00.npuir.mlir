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
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg11 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg11 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = hivm.hir.pointer_cast(%c49184_i64, %c81984_i64) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %13 = arith.divsi %arg11, %4 : i32
    %14 = arith.remsi %13, %arg8 : i32
    %15 = arith.cmpi sge, %14, %arg6 : i32
    scf.if %15 {
    } else {
      %16 = arith.addi %arg7, %c8191_i32 : i32
      %17 = arith.divsi %16, %c8192_i32 : i32
      %18 = arith.muli %14, %arg5 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = arith.index_cast %arg7 : i32 to index
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %21 = scf.for %arg12 = %c0_i32 to %17 step %c1_i32 iter_args(%arg13 = %cst) -> (f32)  : i32 {
        %25 = hivm.hir.pointer_cast(%c0_i64, %c49216_i64) : memref<8192xf32, #hivm.address_space<ub>>
        annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<8192xf32, #hivm.address_space<ub>>
        %26 = arith.muli %arg12, %c8192_i32 : i32
        %27 = arith.index_cast %26 : i32 to index
        %28 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%19, %27]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%28], sizes: [8192], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %29 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%27, %20]
        %30 = affine.min affine_map<()[s0, s1] -> (s1 + 8192, s0)>()[%29, %27]
        %31 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%30, %27]
        %32 = arith.cmpi slt, %31, %c8192 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%31] [1] : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %25[0] [%31] [1] : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %32 {
          hivm.hir.vbrc ins(%cst : f32) outs(%25 : memref<8192xf32, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %33 = hivm.hir.pointer_cast(%c32768_i64) : memref<1xf32, #hivm.address_space<ub>>
        %34 = hivm.hir.pointer_cast(%c32800_i64) : memref<4096xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vreduce <sum> ins(%25 : memref<8192xf32, #hivm.address_space<ub>>) outs(%33 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%34 : memref<4096xf32, #hivm.address_space<ub>>) reduce_dims = [0]
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %35 = memref.load %33[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %36 = arith.addf %arg13, %35 : f32
        scf.yield %36 : f32
      }
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %22 = arith.sitofp %arg7 : i32 to f32
      %23 = arith.divf %21, %22 : f32
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %11]
      memref.store %23, %12[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %24 = arith.index_cast %14 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%24], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%12 : memref<1xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %11]
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
