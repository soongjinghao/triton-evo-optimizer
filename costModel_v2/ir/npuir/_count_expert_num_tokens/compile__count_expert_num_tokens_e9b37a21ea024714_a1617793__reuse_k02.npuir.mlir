// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_count_expert_num_tokens(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c512_i64 = arith.constant 512 : i64
  %c448_i64 = arith.constant 448 : i64
  %c384_i64 = arith.constant 384 : i64
  %c352_i64 = arith.constant 352 : i64
  %c256_i64 = arith.constant 256 : i64
  %c224_i64 = arith.constant 224 : i64
  %c192_i64 = arith.constant 192 : i64
  %c128_i64 = arith.constant 128 : i64
  %c96_i64 = arith.constant 96 : i64
  %c64_i64 = arith.constant 64 : i64
  %c32_i64 = arith.constant 32 : i64
  %c480_i64 = arith.constant 480 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c-1_i32 = arith.constant -1 : i32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    %5 = arith.index_cast %arg9 : i32 to index
    %6 = arith.index_cast %3 : i32 to index
    %7 = arith.index_cast %1 : i32 to index
    %8 = arith.index_cast %c48_i32 : i32 to index
    %9 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%5, %6, %8]
    %10 = arith.index_cast %9 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.select %10, %c0_i64_0, %c1_i64 : i64
    %12 = hivm.hir.pointer_cast(%c448_i64, %c512_i64) : memref<1xi32, #hivm.address_space<ub>>
    annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<1xi32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c0_i64, %c480_i64) : memref<8xi32, #hivm.address_space<ub>>
    annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<8xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %14 = arith.divsi %arg9, %4 : i32
    %15 = arith.remsi %14, %arg6 : i32
    %16 = arith.cmpi sge, %15, %c4_i32 : i32
    scf.if %16 {
    } else {
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<8xi32, strided<[1]>, #hivm.address_space<gm>>
      hivm.hir.load ins(%reinterpret_cast : memref<8xi32, strided<[1]>, #hivm.address_space<gm>>) outs(%13 : memref<8xi32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %17 = hivm.hir.pointer_cast(%c32_i64) : memref<8xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmax ins(%13, %c0_i32 : memref<8xi32, #hivm.address_space<ub>>, i32) outs(%17 : memref<8xi32, #hivm.address_space<ub>>)
      %18 = hivm.hir.pointer_cast(%c32_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%17, %13 : memref<8xi32, #hivm.address_space<ub>>, memref<8xi32, #hivm.address_space<ub>>) outs(%18 : memref<8xi1, #hivm.address_space<ub>>)
      %19 = hivm.hir.pointer_cast(%c64_i64) : memref<8xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg10 = %c0 to %c8 step %c1 {
        %30 = memref.load %13[%arg10] {DiscreteMemAccess} : memref<8xi32, #hivm.address_space<ub>>
        %31 = arith.index_cast %30 : i32 to index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%31], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %32 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.store %32, %19[%arg10] : memref<8xi32, #hivm.address_space<ub>>
      } {ExtractedLoadOrStore}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %20 = hivm.hir.pointer_cast(%c96_i64) : memref<8xi32, #hivm.address_space<ub>>
      %21 = hivm.hir.pointer_cast(%c128_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%18, %19, %c-1_i32 : memref<8xi1, #hivm.address_space<ub>>, memref<8xi32, #hivm.address_space<ub>>, i32) outs(%20 : memref<8xi32, #hivm.address_space<ub>>) temp_buffer(%21 : memref<16xi32, #hivm.address_space<ub>>)
      %22 = hivm.hir.pointer_cast(%c192_i64) : memref<8xi32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%15 : i32) outs(%22 : memref<8xi32, #hivm.address_space<ub>>)
      %23 = hivm.hir.pointer_cast(%c192_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%20, %22 : memref<8xi32, #hivm.address_space<ub>>, memref<8xi32, #hivm.address_space<ub>>) outs(%23 : memref<8xi1, #hivm.address_space<ub>>)
      %24 = hivm.hir.pointer_cast(%c224_i64) : memref<8xi32, #hivm.address_space<ub>>
      %25 = hivm.hir.pointer_cast(%c256_i64) : memref<24xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%23, %c1_i32, %c0_i32 : memref<8xi1, #hivm.address_space<ub>>, i32, i32) outs(%24 : memref<8xi32, #hivm.address_space<ub>>) temp_buffer(%25 : memref<24xi32, #hivm.address_space<ub>>)
      %26 = hivm.hir.pointer_cast(%c352_i64) : memref<1xi32, #hivm.address_space<ub>>
      %27 = hivm.hir.pointer_cast(%c384_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%24 : memref<8xi32, #hivm.address_space<ub>>) outs(%26 : memref<1xi32, #hivm.address_space<ub>>) temp_buffer(%27 : memref<16xi32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %28 = memref.load %26[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %11]
      memref.store %28, %12[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %29 = arith.index_cast %15 : i32 to index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%29], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%12 : memref<1xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast_1 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
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
