// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @moe_align_block_size_stage1(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c32_i64 = arith.constant 32 : i64
  %c64_i64 = arith.constant 64 : i64
  %c0_i64 = arith.constant 0 : i64
  %c4_i32 = arith.constant 4 : i32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg10 = %3 to %1 step %c48_i32  : i32 {
    %5 = hivm.hir.pointer_cast(%c0_i64, %c64_i64) : memref<4xi32, #hivm.address_space<ub>>
    annotation.mark %5 {hivm.multi_buffer = 2 : i32} : memref<4xi32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %6 = arith.divsi %arg10, %4 : i32
    %7 = arith.remsi %6, %arg7 : i32
    %8 = arith.muli %7, %arg6 : i32
    %9 = arith.addi %7, %c1_i32 : i32
    %10 = arith.muli %9, %c4_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%11], sizes: [4], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<4xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%reinterpret_cast : memref<4xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%5 : memref<4xi32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    scf.for %arg11 = %c0_i32 to %arg6 step %c1_i32  : i32 {
      %12 = arith.addi %8, %arg11 : i32
      %13 = arith.cmpi slt, %12, %arg5 : i32
      scf.if %13 {
        %14 = arith.index_cast %8 : i32 to index
        %15 = arith.index_cast %arg11 : i32 to index
        %16 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%14, %15]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %17 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %18 = hivm.hir.pointer_cast(%c32_i64) : memref<4xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%5, %c1_i32 : memref<4xi32, #hivm.address_space<ub>>, i32) outs(%18 : memref<4xi32, #hivm.address_space<ub>>)
        %19 = arith.index_cast %17 : i32 to index
        %20 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%19]
        %21 = arith.cmpi slt, %19, %c0 : index
        %22 = arith.cmpi sge, %19, %c4 : index
        %23 = arith.ori %21, %22 : i1
        scf.if %23 {
        } else {
          %subview = memref.subview %5[0] [%19] [1] : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          %subview_1 = memref.subview %18[0] [%19] [1] : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%subview : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_1 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>)
          %subview_2 = memref.subview %18[0] [%20] [1] : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          %subview_3 = memref.subview %5[0] [%20] [1] : memref<4xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          hivm.hir.copy ins(%subview_2 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>>)
        }
      }
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%5 : memref<4xi32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<4xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
