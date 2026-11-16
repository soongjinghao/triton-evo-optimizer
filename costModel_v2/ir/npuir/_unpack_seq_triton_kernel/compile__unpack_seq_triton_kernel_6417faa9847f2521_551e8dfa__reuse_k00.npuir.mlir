// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_unpack_seq_triton_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c262144_i32 = arith.constant 262144 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  scf.for %arg9 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %5 = arith.divsi %arg9, %4 : i32
    %6 = arith.remsi %5, %arg6 : i32
    %7 = arith.cmpi sge, %6, %c32_i32 : i32
    scf.if %7 {
    } else {
      %8 = arith.index_cast %6 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%8], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %9 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %10 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%8]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%10], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %11 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %12 = arith.subi %11, %9 : i32
      %13 = arith.cmpi eq, %12, %c0_i32 : i32
      scf.if %13 {
      } else {
        %14 = arith.muli %6, %c262144_i32 : i32
        %15 = arith.index_cast %14 : i32 to index
        %16 = arith.index_cast %12 : i32 to index
        %17 = arith.index_cast %9 : i32 to index
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        scf.for %arg10 = %c0_i32 to %12 step %c128_i32  : i32 {
          %18 = arith.index_cast %arg10 : i32 to index
          %19 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%15, %18]
          %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%19], sizes: [128, 64], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<128x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %20 = hivm.hir.pointer_cast(%c0_i64) : memref<128x64xf32, #hivm.address_space<ub>>
          %21 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%18, %16]
          %22 = affine.min affine_map<()[s0, s1] -> (s1 + 128, s0)>()[%21, %18]
          %23 = affine.min affine_map<()[s0, s1] -> (128, s0 - s1)>()[%22, %18]
          %24 = affine.max affine_map<()[s0] -> (0, s0)>()[%23]
          %25 = arith.cmpi slt, %24, %c128 : index
          %subview = memref.subview %reinterpret_cast_1[0, 0] [%24, 64] [1, 1] : memref<128x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_2 = memref.subview %20[0, 0] [%24, 64] [1, 1] : memref<128x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>>
          scf.if %25 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape_7 = memref.collapse_shape %20 [[0, 1]] : memref<128x64xf32, #hivm.address_space<ub>> into memref<8192xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_7 : memref<8192xf32, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_3 = memref.collapse_shape %subview_2 [[0, 1]] : memref<?x64xf32, strided<[64, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.load ins(%collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          %26 = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 * 64)>()[%17, %18]
          %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [128, 64], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<128x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_5 = memref.subview %reinterpret_cast_4[0, 0] [%24, 64] [1, 1] : memref<128x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1]] : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.store ins(%collapse_shape_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        }
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      }
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
