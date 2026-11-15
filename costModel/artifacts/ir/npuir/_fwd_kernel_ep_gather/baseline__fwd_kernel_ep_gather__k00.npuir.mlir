// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_fwd_kernel_ep_gather(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: i32, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, false, true, false, true, false, true, false, true, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c6144_i64 = arith.constant 6144 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c2_i32 = arith.constant 2 : i32
  %c0_i32 = arith.constant 0 : i32
  %c512_i32 = arith.constant 512 : i32
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg16, %arg15 : i32
  %5 = hivm.hir.pointer_cast(%c0_i64) : memref<512xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%5 : memref<512xf32, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg17 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %6 = arith.divsi %arg17, %arg16 : i32
    %7 = arith.remsi %6, %arg15 : i32
    %8 = arith.divsi %arg17, %4 : i32
    %9 = arith.remsi %8, %arg14 : i32
    %10 = arith.muli %9, %c512_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    scf.for %arg18 = %7 to %arg3 step %arg15  : i32 {
      %12 = arith.index_cast %arg18 : i32 to index
      %13 = arith.index_cast %7 : i32 to index
      %14 = arith.index_cast %arg3 : i32 to index
      %15 = arith.index_cast %arg15 : i32 to index
      %16 = arith.index_cast %arg17 : i32 to index
      %17 = arith.index_cast %3 : i32 to index
      %18 = arith.index_cast %1 : i32 to index
      %19 = arith.index_cast %c48_i32 : i32 to index
      %20 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%12, %13, %14, %15, %16, %17, %19]
      %21 = arith.index_cast %20 : index to i1
      %c0_i64_0 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %22 = arith.select %21, %c0_i64_0, %c1_i64 : i64
      %23 = hivm.hir.pointer_cast(%c4096_i64, %c8192_i64) : memref<512xf32, #hivm.address_space<ub>>
      annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
      %24 = arith.muli %arg18, %arg7 : i32
      %25 = arith.index_cast %24 : i32 to index
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %22]
      hivm.hir.copy ins(%5 : memref<512xf32, #hivm.address_space<ub>>) outs(%23 : memref<512xf32, #hivm.address_space<ub>>)
      %26 = scf.for %arg19 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg20 = %23) -> (memref<512xf32, #hivm.address_space<ub>>)  : i32 {
        %30 = arith.index_cast %arg19 : i32 to index
        %31 = arith.index_cast %c0_i32 : i32 to index
        %32 = arith.index_cast %c2_i32 : i32 to index
        %33 = arith.index_cast %c1_i32 : i32 to index
        %34 = arith.index_cast %arg18 : i32 to index
        %35 = arith.index_cast %7 : i32 to index
        %36 = arith.index_cast %arg3 : i32 to index
        %37 = arith.index_cast %arg15 : i32 to index
        %38 = arith.index_cast %arg17 : i32 to index
        %39 = arith.index_cast %3 : i32 to index
        %40 = arith.index_cast %1 : i32 to index
        %41 = arith.index_cast %c48_i32 : i32 to index
        %42 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s7) * ((-s1 + s2 + s3 - 1) floordiv s3) + ((s8 - s9) floordiv s10) * (((-s1 + s2 + s3 - 1) floordiv s3) * ((-s5 + s6 + s7 - 1) floordiv s7))) mod 2)>()[%30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %41]
        %43 = arith.index_cast %42 : index to i1
        %c0_i64_1 = arith.constant 0 : i64
        %c1_i64_2 = arith.constant 1 : i64
        %44 = arith.select %43, %c0_i64_1, %c1_i64_2 : i64
        %45 = hivm.hir.pointer_cast(%c2048_i64, %c6144_i64) : memref<512xf32, #hivm.address_space<ub>>
        annotation.mark %45 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
        %46 = arith.index_cast %arg19 : i32 to index
        %47 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%25, %46]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%47], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %48 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %49 = arith.cmpi sge, %48, %c0_i64 : i64
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %44]
        %50 = scf.if %49 -> (memref<512xf32, #hivm.address_space<ub>>) {
          %51 = arith.muli %arg18, %arg11 : i32
          %52 = arith.index_cast %51 : i32 to index
          %53 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%52, %46]
          %reinterpret_cast_4 = memref.reinterpret_cast %arg10 to offset: [%53], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %54 = memref.load %reinterpret_cast_4[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %55 = arith.muli %arg18, %arg9 : i32
          %56 = arith.index_cast %55 : i32 to index
          %57 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%56, %46]
          %reinterpret_cast_5 = memref.reinterpret_cast %arg8 to offset: [%57], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %58 = memref.load %reinterpret_cast_5[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %59 = arith.extsi %arg5 : i32 to i64
          %60 = arith.muli %54, %59 : i64
          %61 = arith.index_cast %60 : i64 to index
          %62 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%61, %11]
          %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%62], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.load ins(%reinterpret_cast_6 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%45 : memref<512xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.vmul ins(%45, %58 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%45 : memref<512xf32, #hivm.address_space<ub>>)
          scf.yield %45 : memref<512xf32, #hivm.address_space<ub>>
        } else {
          scf.yield %5 : memref<512xf32, #hivm.address_space<ub>>
        }
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vadd ins(%arg20, %50 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%23 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %44]
        scf.yield %23 : memref<512xf32, #hivm.address_space<ub>>
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %27 = arith.muli %arg18, %arg13 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%28, %11]
      %reinterpret_cast = memref.reinterpret_cast %arg12 to offset: [%29], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%26 : memref<512xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %22]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
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
