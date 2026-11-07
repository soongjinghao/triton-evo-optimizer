// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_fwd_kernel_ep_gather(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: i32, %arg12: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, false, true, false, true, false, true, false, true, false, true, false, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c6144_i64 = arith.constant 6144 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c512_i32 = arith.constant 512 : i32
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst = arith.constant 0.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg14, %arg15 : i32
  %1 = arith.muli %0, %arg16 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.divsi %3, %arg16 : i32
  %5 = arith.remsi %4, %arg15 : i32
  %6 = arith.muli %arg16, %arg15 : i32
  %7 = arith.divsi %3, %6 : i32
  %8 = arith.remsi %7, %arg14 : i32
  %9 = hivm.hir.pointer_cast(%c0_i64) : memref<512xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%9 : memref<512xf32, #hivm.address_space<ub>>)
  %10 = arith.muli %8, %c512_i32 : i32
  %11 = arith.index_cast %10 : i32 to index
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  scf.for %arg17 = %5 to %arg3 step %arg15  : i32 {
    %12 = arith.index_cast %arg17 : i32 to index
    %13 = arith.index_cast %5 : i32 to index
    %14 = arith.index_cast %arg3 : i32 to index
    %15 = arith.index_cast %arg15 : i32 to index
    %16 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%12, %13, %15]
    %17 = arith.index_cast %16 : index to i1
    %c0_i64_0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %18 = arith.select %17, %c0_i64_0, %c1_i64 : i64
    %19 = hivm.hir.pointer_cast(%c4096_i64, %c8192_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %20 = arith.muli %arg17, %arg7 : i32
    %21 = arith.index_cast %20 : i32 to index
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %18]
    hivm.hir.copy ins(%9 : memref<512xf32, #hivm.address_space<ub>>) outs(%19 : memref<512xf32, #hivm.address_space<ub>>)
    %22 = scf.for %arg18 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg19 = %19) -> (memref<512xf32, #hivm.address_space<ub>>)  : i32 {
      %26 = arith.index_cast %arg18 : i32 to index
      %27 = arith.index_cast %c0_i32 : i32 to index
      %28 = arith.index_cast %c2_i32 : i32 to index
      %29 = arith.index_cast %c1_i32 : i32 to index
      %30 = arith.index_cast %arg17 : i32 to index
      %31 = arith.index_cast %5 : i32 to index
      %32 = arith.index_cast %arg3 : i32 to index
      %33 = arith.index_cast %arg15 : i32 to index
      %34 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%26, %27, %28, %29, %30, %31, %33]
      %35 = arith.index_cast %34 : index to i1
      %c0_i64_1 = arith.constant 0 : i64
      %c1_i64_2 = arith.constant 1 : i64
      %36 = arith.select %35, %c0_i64_1, %c1_i64_2 : i64
      %37 = hivm.hir.pointer_cast(%c2048_i64, %c6144_i64) : memref<512xf32, #hivm.address_space<ub>>
      annotation.mark %37 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
      %38 = arith.index_cast %arg18 : i32 to index
      %39 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%21, %38]
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%39], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %40 = memref.load %reinterpret_cast_3[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %41 = arith.cmpi sge, %40, %c0_i64 : i64
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %36]
      %42 = scf.if %41 -> (memref<512xf32, #hivm.address_space<ub>>) {
        %43 = arith.muli %arg17, %arg11 : i32
        %44 = arith.index_cast %43 : i32 to index
        %45 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%44, %38]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg10 to offset: [%45], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %46 = memref.load %reinterpret_cast_4[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %47 = arith.muli %arg17, %arg9 : i32
        %48 = arith.index_cast %47 : i32 to index
        %49 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%48, %38]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg8 to offset: [%49], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %50 = memref.load %reinterpret_cast_5[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %51 = arith.extsi %arg5 : i32 to i64
        %52 = arith.muli %46, %51 : i64
        %53 = arith.index_cast %52 : i64 to index
        %54 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%53, %11]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.load ins(%reinterpret_cast_6 : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%37 : memref<512xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%37, %50 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%37 : memref<512xf32, #hivm.address_space<ub>>)
        scf.yield %37 : memref<512xf32, #hivm.address_space<ub>>
      } else {
        scf.yield %9 : memref<512xf32, #hivm.address_space<ub>>
      }
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%arg19, %42 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%19 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %36]
      scf.yield %19 : memref<512xf32, #hivm.address_space<ub>>
    }
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %23 = arith.muli %arg17, %arg13 : i32
    %24 = arith.index_cast %23 : i32 to index
    %25 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%24, %11]
    %reinterpret_cast = memref.reinterpret_cast %arg12 to offset: [%25], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%22 : memref<512xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %18]
  }
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
