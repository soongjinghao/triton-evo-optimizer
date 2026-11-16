// -----// IR Dump After PlanMemory Failed (hivm-plan-memory) //----- //
func.func @_trtllm_prefill_attn_kvfp8_dequant(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1_i64 = arith.constant 1 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg11, %arg10 : i32
  %5 = arith.extsi %arg5 : i32 to i64
  scf.for %arg12 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %6 = arith.divsi %arg12, %arg11 : i32
    %7 = arith.remsi %6, %arg10 : i32
    %8 = arith.divsi %arg12, %4 : i32
    %9 = arith.remsi %8, %arg9 : i32
    %10 = arith.extsi %9 : i32 to i64
    %11 = arith.extsi %7 : i32 to i64
    %12 = arith.muli %10, %5 : i64
    %13 = arith.addi %12, %11 : i64
    %14 = arith.index_cast %13 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %15 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %16 = arith.extsi %15 : i32 to i64
    %17 = arith.cmpi sle, %16, %c0_i64 : i64
    scf.if %17 {
    } else {
      %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %18 = memref.load %reinterpret_cast_0[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %19 = memref.load %reinterpret_cast_1[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %20 = arith.muli %16, %c16384_i64 : i64
      %21 = arith.index_cast %20 : i64 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%21], sizes: [16384], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<16384xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %alloc = memref.alloc() : memref<16384xi8, #hivm.address_space<ub>>
      annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<16384xi8, #hivm.address_space<ub>>
      hivm.hir.load ins(%reinterpret_cast_2 : memref<16384xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16384xi8, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%18 : f32) outs(%alloc_3 : memref<16384xf32, #hivm.address_space<ub>>)
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%19 : f32) outs(%alloc_4 : memref<16384xf32, #hivm.address_space<ub>>)
      %subview = memref.subview %alloc_3[0] [8192] [1] : memref<16384xf32, #hivm.address_space<ub>> to memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_5 = memref.subview %alloc_4[0] [8192] [1] : memref<16384xf32, #hivm.address_space<ub>> to memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview : memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>)
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<16384xf16, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc : memref<16384xi8, #hivm.address_space<ub>>) outs(%alloc_6 : memref<16384xf16, #hivm.address_space<ub>>) cast = <cast_unsigned>
      %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_6 : memref<16384xf16, #hivm.address_space<ub>>) outs(%alloc_7 : memref<16384xf32, #hivm.address_space<ub>>)
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%alloc_7, %alloc_4 : memref<16384xf32, #hivm.address_space<ub>>, memref<16384xf32, #hivm.address_space<ub>>) outs(%alloc_8 : memref<16384xf32, #hivm.address_space<ub>>)
      %22 = arith.addi %13, %c1_i64 : i64
      %23 = arith.muli %22, %c16384_i64 : i64
      %24 = arith.index_cast %23 : i64 to index
      %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [16384], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<16384xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<16384xbf16, #hivm.address_space<ub>>
      annotation.mark %alloc_10 {hivm.multi_buffer = 2 : i32} : memref<16384xbf16, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_8 : memref<16384xf32, #hivm.address_space<ub>>) outs(%alloc_10 : memref<16384xbf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%alloc_10 : memref<16384xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_9 : memref<16384xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    } {pipeline.veconly}
  } {autoblockify.subloop}
  return
}

[NOTE] Ub overflow detected; automatically set enable-code-motion to false and retrying compilation.
// -----// IR Dump After PlanMemory Failed (hivm-plan-memory) //----- //
func.func @_trtllm_prefill_attn_kvfp8_dequant(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c1_i64 = arith.constant 1 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  scf.for %arg12 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %4 = arith.divsi %arg12, %arg11 : i32
    %5 = arith.remsi %4, %arg10 : i32
    %6 = arith.muli %arg11, %arg10 : i32
    %7 = arith.divsi %arg12, %6 : i32
    %8 = arith.remsi %7, %arg9 : i32
    %9 = arith.extsi %8 : i32 to i64
    %10 = arith.extsi %5 : i32 to i64
    %11 = arith.extsi %arg5 : i32 to i64
    %12 = arith.muli %9, %11 : i64
    %13 = arith.addi %12, %10 : i64
    %14 = arith.index_cast %13 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %15 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %16 = arith.extsi %15 : i32 to i64
    %17 = arith.cmpi sle, %16, %c0_i64 : i64
    scf.if %17 {
    } else {
      %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %18 = memref.load %reinterpret_cast_0[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %19 = memref.load %reinterpret_cast_1[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %20 = arith.muli %16, %c16384_i64 : i64
      %21 = arith.index_cast %20 : i64 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%21], sizes: [16384], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<16384xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %alloc = memref.alloc() : memref<16384xi8, #hivm.address_space<ub>>
      annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<16384xi8, #hivm.address_space<ub>>
      hivm.hir.load ins(%reinterpret_cast_2 : memref<16384xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<16384xi8, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%18 : f32) outs(%alloc_3 : memref<16384xf32, #hivm.address_space<ub>>)
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%19 : f32) outs(%alloc_4 : memref<16384xf32, #hivm.address_space<ub>>)
      %subview = memref.subview %alloc_3[0] [8192] [1] : memref<16384xf32, #hivm.address_space<ub>> to memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_5 = memref.subview %alloc_4[0] [8192] [1] : memref<16384xf32, #hivm.address_space<ub>> to memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.copy ins(%subview : memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_5 : memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>)
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<16384xf16, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc : memref<16384xi8, #hivm.address_space<ub>>) outs(%alloc_6 : memref<16384xf16, #hivm.address_space<ub>>) cast = <cast_unsigned>
      %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_6 : memref<16384xf16, #hivm.address_space<ub>>) outs(%alloc_7 : memref<16384xf32, #hivm.address_space<ub>>)
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%alloc_7, %alloc_4 : memref<16384xf32, #hivm.address_space<ub>>, memref<16384xf32, #hivm.address_space<ub>>) outs(%alloc_8 : memref<16384xf32, #hivm.address_space<ub>>)
      %22 = arith.addi %13, %c1_i64 : i64
      %23 = arith.muli %22, %c16384_i64 : i64
      %24 = arith.index_cast %23 : i64 to index
      %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [16384], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<16384xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<16384xbf16, #hivm.address_space<ub>>
      annotation.mark %alloc_10 {hivm.multi_buffer = 2 : i32} : memref<16384xbf16, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_8 : memref<16384xf32, #hivm.address_space<ub>>) outs(%alloc_10 : memref<16384xbf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%alloc_10 : memref<16384xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_9 : memref<16384xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    } {pipeline.veconly}
  } {autoblockify.subloop}
  return
}

[NOTE] Ub overflow detected; automatically set enable-auto-multi-buffer to false and retrying compilation.
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_trtllm_prefill_attn_kvfp8_dequant(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c147456_i64 = arith.constant 147456 : i64
  %c81920_i64 = arith.constant 81920 : i64
  %c1_i64 = arith.constant 1 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg9, %arg10 : i32
  %1 = arith.muli %0, %arg11 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  scf.for %arg12 = %3 to %1 step %c48_i32  : i32 {
    hivm.hir.set_mask_norm
    %4 = arith.divsi %arg12, %arg11 : i32
    %5 = arith.remsi %4, %arg10 : i32
    %6 = arith.muli %arg11, %arg10 : i32
    %7 = arith.divsi %arg12, %6 : i32
    %8 = arith.remsi %7, %arg9 : i32
    %9 = arith.extsi %8 : i32 to i64
    %10 = arith.extsi %5 : i32 to i64
    %11 = arith.extsi %arg5 : i32 to i64
    %12 = arith.muli %9, %11 : i64
    %13 = arith.addi %12, %10 : i64
    %14 = arith.index_cast %13 : i64 to index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%14], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %15 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %16 = arith.extsi %15 : i32 to i64
    %17 = arith.cmpi sle, %16, %c0_i64 : i64
    scf.if %17 {
    } else {
      %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %18 = memref.load %reinterpret_cast_0[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %19 = memref.load %reinterpret_cast_1[%c0] : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
      %20 = arith.muli %16, %c16384_i64 : i64
      %21 = arith.index_cast %20 : i64 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%21], sizes: [16384], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<16384xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %22 = hivm.hir.pointer_cast(%c0_i64) : memref<16384xi8, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%reinterpret_cast_2 : memref<16384xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%22 : memref<16384xi8, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %23 = hivm.hir.pointer_cast(%c16384_i64) : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vbrc ins(%18 : f32) outs(%23 : memref<16384xf32, #hivm.address_space<ub>>)
      %24 = hivm.hir.pointer_cast(%c81920_i64) : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%19 : f32) outs(%24 : memref<16384xf32, #hivm.address_space<ub>>)
      %subview = memref.subview %23[0] [8192] [1] : memref<16384xf32, #hivm.address_space<ub>> to memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %24[0] [8192] [1] : memref<16384xf32, #hivm.address_space<ub>> to memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.copy ins(%subview : memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<8192xf32, strided<[1]>, #hivm.address_space<ub>>)
      %25 = hivm.hir.pointer_cast(%c147456_i64) : memref<16384xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vcast ins(%22 : memref<16384xi8, #hivm.address_space<ub>>) outs(%25 : memref<16384xf16, #hivm.address_space<ub>>) cast = <cast_unsigned>
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      %26 = hivm.hir.pointer_cast(%c16384_i64) : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%25 : memref<16384xf16, #hivm.address_space<ub>>) outs(%26 : memref<16384xf32, #hivm.address_space<ub>>)
      %27 = hivm.hir.pointer_cast(%c16384_i64) : memref<16384xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%26, %24 : memref<16384xf32, #hivm.address_space<ub>>, memref<16384xf32, #hivm.address_space<ub>>) outs(%27 : memref<16384xf32, #hivm.address_space<ub>>)
      %28 = arith.addi %13, %c1_i64 : i64
      %29 = arith.muli %28, %c16384_i64 : i64
      %30 = arith.index_cast %29 : i64 to index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%30], sizes: [16384], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<16384xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %31 = hivm.hir.pointer_cast(%c16384_i64) : memref<16384xbf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%27 : memref<16384xf32, #hivm.address_space<ub>>) outs(%31 : memref<16384xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%31 : memref<16384xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_4 : memref<16384xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    } {pipeline.veconly}
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

[NOTE] Due to Ub overflow, automatically set enable-code-motion to false, set enable-auto-multi-buffer to false; compilation then succeeded.
bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
