#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0] -> (s0 + 1)>
#map2 = affine_map<()[s0] -> (0, s0)>
#map3 = affine_map<()[s0] -> (32, s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @moe_align_block_size_stage4_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_int32_t_to_1d(i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @moe_align_block_size_stage4(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c160_i64 = arith.constant 160 : i64
    %c288_i64 = arith.constant 288 : i64
    %c416_i64 = arith.constant 416 : i64
    %c544_i64 = arith.constant 544 : i64
    %c672_i64 = arith.constant 672 : i64
    %c704_i64 = arith.constant 704 : i64
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c736_i64 = arith.constant 736 : i64
    %0 = hivm.hir.pointer_cast(%c128_i64) : memref<1xi32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c736_i64) : memref<1xi32, #hivm.address_space<ub>>
    %2 = arith.muli %arg10, %arg11 : i32
    %3 = arith.muli %2, %arg12 : i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.trunci %4 : i64 to i32
    %6 = arith.muli %arg12, %arg11 : i32
    %7 = hivm.hir.pointer_cast(%c0_i64) : memref<32xi32, #hivm.address_space<ub>>
    %cast = memref.cast %7 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_int32_t_to_1d(%c0_i32, %cast) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    scf.for %arg13 = %5 to %3 step %c48_i32  : i32 {
      %8 = arith.index_cast %arg13 : i32 to index
      %9 = arith.index_cast %5 : i32 to index
      %10 = affine.apply #map()[%8, %9]
      %11 = arith.index_cast %10 : index to i64
      %12 = arith.cmpi eq, %11, %c1_i64 : i64
      %13 = arith.select %12, %1, %0 : memref<1xi32, #hivm.address_space<ub>>
      %14 = arith.index_cast %arg13 : i32 to index
      %15 = arith.index_cast %5 : i32 to index
      %16 = affine.apply #map()[%14, %15]
      %17 = arith.index_cast %16 : index to i1
      %18 = arith.xori %17, %true : i1
      %19 = arith.extui %18 : i1 to i64
      hivm.hir.set_mask_norm
      %20 = arith.divsi %arg13, %6 : i32
      %21 = arith.remsi %20, %arg10 : i32
      %22 = arith.index_cast %21 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%22], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %23 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %24 = affine.apply #map1()[%22]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [%24], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %25 = memref.load %reinterpret_cast_0[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %19]
      memref.store %21, %13[%c0] : memref<1xi32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.for %arg14 = %23 to %25 step %c32_i32  : i32 {
        %32 = arith.divsi %arg14, %c32_i32 : i32
        %33 = arith.index_cast %32 : i32 to index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_2 = memref.cast %13 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_3 = memref.cast %reinterpret_cast_1 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_2, %cast_3, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      } {pipeline.veconly}
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %19]
      %26 = arith.muli %21, %arg9 : i32
      %27 = arith.muli %21, %c4_i32 : i32
      %28 = arith.addi %26, %arg9 : i32
      %29 = arith.minsi %28, %arg8 : i32
      %30 = arith.extsi %27 : i32 to i64
      %31 = scf.while (%arg14 = %26) : (i32) -> i32 {
        %32 = arith.cmpi slt, %arg14, %29 : i32
        scf.condition(%32) %arg14 : i32
      } do {
      ^bb0(%arg14: i32):
        %32 = arith.subi %29, %arg14 : i32
        %33 = arith.minsi %32, %c32_i32 : i32
        %34 = arith.index_cast %arg14 : i32 to index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%34], sizes: [32], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<32xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %35 = hivm.hir.pointer_cast(%c160_i64) : memref<32xi32, #hivm.address_space<ub>>
        %36 = arith.index_cast %33 : i32 to index
        %37 = affine.max #map2()[%36]
        %38 = affine.min #map3()[%37]
        %39 = arith.cmpi slt, %38, %c32 : index
        %subview = memref.subview %reinterpret_cast_1[0] [%38] [1] : memref<32xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_2 = memref.subview %35[0] [%38] [1] : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %39 {
          hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %cast_11 = memref.cast %35 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_int32_t_to_1d(%c0_i32, %cast_11) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_3 = memref.cast %subview : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_4 = memref.cast %subview_2 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_int32_t(%cast_3, %cast_4, %c2_i32, %c0_i32, %c0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
        %40 = hivm.hir.pointer_cast(%c288_i64) : memref<32xi32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
        scf.for %arg15 = %c0 to %38 step %c1 {
          %44 = memref.load %35[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
          %45 = arith.extsi %44 : i32 to i64
          %46 = arith.addi %30, %45 : i64
          %47 = arith.index_cast %46 : i64 to index
          %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%47], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %48 = memref.load %reinterpret_cast_11[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          memref.store %48, %40[%arg15] : memref<32xi32, #hivm.address_space<ub>>
        } {ExtractedLoadOrStore}
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %subview_5 = memref.subview %40[0] [%38] [1] : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
        %41 = hivm.hir.pointer_cast(%c416_i64) : memref<32xi32, #hivm.address_space<ub>>
        %cast_6 = memref.cast %7 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %41 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @copy_ubuf_to_ubuf_1d_int32_t(%cast_6, %cast_7, %c0_i32, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32) -> ()
        %subview_8 = memref.subview %41[0] [%38] [1] : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_9 = memref.cast %subview_5 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_10 = memref.cast %subview_8 : memref<?xi32, strided<[1]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @copy_ubuf_to_ubuf_1d_int32_t(%cast_9, %cast_10, %c0_i32, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %42 = hivm.hir.pointer_cast(%c544_i64) : memref<32xi32, #hivm.address_space<ub>>
        scf.for %arg15 = %c0 to %c32 step %c1 {
          %44 = memref.load %35[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
          %45 = arith.index_cast %44 : i32 to index
          %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [%45], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %46 = memref.load %reinterpret_cast_11[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          memref.store %46, %42[%arg15] : memref<32xi32, #hivm.address_space<ub>>
        } {ExtractedLoadOrStore}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
        scf.for %arg15 = %c0 to %38 step %c1 {
          %44 = memref.load %41[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
          %45 = memref.load %42[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
          %46 = arith.addi %44, %45 : i32
          %47 = arith.index_cast %arg15 : index to i32
          %48 = arith.addi %arg14, %47 : i32
          %49 = arith.cmpi slt, %47, %33 : i32
          scf.if %49 {
            %50 = hivm.hir.pointer_cast(%c672_i64) : memref<1xi32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
            memref.store %48, %50[%c0] : memref<1xi32, #hivm.address_space<ub>>
            hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            %51 = arith.index_cast %46 : i32 to index
            %reinterpret_cast_11 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            %cast_12 = memref.cast %50 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_13 = memref.cast %reinterpret_cast_11 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            func.call @store_ubuf_to_gm_1d_int32_t(%cast_12, %cast_13, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
          } {pipeline.veconly}
        } {ExtractedLoadOrStore}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
        scf.for %arg15 = %c0 to %38 step %c1 {
          %44 = memref.load %35[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
          %45 = arith.extsi %44 : i32 to i64
          %46 = arith.addi %30, %45 : i64
          %47 = memref.load %41[%arg15] {DiscreteMemAccess} : memref<32xi32, #hivm.address_space<ub>>
          %48 = arith.addi %47, %c1_i32 : i32
          %49 = arith.index_cast %arg15 : index to i32
          %50 = arith.cmpi slt, %49, %33 : i32
          scf.if %50 {
            %51 = hivm.hir.pointer_cast(%c704_i64) : memref<1xi32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
            memref.store %48, %51[%c0] : memref<1xi32, #hivm.address_space<ub>>
            hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            %52 = arith.index_cast %46 : i64 to index
            %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%52], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            %cast_12 = memref.cast %51 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_13 = memref.cast %reinterpret_cast_11 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            func.call @store_ubuf_to_gm_1d_int32_t(%cast_12, %cast_13, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
          } {pipeline.veconly}
        } {ExtractedLoadOrStore}
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
        %43 = arith.addi %arg14, %33 : i32
        scf.yield %43 : i32
      } attributes {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
