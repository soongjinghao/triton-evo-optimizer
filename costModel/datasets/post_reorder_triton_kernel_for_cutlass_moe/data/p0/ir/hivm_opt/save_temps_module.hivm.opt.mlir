#map = affine_map<()[s0, s1, s2, s3] -> ((s0 floordiv 128 + ((s2 - s3) floordiv 48) * ((s1 + 127) floordiv 128)) mod 2)>
#map1 = affine_map<()[s0, s1, s2, s3, s4, s5] -> ((s0 + (s2 floordiv 128) * s1 + ((s4 - s5) floordiv 48) * (s1 * ((s3 + 127) floordiv 128))) mod 2)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1)>
#map3 = affine_map<()[s0, s1] -> (s1, s0)>
#map4 = affine_map<()[s0, s1] -> (s1 + 128, s0)>
#map5 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @post_reorder_triton_kernel_for_cutlass_moe_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_half_to_1d(f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @post_reorder_triton_kernel_for_cutlass_moe(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant 0.000000e+00 : f16
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c768_i64 = arith.constant 768 : i64
    %c256_i64 = arith.constant 256 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %0 = hivm.hir.pointer_cast(%c512_i64) : memref<128xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c256_i64) : memref<128xf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c768_i64) : memref<128xf16, #hivm.address_space<ub>>
    %4 = arith.muli %arg11, %arg12 : i32
    %5 = arith.muli %4, %arg13 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg13, %arg12 : i32
    %9 = hivm.hir.pointer_cast(%c0_i64) : memref<128xf16, #hivm.address_space<ub>>
    %cast = memref.cast %9 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_half_to_1d(%cst, %cast) : (f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %10 = arith.extsi %arg8 : i32 to i64
    %11 = arith.extsi %arg10 : i32 to i64
    %12 = arith.index_cast %arg10 : i32 to index
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    scf.for %arg14 = %7 to %5 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %13 = arith.divsi %arg14, %8 : i32
      %14 = arith.remsi %13, %arg11 : i32
      %15 = arith.extsi %14 : i32 to i64
      %16 = arith.muli %15, %10 : i64
      %17 = arith.muli %15, %11 : i64
      %18 = arith.index_cast %16 : i64 to index
      %19 = arith.index_cast %17 : i64 to index
      scf.for %arg15 = %c0_i32 to %arg10 step %c128_i32  : i32 {
        %20 = arith.index_cast %arg15 : i32 to index
        %21 = arith.index_cast %arg10 : i32 to index
        %22 = arith.index_cast %arg14 : i32 to index
        %23 = arith.index_cast %7 : i32 to index
        %24 = affine.apply #map()[%20, %21, %22, %23]
        %25 = arith.index_cast %24 : index to i64
        %26 = arith.cmpi eq, %25, %c1_i64 : i64
        %27 = arith.select %26, %1, %0 : memref<128xf16, #hivm.address_space<ub>>
        %28 = arith.index_cast %arg15 : i32 to index
        %29 = arith.index_cast %arg10 : i32 to index
        %30 = arith.index_cast %arg14 : i32 to index
        %31 = arith.index_cast %7 : i32 to index
        %32 = affine.apply #map()[%28, %29, %30, %31]
        %33 = arith.index_cast %32 : index to i1
        %34 = arith.xori %33, %true : i1
        %35 = arith.extui %34 : i1 to i64
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %35]
        %cast_0 = memref.cast %9 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_1 = memref.cast %27 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @copy_ubuf_to_ubuf_1d_half(%cast_0, %cast_1, %c0_i32, %cst) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16) -> ()
        %36 = scf.for %arg16 = %c0_i32 to %arg8 step %c1_i32 iter_args(%arg17 = %27) -> (memref<128xf16, #hivm.address_space<ub>>)  : i32 {
          %42 = arith.index_cast %arg16 : i32 to index
          %43 = arith.index_cast %arg8 : i32 to index
          %44 = arith.index_cast %arg15 : i32 to index
          %45 = arith.index_cast %arg10 : i32 to index
          %46 = arith.index_cast %arg14 : i32 to index
          %47 = arith.index_cast %7 : i32 to index
          %48 = affine.apply #map1()[%42, %43, %44, %45, %46, %47]
          %49 = arith.index_cast %48 : index to i64
          %50 = arith.cmpi eq, %49, %c1_i64 : i64
          %51 = arith.select %50, %3, %2 : memref<128xf16, #hivm.address_space<ub>>
          %52 = arith.index_cast %arg16 : i32 to index
          %53 = arith.index_cast %arg8 : i32 to index
          %54 = arith.index_cast %arg15 : i32 to index
          %55 = arith.index_cast %arg10 : i32 to index
          %56 = arith.index_cast %arg14 : i32 to index
          %57 = arith.index_cast %7 : i32 to index
          %58 = affine.apply #map1()[%52, %53, %54, %55, %56, %57]
          %59 = arith.index_cast %58 : index to i1
          %60 = arith.select %59, %c1_i64, %c2_i64 : i64
          %61 = arith.index_cast %arg16 : i32 to index
          %62 = affine.apply #map2()[%18, %61]
          %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%62], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %63 = memref.load %reinterpret_cast_5[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %64 = arith.cmpi ne, %63, %arg9 : i32
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %60]
          %65 = arith.select %64, %51, %9 : memref<128xf16, #hivm.address_space<ub>>
          scf.if %64 {
            %reinterpret_cast_10 = memref.reinterpret_cast %arg5 to offset: [%62], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %67 = memref.load %reinterpret_cast_10[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %68 = arith.extsi %67 : i32 to i64
            %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [%62], sizes: [1], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %69 = memref.load %reinterpret_cast_11[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %70 = arith.muli %68, %11 : i64
            %71 = arith.index_cast %70 : i64 to index
            %72 = arith.index_cast %arg15 : i32 to index
            %73 = affine.apply #map2()[%71, %72]
            %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%73], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %74 = affine.max #map3()[%72, %12]
            %75 = affine.min #map4()[%74, %72]
            %76 = affine.apply #map5()[%75, %72]
            %77 = arith.cmpi slt, %76, %c128 : index
            %subview_13 = memref.subview %reinterpret_cast_12[0] [%76] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_14 = memref.subview %51[0] [%76] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
            scf.if %77 {
              hivm.hir.pipe_barrier[<PIPE_V>]
              %cast_20 = memref.cast %51 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
              func.call @broadcast_scalar_half_to_1d(%cst, %cast_20) : (f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
              hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            } {hivm.unlikely_condition}
            %cast_15 = memref.cast %subview_13 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %cast_16 = memref.cast %subview_14 : memref<?xf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @load_gm_to_ubuf_1d_half(%cast_15, %cast_16, %c2_i32, %cst, %c0) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f16, index) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            %78 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf16, #hivm.address_space<ub>>
            %cast_17 = memref.cast %51 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_18 = memref.cast %51 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_19 = memref.cast %78 : memref<0xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmuls_vs_1d_half(%cast_17, %69, %cast_18, %cast_19) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          }
          hivm.hir.pipe_barrier[<PIPE_V>]
          %66 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf16, #hivm.address_space<ub>>
          %cast_6 = memref.cast %arg17 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_7 = memref.cast %65 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_8 = memref.cast %27 : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_9 = memref.cast %66 : memref<0xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vadd_1d_half(%cast_6, %cast_7, %cast_8, %cast_9) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %60]
          scf.yield %27 : memref<128xf16, #hivm.address_space<ub>>
        } {pipeline.veconly}
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %37 = arith.index_cast %arg15 : i32 to index
        %38 = affine.apply #map2()[%19, %37]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%38], sizes: [128], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %39 = affine.max #map3()[%37, %12]
        %40 = affine.min #map4()[%39, %37]
        %41 = affine.apply #map5()[%40, %37]
        %subview = memref.subview %36[0] [%41] [1] : memref<128xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[1]>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %reinterpret_cast[0] [%41] [1] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_3 = memref.cast %subview : memref<?xf16, strided<[1]>, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_4 = memref.cast %subview_2 : memref<?xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_half(%cast_3, %cast_4, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %35]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
