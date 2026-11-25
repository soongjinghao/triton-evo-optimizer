#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_trtllm_prefill_attn_kvfp8_dequant_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @load_gm_to_ubuf_1d_int8_t(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i8, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_uint8_t_to_half_1d_with_mode(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_half_to_float_1d_with_mode(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_half_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_trtllm_prefill_attn_kvfp8_dequant(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi8, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i8 = arith.constant 0 : i8
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c0 = arith.constant 0 : index
    %c16384_i64 = arith.constant 16384 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c114688_i64 = arith.constant 114688 : i64
    %c0_i64 = arith.constant 0 : i64
    %c122880_i64 = arith.constant 122880 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c155648_i64 = arith.constant 155648 : i64
    %c57344_i64 = arith.constant 57344 : i64
    %c81920_i64 = arith.constant 81920 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %0 = hivm.hir.pointer_cast(%c81920_i64) : memref<8192xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c163840_i64) : memref<8192xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c57344_i64) : memref<8192xi8, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c155648_i64) : memref<8192xi8, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c24576_i64) : memref<8192xf16, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c122880_i64) : memref<8192xf16, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c0_i64) : memref<8192xi8, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c114688_i64) : memref<8192xi8, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c24576_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c122880_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c24576_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c122880_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c81920_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c163840_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c81920_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c163840_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %16 = arith.muli %arg9, %arg10 : i32
    %17 = arith.muli %16, %arg11 : i32
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.trunci %18 : i64 to i32
    %20 = arith.muli %arg11, %arg10 : i32
    %21 = arith.extsi %arg5 : i32 to i64
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    scf.for %arg12 = %19 to %17 step %c48_i32  : i32 {
      %22 = arith.index_cast %arg12 : i32 to index
      %23 = arith.index_cast %19 : i32 to index
      %24 = affine.apply #map()[%22, %23]
      %25 = arith.index_cast %24 : index to i64
      %26 = arith.cmpi eq, %25, %c1_i64 : i64
      %27 = arith.select %26, %1, %0 : memref<8192xf16, #hivm.address_space<ub>>
      %28 = arith.index_cast %arg12 : i32 to index
      %29 = arith.index_cast %19 : i32 to index
      %30 = affine.apply #map()[%28, %29]
      %31 = arith.index_cast %30 : index to i64
      %32 = arith.cmpi eq, %31, %c1_i64 : i64
      %33 = arith.select %32, %3, %2 : memref<8192xi8, #hivm.address_space<ub>>
      %34 = arith.index_cast %arg12 : i32 to index
      %35 = arith.index_cast %19 : i32 to index
      %36 = affine.apply #map()[%34, %35]
      %37 = arith.index_cast %36 : index to i64
      %38 = arith.cmpi eq, %37, %c1_i64 : i64
      %39 = arith.select %38, %5, %4 : memref<8192xf16, #hivm.address_space<ub>>
      %40 = arith.index_cast %arg12 : i32 to index
      %41 = arith.index_cast %19 : i32 to index
      %42 = affine.apply #map()[%40, %41]
      %43 = arith.index_cast %42 : index to i64
      %44 = arith.cmpi eq, %43, %c1_i64 : i64
      %45 = arith.select %44, %7, %6 : memref<8192xi8, #hivm.address_space<ub>>
      %46 = arith.index_cast %arg12 : i32 to index
      %47 = arith.index_cast %19 : i32 to index
      %48 = affine.apply #map()[%46, %47]
      %49 = arith.index_cast %48 : index to i64
      %50 = arith.cmpi eq, %49, %c1_i64 : i64
      %51 = arith.select %50, %9, %8 : memref<8192xf32, #hivm.address_space<ub>>
      %52 = arith.index_cast %arg12 : i32 to index
      %53 = arith.index_cast %19 : i32 to index
      %54 = affine.apply #map()[%52, %53]
      %55 = arith.index_cast %54 : index to i64
      %56 = arith.cmpi eq, %55, %c1_i64 : i64
      %57 = arith.select %56, %11, %10 : memref<8192xf32, #hivm.address_space<ub>>
      %58 = arith.index_cast %arg12 : i32 to index
      %59 = arith.index_cast %19 : i32 to index
      %60 = affine.apply #map()[%58, %59]
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.cmpi eq, %61, %c1_i64 : i64
      %63 = arith.select %62, %13, %12 : memref<8192xf32, #hivm.address_space<ub>>
      %64 = arith.index_cast %arg12 : i32 to index
      %65 = arith.index_cast %19 : i32 to index
      %66 = affine.apply #map()[%64, %65]
      %67 = arith.index_cast %66 : index to i64
      %68 = arith.cmpi eq, %67, %c1_i64 : i64
      %69 = arith.select %68, %15, %14 : memref<8192xf32, #hivm.address_space<ub>>
      %70 = arith.index_cast %arg12 : i32 to index
      %71 = arith.index_cast %19 : i32 to index
      %72 = affine.apply #map()[%70, %71]
      %73 = arith.index_cast %72 : index to i1
      %74 = arith.xori %73, %true : i1
      %75 = arith.extui %74 : i1 to i64
      %76 = arith.index_cast %72 : index to i1
      %77 = arith.select %76, %c2_i64, %c3_i64 : i64
      hivm.hir.set_mask_norm
      %78 = arith.divsi %arg12, %arg11 : i32
      %79 = arith.remsi %78, %arg10 : i32
      %80 = arith.divsi %arg12, %20 : i32
      %81 = arith.remsi %80, %arg9 : i32
      %82 = arith.extsi %81 : i32 to i64
      %83 = arith.extsi %79 : i32 to i64
      %84 = arith.muli %82, %21 : i64
      %85 = arith.addi %84, %83 : i64
      %86 = arith.index_cast %85 : i64 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%86], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %87 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %88 = arith.extsi %87 : i32 to i64
      %89 = arith.cmpi sle, %88, %c0_i64 : i64
      scf.if %89 {
      } else {
        %reinterpret_cast_0 = memref.reinterpret_cast %arg7 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
        %90 = memref.load %reinterpret_cast_0[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg8 to offset: [0], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
        %91 = memref.load %reinterpret_cast_1[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1]>, #hivm.address_space<gm>>
        %92 = arith.muli %88, %c16384_i64 : i64
        %93 = arith.addi %85, %c1_i64 : i64
        %94 = arith.muli %93, %c16384_i64 : i64
        %95 = arith.index_cast %92 : i64 to index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%95], sizes: [8192], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<8192xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %77]
        %cast = memref.cast %reinterpret_cast_2 : memref<8192xi8, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_3 = memref.cast %45 : memref<8192xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_int8_t(%cast, %cast_3, %c0_i32, %c0_i8, %c0) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i8, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %96 = hivm.hir.pointer_cast(%c8192_i64) : memref<8192xf16, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_4 = memref.cast %45 : memref<8192xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_5 = memref.cast %96 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_uint8_t_to_half_1d_with_mode(%cast_4, %cast_5, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %77]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %77]
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_6 = memref.cast %96 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_7 = memref.cast %51 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_half_to_float_1d_with_mode(%cast_6, %cast_7, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %97 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_8 = memref.cast %51 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_9 = memref.cast %57 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_10 = memref.cast %97 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_8, %90, %cast_9, %cast_10) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %98 = arith.index_cast %94 : i64 to index
        %reinterpret_cast_11 = memref.reinterpret_cast %arg6 to offset: [%98], sizes: [8192], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_12 = memref.cast %57 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_13 = memref.cast %39 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_float_to_half_1d_with_mode(%cast_12, %cast_13, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_14 = memref.cast %39 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_15 = memref.cast %reinterpret_cast_11 : memref<8192xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_half(%cast_14, %cast_15, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %77]
        %99 = arith.addi %92, %c8192_i64 : i64
        %100 = arith.index_cast %99 : i64 to index
        %reinterpret_cast_16 = memref.reinterpret_cast %arg3 to offset: [%100], sizes: [8192], strides: [1] : memref<?xi8, #hivm.address_space<gm>> to memref<8192xi8, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %75]
        %cast_17 = memref.cast %reinterpret_cast_16 : memref<8192xi8, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_18 = memref.cast %33 : memref<8192xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_int8_t(%cast_17, %cast_18, %c0_i32, %c0_i8, %c0) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i8, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %101 = hivm.hir.pointer_cast(%c65536_i64) : memref<8192xf16, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %cast_19 = memref.cast %33 : memref<8192xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_20 = memref.cast %101 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_uint8_t_to_half_1d_with_mode(%cast_19, %cast_20, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %75]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %75]
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_21 = memref.cast %101 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_22 = memref.cast %63 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_half_to_float_1d_with_mode(%cast_21, %cast_22, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.pipe_barrier[<PIPE_V>]
        %102 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_23 = memref.cast %63 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_24 = memref.cast %69 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_25 = memref.cast %102 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_23, %91, %cast_24, %cast_25) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %103 = arith.addi %94, %c8192_i64 : i64
        %104 = arith.index_cast %103 : i64 to index
        %reinterpret_cast_26 = memref.reinterpret_cast %arg6 to offset: [%104], sizes: [8192], strides: [1] : memref<?xf16, #hivm.address_space<gm>> to memref<8192xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_27 = memref.cast %69 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_28 = memref.cast %27 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_float_to_half_1d_with_mode(%cast_27, %cast_28, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_29 = memref.cast %27 : memref<8192xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_30 = memref.cast %reinterpret_cast_26 : memref<8192xf16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_half(%cast_29, %cast_30, %c0_i32) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %75]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
