#map = affine_map<()[s0] -> (s0 mod 2)>
#map1 = affine_map<()[s0] -> (s0 + 64)>
#map2 = affine_map<()[s0] -> (s0 + 128)>
#map3 = affine_map<()[s0] -> (s0 + 256)>
#map4 = affine_map<()[s0] -> (s0 + 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @fused_qkvzba_split_reshape_cat_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @fused_qkvzba_split_reshape_cat_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i64 = arith.constant 16 : i64
    %c32_i64 = arith.constant 32 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c3072_i64 = arith.constant 3072 : i64
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %c384_i32 = arith.constant 384 : i32
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c32_i32 = arith.constant 32 : i32
    %c6144_i32 = arith.constant 6144 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %c1600_i64 = arith.constant 1600 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1856_i64 = arith.constant 1856 : i64
    %c256_i64 = arith.constant 256 : i64
    %c2112_i64 = arith.constant 2112 : i64
    %c512_i64 = arith.constant 512 : i64
    %c2624_i64 = arith.constant 2624 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c3136_i64 = arith.constant 3136 : i64
    %c1536_i64 = arith.constant 1536 : i64
    %c1568_i64 = arith.constant 1568 : i64
    %c3168_i64 = arith.constant 3168 : i64
    %0 = hivm.hir.pointer_cast(%c1568_i64) : memref<2xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c3168_i64) : memref<2xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c1536_i64) : memref<2xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c3136_i64) : memref<2xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c1024_i64) : memref<128xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c2624_i64) : memref<128xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c512_i64) : memref<128xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c2112_i64) : memref<128xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c256_i64) : memref<64xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c1856_i64) : memref<64xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c0_i64) : memref<64xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c1600_i64) : memref<64xf32, #hivm.address_space<ub>>
    %12 = arith.muli %arg9, %arg10 : i32
    %13 = arith.muli %12, %arg11 : i32
    %14 = hivm.hir.get_block_idx -> i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = arith.muli %arg11, %arg10 : i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID6>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID7>]
    scf.for %arg12 = %15 to %13 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %17 = arith.divsi %arg12, %16 : i32
      %18 = arith.remsi %17, %arg9 : i32
      %19 = arith.divsi %18, %c8_i32 : i32
      %20 = arith.remsi %18, %c8_i32 : i32
      %21 = arith.muli %19, %c2_i32 : i32
      %22 = arith.muli %19, %c6144_i32 : i32
      %23 = arith.extsi %22 : i32 to i64
      %24 = arith.muli %20, %c384_i32 : i32
      %25 = arith.extsi %24 : i32 to i64
      %26 = arith.addi %23, %25 : i64
      %27 = arith.muli %19, %c4096_i32 : i32
      %28 = arith.extsi %27 : i32 to i64
      %29 = arith.muli %20, %c64_i32 : i32
      %30 = arith.extsi %29 : i32 to i64
      %31 = arith.addi %28, %30 : i64
      %32 = arith.addi %28, %c512_i64 : i64
      %33 = arith.addi %32, %30 : i64
      %34 = arith.addi %28, %c1024_i64 : i64
      %35 = arith.muli %20, %c128_i32 : i32
      %36 = arith.extsi %35 : i32 to i64
      %37 = arith.addi %34, %36 : i64
      %38 = arith.muli %19, %c2048_i32 : i32
      %39 = arith.extsi %38 : i32 to i64
      %40 = arith.addi %39, %36 : i64
      %41 = arith.muli %19, %c64_i32 : i32
      %42 = arith.extsi %41 : i32 to i64
      %43 = arith.muli %20, %c4_i32 : i32
      %44 = arith.extsi %43 : i32 to i64
      %45 = arith.addi %42, %44 : i64
      %46 = arith.muli %19, %c32_i32 : i32
      %47 = arith.extsi %46 : i32 to i64
      %48 = arith.muli %20, %c2_i32 : i32
      %49 = arith.extsi %48 : i32 to i64
      %50 = arith.addi %47, %49 : i64
      %51:8 = scf.for %arg13 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg14 = %26, %arg15 = %31, %arg16 = %33, %arg17 = %37, %arg18 = %40, %arg19 = %45, %arg20 = %50, %arg21 = %50) -> (i64, i64, i64, i64, i64, i64, i64, i64)  : i32 {
        %52 = arith.index_cast %arg13 : i32 to index
        %53 = affine.apply #map()[%52]
        %54 = arith.index_cast %53 : index to i64
        %55 = arith.cmpi eq, %54, %c1_i64 : i64
        %56 = arith.select %55, %1, %0 : memref<2xf32, #hivm.address_space<ub>>
        %57 = arith.index_cast %arg13 : i32 to index
        %58 = affine.apply #map()[%57]
        %59 = arith.index_cast %58 : index to i64
        %60 = arith.cmpi eq, %59, %c1_i64 : i64
        %61 = arith.select %60, %3, %2 : memref<2xf32, #hivm.address_space<ub>>
        %62 = arith.index_cast %arg13 : i32 to index
        %63 = affine.apply #map()[%62]
        %64 = arith.index_cast %63 : index to i64
        %65 = arith.cmpi eq, %64, %c1_i64 : i64
        %66 = arith.select %65, %5, %4 : memref<128xf32, #hivm.address_space<ub>>
        %67 = arith.index_cast %arg13 : i32 to index
        %68 = affine.apply #map()[%67]
        %69 = arith.index_cast %68 : index to i64
        %70 = arith.cmpi eq, %69, %c1_i64 : i64
        %71 = arith.select %70, %7, %6 : memref<128xf32, #hivm.address_space<ub>>
        %72 = arith.index_cast %arg13 : i32 to index
        %73 = affine.apply #map()[%72]
        %74 = arith.index_cast %73 : index to i64
        %75 = arith.cmpi eq, %74, %c1_i64 : i64
        %76 = arith.select %75, %9, %8 : memref<64xf32, #hivm.address_space<ub>>
        %77 = arith.index_cast %arg13 : i32 to index
        %78 = affine.apply #map()[%77]
        %79 = arith.index_cast %78 : index to i64
        %80 = arith.cmpi eq, %79, %c1_i64 : i64
        %81 = arith.select %80, %11, %10 : memref<64xf32, #hivm.address_space<ub>>
        %82 = arith.index_cast %arg13 : i32 to index
        %83 = affine.apply #map()[%82]
        %84 = arith.index_cast %83 : index to i1
        %85 = arith.xori %84, %true : i1
        %86 = arith.extui %85 : i1 to i64
        %87 = arith.index_cast %83 : index to i1
        %88 = arith.select %87, %c2_i64, %c3_i64 : i64
        %89 = arith.index_cast %83 : index to i1
        %90 = arith.select %89, %c4_i64, %c5_i64 : i64
        %91 = arith.index_cast %83 : index to i1
        %92 = arith.select %91, %c6_i64, %c7_i64 : i64
        %93 = arith.addi %21, %arg13 : i32
        %94 = arith.cmpi slt, %93, %c4_i32 : i32
        scf.if %94 {
          %103 = arith.index_cast %arg14 : i64 to index
          %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%103], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %92]
          %cast = memref.cast %reinterpret_cast : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_0 = memref.cast %81 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast, %cast_0, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
          %104 = affine.apply #map1()[%103]
          %reinterpret_cast_1 = memref.reinterpret_cast %arg7 to offset: [%104], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %cast_2 = memref.cast %reinterpret_cast_1 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_3 = memref.cast %76 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_2, %cast_3, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
          %105 = affine.apply #map2()[%103]
          %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%105], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %cast_5 = memref.cast %reinterpret_cast_4 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_6 = memref.cast %71 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_5, %cast_6, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
          %106 = affine.apply #map3()[%103]
          %reinterpret_cast_7 = memref.reinterpret_cast %arg7 to offset: [%106], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %90]
          %cast_8 = memref.cast %reinterpret_cast_7 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_9 = memref.cast %66 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_8, %cast_9, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          %107 = arith.index_cast %arg15 : i64 to index
          %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%107], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID3>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_11 = memref.cast %81 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_12 = memref.cast %reinterpret_cast_10 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_11, %cast_12, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          %108 = arith.index_cast %arg16 : i64 to index
          %reinterpret_cast_13 = memref.reinterpret_cast %arg3 to offset: [%108], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID2>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_14 = memref.cast %76 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_15 = memref.cast %reinterpret_cast_13 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_14, %cast_15, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          %109 = arith.index_cast %arg17 : i64 to index
          %reinterpret_cast_16 = memref.reinterpret_cast %arg3 to offset: [%109], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          %cast_17 = memref.cast %71 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_18 = memref.cast %reinterpret_cast_16 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_17, %cast_18, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %92]
          %110 = arith.index_cast %arg18 : i64 to index
          %reinterpret_cast_19 = memref.reinterpret_cast %arg4 to offset: [%110], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_20 = memref.cast %66 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_21 = memref.cast %reinterpret_cast_19 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_20, %cast_21, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %90]
          %111 = arith.index_cast %arg19 : i64 to index
          %reinterpret_cast_22 = memref.reinterpret_cast %arg8 to offset: [%111], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %88]
          %cast_23 = memref.cast %reinterpret_cast_22 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_24 = memref.cast %61 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_23, %cast_24, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
          %112 = affine.apply #map4()[%111]
          %reinterpret_cast_25 = memref.reinterpret_cast %arg8 to offset: [%112], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %86]
          %cast_26 = memref.cast %reinterpret_cast_25 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_27 = memref.cast %56 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_26, %cast_27, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          %113 = arith.index_cast %arg20 : i64 to index
          %reinterpret_cast_28 = memref.reinterpret_cast %arg5 to offset: [%113], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
          %cast_29 = memref.cast %61 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_30 = memref.cast %reinterpret_cast_28 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_29, %cast_30, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %88]
          %114 = arith.index_cast %arg21 : i64 to index
          %reinterpret_cast_31 = memref.reinterpret_cast %arg6 to offset: [%114], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_32 = memref.cast %56 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_33 = memref.cast %reinterpret_cast_31 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_32, %cast_33, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %86]
        } {pipeline.veconly}
        %95 = arith.addi %arg14, %c3072_i64 : i64
        %96 = arith.addi %arg15, %c2048_i64 : i64
        %97 = arith.addi %arg16, %c2048_i64 : i64
        %98 = arith.addi %arg17, %c2048_i64 : i64
        %99 = arith.addi %arg18, %c1024_i64 : i64
        %100 = arith.addi %arg19, %c32_i64 : i64
        %101 = arith.addi %arg20, %c16_i64 : i64
        %102 = arith.addi %arg21, %c16_i64 : i64
        scf.yield %95, %96, %97, %98, %99, %100, %101, %102 : i64, i64, i64, i64, i64, i64, i64, i64
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID6>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID7>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
