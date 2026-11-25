#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0] -> (s0 + 1)>
#map2 = affine_map<()[s0] -> (s0 + 2)>
#map3 = affine_map<()[s0] -> (s0 + 3)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_count_expert_num_tokens_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @load_gm_to_ubuf_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmaxs_vs_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vs_1d_int32_t(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_int32_t_to_1d(i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_ss_1d_int32_t(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_sum_r_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_count_expert_num_tokens(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant 4 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i64 = arith.constant 32 : i64
    %c64_i64 = arith.constant 64 : i64
    %c96_i64 = arith.constant 96 : i64
    %c128_i64 = arith.constant 128 : i64
    %c192_i64 = arith.constant 192 : i64
    %c224_i64 = arith.constant 224 : i64
    %c256_i64 = arith.constant 256 : i64
    %c352_i64 = arith.constant 352 : i64
    %c384_i64 = arith.constant 384 : i64
    %c416_i64 = arith.constant 416 : i64
    %c512_i64 = arith.constant 512 : i64
    %c544_i64 = arith.constant 544 : i64
    %c576_i64 = arith.constant 576 : i64
    %c672_i64 = arith.constant 672 : i64
    %c704_i64 = arith.constant 704 : i64
    %c736_i64 = arith.constant 736 : i64
    %c832_i64 = arith.constant 832 : i64
    %c864_i64 = arith.constant 864 : i64
    %c960_i64 = arith.constant 960 : i64
    %c992_i64 = arith.constant 992 : i64
    %c1088_i64 = arith.constant 1088 : i64
    %c1120_i64 = arith.constant 1120 : i64
    %c1216_i64 = arith.constant 1216 : i64
    %c1248_i64 = arith.constant 1248 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1344_i64 = arith.constant 1344 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1376_i64 = arith.constant 1376 : i64
    %c928_i64 = arith.constant 928 : i64
    %c1408_i64 = arith.constant 1408 : i64
    %c1056_i64 = arith.constant 1056 : i64
    %c1440_i64 = arith.constant 1440 : i64
    %c1184_i64 = arith.constant 1184 : i64
    %c1312_i64 = arith.constant 1312 : i64
    %c1472_i64 = arith.constant 1472 : i64
    %0 = hivm.hir.pointer_cast(%c1312_i64) : memref<1xi32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c1472_i64) : memref<1xi32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c1184_i64) : memref<1xi32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c1440_i64) : memref<1xi32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c1056_i64) : memref<1xi32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c1408_i64) : memref<1xi32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c928_i64) : memref<1xi32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c1376_i64) : memref<1xi32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c0_i64) : memref<8xi32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c1344_i64) : memref<8xi32, #hivm.address_space<ub>>
    %10 = arith.muli %arg6, %arg7 : i32
    %11 = arith.muli %10, %arg8 : i32
    %12 = hivm.hir.get_block_idx -> i64
    %13 = arith.trunci %12 : i64 to i32
    %14 = arith.muli %arg8, %arg7 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<8xi32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID6>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID7>]
    scf.for %arg9 = %13 to %11 step %c48_i32  : i32 {
      %15 = arith.index_cast %arg9 : i32 to index
      %16 = arith.index_cast %13 : i32 to index
      %17 = affine.apply #map()[%15, %16]
      %18 = arith.index_cast %17 : index to i64
      %19 = arith.cmpi eq, %18, %c1_i64 : i64
      %20 = arith.select %19, %1, %0 : memref<1xi32, #hivm.address_space<ub>>
      %21 = arith.index_cast %arg9 : i32 to index
      %22 = arith.index_cast %13 : i32 to index
      %23 = affine.apply #map()[%21, %22]
      %24 = arith.index_cast %23 : index to i64
      %25 = arith.cmpi eq, %24, %c1_i64 : i64
      %26 = arith.select %25, %3, %2 : memref<1xi32, #hivm.address_space<ub>>
      %27 = arith.index_cast %arg9 : i32 to index
      %28 = arith.index_cast %13 : i32 to index
      %29 = affine.apply #map()[%27, %28]
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.cmpi eq, %30, %c1_i64 : i64
      %32 = arith.select %31, %5, %4 : memref<1xi32, #hivm.address_space<ub>>
      %33 = arith.index_cast %arg9 : i32 to index
      %34 = arith.index_cast %13 : i32 to index
      %35 = affine.apply #map()[%33, %34]
      %36 = arith.index_cast %35 : index to i64
      %37 = arith.cmpi eq, %36, %c1_i64 : i64
      %38 = arith.select %37, %7, %6 : memref<1xi32, #hivm.address_space<ub>>
      %39 = arith.index_cast %arg9 : i32 to index
      %40 = arith.index_cast %13 : i32 to index
      %41 = affine.apply #map()[%39, %40]
      %42 = arith.index_cast %41 : index to i64
      %43 = arith.cmpi eq, %42, %c1_i64 : i64
      %44 = arith.select %43, %9, %8 : memref<8xi32, #hivm.address_space<ub>>
      %45 = arith.index_cast %arg9 : i32 to index
      %46 = arith.index_cast %13 : i32 to index
      %47 = affine.apply #map()[%45, %46]
      %48 = arith.index_cast %47 : index to i1
      %49 = arith.select %48, %c2_i64, %c3_i64 : i64
      %50 = arith.index_cast %47 : index to i1
      %51 = arith.select %50, %c4_i64, %c5_i64 : i64
      %52 = arith.index_cast %47 : index to i1
      %53 = arith.select %52, %c6_i64, %c7_i64 : i64
      %54 = arith.index_cast %47 : index to i1
      %55 = arith.xori %54, %true : i1
      %56 = arith.extui %55 : i1 to i64
      hivm.hir.set_mask_norm
      %57 = arith.divsi %arg9, %14 : i32
      %58 = arith.remsi %57, %arg6 : i32
      %59 = arith.muli %58, %c4_i32 : i32
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %56]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, %56]
      %cast = memref.cast %reinterpret_cast : memref<8xi32, strided<[1]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_0 = memref.cast %44 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_int32_t(%cast, %cast_0, %c0_i32, %c0_i32, %c0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %60 = hivm.hir.pointer_cast(%c32_i64) : memref<8xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %61 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_1 = memref.cast %44 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_2 = memref.cast %60 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_3 = memref.cast %61 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmaxs_vs_1d_int32_t(%cast_1, %c0_i32, %cast_2, %cast_3) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %62 = hivm.hir.pointer_cast(%c32_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_4 = memref.cast %60 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_5 = memref.cast %44 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_6 = memref.cast %62 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_4, %cast_5, %cast_6) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %56]
      %63 = hivm.hir.pointer_cast(%c64_i64) : memref<8xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg10 = %c0 to %c8 step %c1 {
        %89 = memref.load %44[%arg10] {DiscreteMemAccess} : memref<8xi32, #hivm.address_space<ub>>
        %90 = arith.index_cast %89 : i32 to index
        %reinterpret_cast_39 = memref.reinterpret_cast %arg5 to offset: [%90], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %91 = memref.load %reinterpret_cast_39[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.store %91, %63[%arg10] : memref<8xi32, #hivm.address_space<ub>>
      } {ExtractedLoadOrStore}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, %56]
      %64 = hivm.hir.pointer_cast(%c96_i64) : memref<8xi32, #hivm.address_space<ub>>
      %65 = hivm.hir.pointer_cast(%c128_i64) : memref<16xi32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_7 = memref.cast %62 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_8 = memref.cast %63 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_9 = memref.cast %64 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_10 = memref.cast %65 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_int32_t(%cast_7, %cast_8, %c-1_i32, %cast_9, %cast_10) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %66 = hivm.hir.pointer_cast(%c192_i64) : memref<8xi32, #hivm.address_space<ub>>
      %cast_11 = memref.cast %66 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_int32_t_to_1d(%59, %cast_11) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %67 = hivm.hir.pointer_cast(%c192_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_12 = memref.cast %64 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %66 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %67 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_12, %cast_13, %cast_14) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %68 = hivm.hir.pointer_cast(%c224_i64) : memref<8xi32, #hivm.address_space<ub>>
      %69 = hivm.hir.pointer_cast(%c256_i64) : memref<24xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_15 = memref.cast %67 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_16 = memref.cast %68 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_17 = memref.cast %69 : memref<24xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_ss_1d_int32_t(%cast_15, %c1_i32, %c0_i32, %cast_16, %cast_17) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %70 = arith.addi %59, %c1_i32 : i32
      %71 = hivm.hir.pointer_cast(%c352_i64) : memref<8xi32, #hivm.address_space<ub>>
      %cast_18 = memref.cast %71 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_int32_t_to_1d(%70, %cast_18) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %72 = hivm.hir.pointer_cast(%c352_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_19 = memref.cast %64 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %71 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_21 = memref.cast %72 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_19, %cast_20, %cast_21) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %73 = hivm.hir.pointer_cast(%c384_i64) : memref<8xi32, #hivm.address_space<ub>>
      %74 = hivm.hir.pointer_cast(%c416_i64) : memref<24xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_22 = memref.cast %72 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_23 = memref.cast %73 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_24 = memref.cast %74 : memref<24xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_ss_1d_int32_t(%cast_22, %c1_i32, %c0_i32, %cast_23, %cast_24) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %75 = arith.addi %59, %c2_i32 : i32
      %76 = hivm.hir.pointer_cast(%c512_i64) : memref<8xi32, #hivm.address_space<ub>>
      %cast_25 = memref.cast %76 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_int32_t_to_1d(%75, %cast_25) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %77 = hivm.hir.pointer_cast(%c512_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_26 = memref.cast %64 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_27 = memref.cast %76 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_28 = memref.cast %77 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_26, %cast_27, %cast_28) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %78 = hivm.hir.pointer_cast(%c544_i64) : memref<8xi32, #hivm.address_space<ub>>
      %79 = hivm.hir.pointer_cast(%c576_i64) : memref<24xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_29 = memref.cast %77 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_30 = memref.cast %78 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_31 = memref.cast %79 : memref<24xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_ss_1d_int32_t(%cast_29, %c1_i32, %c0_i32, %cast_30, %cast_31) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %80 = arith.addi %59, %c3_i32 : i32
      %81 = hivm.hir.pointer_cast(%c672_i64) : memref<8xi32, #hivm.address_space<ub>>
      %cast_32 = memref.cast %81 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_int32_t_to_1d(%80, %cast_32) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %82 = hivm.hir.pointer_cast(%c96_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_33 = memref.cast %64 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_34 = memref.cast %81 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_35 = memref.cast %82 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_33, %cast_34, %cast_35) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %83 = hivm.hir.pointer_cast(%c704_i64) : memref<8xi32, #hivm.address_space<ub>>
      %84 = hivm.hir.pointer_cast(%c736_i64) : memref<24xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_36 = memref.cast %82 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_37 = memref.cast %83 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_38 = memref.cast %84 : memref<24xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_ss_1d_int32_t(%cast_36, %c1_i32, %c0_i32, %cast_37, %cast_38) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %85 = arith.cmpi slt, %59, %c4_i32 : i32
      scf.if %85 {
        %89 = hivm.hir.pointer_cast(%c832_i64) : memref<1xi32, #hivm.address_space<ub>>
        %90 = hivm.hir.pointer_cast(%c864_i64) : memref<16xi32, #hivm.address_space<ub>>
        %cast_39 = memref.cast %68 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_40 = memref.cast %89 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_41 = memref.cast %90 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_r_int32_t(%cast_39, %cast_40, %cast_41, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
        %91 = memref.load %89[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %53]
        memref.store %91, %38[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %92 = arith.index_cast %59 : i32 to index
        %reinterpret_cast_42 = memref.reinterpret_cast %arg4 to offset: [%92], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_43 = memref.cast %38 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_44 = memref.cast %reinterpret_cast_42 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_43, %cast_44, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %53]
      } {pipeline.veconly}
      %86 = arith.cmpi slt, %70, %c4_i32 : i32
      scf.if %86 {
        %89 = hivm.hir.pointer_cast(%c960_i64) : memref<1xi32, #hivm.address_space<ub>>
        %90 = hivm.hir.pointer_cast(%c992_i64) : memref<16xi32, #hivm.address_space<ub>>
        %cast_39 = memref.cast %73 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_40 = memref.cast %89 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_41 = memref.cast %90 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_r_int32_t(%cast_39, %cast_40, %cast_41, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
        %91 = memref.load %89[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %51]
        memref.store %91, %32[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %92 = arith.index_cast %59 : i32 to index
        %93 = affine.apply #map1()[%92]
        %reinterpret_cast_42 = memref.reinterpret_cast %arg4 to offset: [%93], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_43 = memref.cast %32 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_44 = memref.cast %reinterpret_cast_42 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_43, %cast_44, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %51]
      } {pipeline.veconly}
      %87 = arith.cmpi slt, %75, %c4_i32 : i32
      scf.if %87 {
        %89 = hivm.hir.pointer_cast(%c1088_i64) : memref<1xi32, #hivm.address_space<ub>>
        %90 = hivm.hir.pointer_cast(%c1120_i64) : memref<16xi32, #hivm.address_space<ub>>
        %cast_39 = memref.cast %78 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_40 = memref.cast %89 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_41 = memref.cast %90 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_r_int32_t(%cast_39, %cast_40, %cast_41, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
        %91 = memref.load %89[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %49]
        memref.store %91, %26[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %92 = arith.index_cast %59 : i32 to index
        %93 = affine.apply #map2()[%92]
        %reinterpret_cast_42 = memref.reinterpret_cast %arg4 to offset: [%93], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_43 = memref.cast %26 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_44 = memref.cast %reinterpret_cast_42 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_43, %cast_44, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %49]
      } {pipeline.veconly}
      %88 = arith.cmpi slt, %80, %c4_i32 : i32
      scf.if %88 {
        %89 = hivm.hir.pointer_cast(%c1216_i64) : memref<1xi32, #hivm.address_space<ub>>
        %90 = hivm.hir.pointer_cast(%c1248_i64) : memref<16xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_39 = memref.cast %83 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_40 = memref.cast %89 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_41 = memref.cast %90 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_r_int32_t(%cast_39, %cast_40, %cast_41, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
        %91 = memref.load %89[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %56]
        memref.store %91, %20[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %92 = arith.index_cast %59 : i32 to index
        %93 = affine.apply #map3()[%92]
        %reinterpret_cast_42 = memref.reinterpret_cast %arg4 to offset: [%93], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_43 = memref.cast %20 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_44 = memref.cast %reinterpret_cast_42 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_43, %cast_44, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %56]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID6>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID7>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
