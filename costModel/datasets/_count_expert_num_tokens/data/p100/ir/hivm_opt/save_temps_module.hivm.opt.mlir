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
  func.func private @broadcast_scalar_int32_t_to_1d(i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_ss_1d_int32_t(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_sum_r_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_count_expert_num_tokens(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c32_i64 = arith.constant 32 : i64
    %c64_i64 = arith.constant 64 : i64
    %c96_i64 = arith.constant 96 : i64
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
    %c800_i64 = arith.constant 800 : i64
    %c832_i64 = arith.constant 832 : i64
    %c928_i64 = arith.constant 928 : i64
    %c960_i64 = arith.constant 960 : i64
    %c1056_i64 = arith.constant 1056 : i64
    %c1088_i64 = arith.constant 1088 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1184_i64 = arith.constant 1184 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1216_i64 = arith.constant 1216 : i64
    %c768_i64 = arith.constant 768 : i64
    %c1248_i64 = arith.constant 1248 : i64
    %c896_i64 = arith.constant 896 : i64
    %c1280_i64 = arith.constant 1280 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c1152_i64 = arith.constant 1152 : i64
    %c1312_i64 = arith.constant 1312 : i64
    %0 = hivm.hir.pointer_cast(%c1152_i64) : memref<1xi32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c1312_i64) : memref<1xi32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c1024_i64) : memref<1xi32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c1280_i64) : memref<1xi32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c896_i64) : memref<1xi32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c1248_i64) : memref<1xi32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c768_i64) : memref<1xi32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c1216_i64) : memref<1xi32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c0_i64) : memref<8xi32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c1184_i64) : memref<8xi32, #hivm.address_space<ub>>
    %10 = arith.muli %arg5, %arg6 : i32
    %11 = arith.muli %10, %arg7 : i32
    %12 = hivm.hir.get_block_idx -> i64
    %13 = arith.trunci %12 : i64 to i32
    %14 = arith.muli %arg7, %arg6 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [8], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<8xi32, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID6>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID7>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID3>]
    scf.for %arg8 = %13 to %11 step %c48_i32  : i32 {
      %15 = arith.index_cast %arg8 : i32 to index
      %16 = arith.index_cast %13 : i32 to index
      %17 = affine.apply #map()[%15, %16]
      %18 = arith.index_cast %17 : index to i64
      %19 = arith.cmpi eq, %18, %c1_i64 : i64
      %20 = arith.select %19, %1, %0 : memref<1xi32, #hivm.address_space<ub>>
      %21 = arith.index_cast %arg8 : i32 to index
      %22 = arith.index_cast %13 : i32 to index
      %23 = affine.apply #map()[%21, %22]
      %24 = arith.index_cast %23 : index to i64
      %25 = arith.cmpi eq, %24, %c1_i64 : i64
      %26 = arith.select %25, %3, %2 : memref<1xi32, #hivm.address_space<ub>>
      %27 = arith.index_cast %arg8 : i32 to index
      %28 = arith.index_cast %13 : i32 to index
      %29 = affine.apply #map()[%27, %28]
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.cmpi eq, %30, %c1_i64 : i64
      %32 = arith.select %31, %5, %4 : memref<1xi32, #hivm.address_space<ub>>
      %33 = arith.index_cast %arg8 : i32 to index
      %34 = arith.index_cast %13 : i32 to index
      %35 = affine.apply #map()[%33, %34]
      %36 = arith.index_cast %35 : index to i64
      %37 = arith.cmpi eq, %36, %c1_i64 : i64
      %38 = arith.select %37, %7, %6 : memref<1xi32, #hivm.address_space<ub>>
      %39 = arith.index_cast %arg8 : i32 to index
      %40 = arith.index_cast %13 : i32 to index
      %41 = affine.apply #map()[%39, %40]
      %42 = arith.index_cast %41 : index to i64
      %43 = arith.cmpi eq, %42, %c1_i64 : i64
      %44 = arith.select %43, %9, %8 : memref<8xi32, #hivm.address_space<ub>>
      %45 = arith.index_cast %arg8 : i32 to index
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
      %57 = arith.divsi %arg8, %14 : i32
      %58 = arith.remsi %57, %arg5 : i32
      %59 = arith.muli %58, %c4_i32 : i32
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %56]
      %cast = memref.cast %reinterpret_cast : memref<8xi32, strided<[1]>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_0 = memref.cast %44 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_int32_t(%cast, %cast_0, %c0_i32, %c0_i32, %c0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %60 = hivm.hir.pointer_cast(%c32_i64) : memref<8xi32, #hivm.address_space<ub>>
      %cast_1 = memref.cast %60 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_int32_t_to_1d(%59, %cast_1) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %61 = hivm.hir.pointer_cast(%c32_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_2 = memref.cast %44 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_3 = memref.cast %60 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_4 = memref.cast %61 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_2, %cast_3, %cast_4) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %62 = hivm.hir.pointer_cast(%c64_i64) : memref<8xi32, #hivm.address_space<ub>>
      %63 = hivm.hir.pointer_cast(%c96_i64) : memref<24xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_5 = memref.cast %61 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_6 = memref.cast %62 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_7 = memref.cast %63 : memref<24xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_ss_1d_int32_t(%cast_5, %c1_i32, %c0_i32, %cast_6, %cast_7) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %64 = arith.addi %59, %c1_i32 : i32
      %65 = hivm.hir.pointer_cast(%c192_i64) : memref<8xi32, #hivm.address_space<ub>>
      %cast_8 = memref.cast %65 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_int32_t_to_1d(%64, %cast_8) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %66 = hivm.hir.pointer_cast(%c192_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_9 = memref.cast %44 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_10 = memref.cast %65 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_11 = memref.cast %66 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_9, %cast_10, %cast_11) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %67 = hivm.hir.pointer_cast(%c224_i64) : memref<8xi32, #hivm.address_space<ub>>
      %68 = hivm.hir.pointer_cast(%c256_i64) : memref<24xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_12 = memref.cast %66 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %67 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %68 : memref<24xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_ss_1d_int32_t(%cast_12, %c1_i32, %c0_i32, %cast_13, %cast_14) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %69 = arith.addi %59, %c2_i32 : i32
      %70 = hivm.hir.pointer_cast(%c352_i64) : memref<8xi32, #hivm.address_space<ub>>
      %cast_15 = memref.cast %70 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_int32_t_to_1d(%69, %cast_15) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %71 = hivm.hir.pointer_cast(%c352_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_16 = memref.cast %44 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_17 = memref.cast %70 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_18 = memref.cast %71 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_16, %cast_17, %cast_18) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %72 = hivm.hir.pointer_cast(%c384_i64) : memref<8xi32, #hivm.address_space<ub>>
      %73 = hivm.hir.pointer_cast(%c416_i64) : memref<24xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_19 = memref.cast %71 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %72 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_21 = memref.cast %73 : memref<24xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_ss_1d_int32_t(%cast_19, %c1_i32, %c0_i32, %cast_20, %cast_21) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %74 = arith.addi %59, %c3_i32 : i32
      %75 = hivm.hir.pointer_cast(%c512_i64) : memref<8xi32, #hivm.address_space<ub>>
      %cast_22 = memref.cast %75 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_int32_t_to_1d(%74, %cast_22) : (i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %76 = hivm.hir.pointer_cast(%c512_i64) : memref<8xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_23 = memref.cast %44 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_24 = memref.cast %75 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_25 = memref.cast %76 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_23, %cast_24, %cast_25) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %56]
      %77 = hivm.hir.pointer_cast(%c544_i64) : memref<8xi32, #hivm.address_space<ub>>
      %78 = hivm.hir.pointer_cast(%c576_i64) : memref<24xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_26 = memref.cast %76 : memref<8xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_27 = memref.cast %77 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_28 = memref.cast %78 : memref<24xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_ss_1d_int32_t(%cast_26, %c1_i32, %c0_i32, %cast_27, %cast_28) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %79 = arith.cmpi slt, %59, %c4_i32 : i32
      scf.if %79 {
        %83 = hivm.hir.pointer_cast(%c672_i64) : memref<1xi32, #hivm.address_space<ub>>
        %84 = hivm.hir.pointer_cast(%c704_i64) : memref<16xi32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID3>]
        %cast_29 = memref.cast %62 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_30 = memref.cast %83 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_31 = memref.cast %84 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_r_int32_t(%cast_29, %cast_30, %cast_31, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %85 = memref.load %83[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID3>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %53]
        memref.store %85, %38[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %86 = arith.index_cast %59 : i32 to index
        %reinterpret_cast_32 = memref.reinterpret_cast %arg4 to offset: [%86], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_33 = memref.cast %38 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_34 = memref.cast %reinterpret_cast_32 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_33, %cast_34, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %53]
      } {pipeline.veconly}
      %80 = arith.cmpi slt, %64, %c4_i32 : i32
      scf.if %80 {
        %83 = hivm.hir.pointer_cast(%c800_i64) : memref<1xi32, #hivm.address_space<ub>>
        %84 = hivm.hir.pointer_cast(%c832_i64) : memref<16xi32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID2>]
        %cast_29 = memref.cast %67 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_30 = memref.cast %83 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_31 = memref.cast %84 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_r_int32_t(%cast_29, %cast_30, %cast_31, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %85 = memref.load %83[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %51]
        memref.store %85, %32[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %86 = arith.index_cast %59 : i32 to index
        %87 = affine.apply #map1()[%86]
        %reinterpret_cast_32 = memref.reinterpret_cast %arg4 to offset: [%87], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_33 = memref.cast %32 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_34 = memref.cast %reinterpret_cast_32 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_33, %cast_34, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %51]
      } {pipeline.veconly}
      %81 = arith.cmpi slt, %69, %c4_i32 : i32
      scf.if %81 {
        %83 = hivm.hir.pointer_cast(%c928_i64) : memref<1xi32, #hivm.address_space<ub>>
        %84 = hivm.hir.pointer_cast(%c960_i64) : memref<16xi32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
        %cast_29 = memref.cast %72 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_30 = memref.cast %83 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_31 = memref.cast %84 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_r_int32_t(%cast_29, %cast_30, %cast_31, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %85 = memref.load %83[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %49]
        memref.store %85, %26[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %86 = arith.index_cast %59 : i32 to index
        %87 = affine.apply #map2()[%86]
        %reinterpret_cast_32 = memref.reinterpret_cast %arg4 to offset: [%87], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_33 = memref.cast %26 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_34 = memref.cast %reinterpret_cast_32 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_33, %cast_34, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %49]
      } {pipeline.veconly}
      %82 = arith.cmpi slt, %74, %c4_i32 : i32
      scf.if %82 {
        %83 = hivm.hir.pointer_cast(%c1056_i64) : memref<1xi32, #hivm.address_space<ub>>
        %84 = hivm.hir.pointer_cast(%c1088_i64) : memref<16xi32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_29 = memref.cast %77 : memref<8xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_30 = memref.cast %83 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_31 = memref.cast %84 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_r_int32_t(%cast_29, %cast_30, %cast_31, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %85 = memref.load %83[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %56]
        memref.store %85, %20[%c0] : memref<1xi32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %86 = arith.index_cast %59 : i32 to index
        %87 = affine.apply #map3()[%86]
        %reinterpret_cast_32 = memref.reinterpret_cast %arg4 to offset: [%87], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_33 = memref.cast %20 : memref<1xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_34 = memref.cast %reinterpret_cast_32 : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int32_t(%cast_33, %cast_34, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %56]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID6>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID7>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
