#map = affine_map<()[s0] -> (s0 mod 2)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_correct_attn_cp_out_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_2d(f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_2d_bool(memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vor_2d_bool(memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_sv_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_max_ra_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_sum_ra_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vln_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_correct_attn_cp_out_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 0x7F800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %c256_i64 = arith.constant 256 : i64
    %c384_i64 = arith.constant 384 : i64
    %c512_i64 = arith.constant 512 : i64
    %c640_i64 = arith.constant 640 : i64
    %c704_i64 = arith.constant 704 : i64
    %c736_i64 = arith.constant 736 : i64
    %c864_i64 = arith.constant 864 : i64
    %c896_i64 = arith.constant 896 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c1088_i64 = arith.constant 1088 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1632_i64 = arith.constant 1632 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1760_i64 = arith.constant 1760 : i64
    %c1056_i64 = arith.constant 1056 : i64
    %c1792_i64 = arith.constant 1792 : i64
    %c1120_i64 = arith.constant 1120 : i64
    %c1376_i64 = arith.constant 1376 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %0 = hivm.hir.pointer_cast(%c1376_i64) : memref<64xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c2048_i64) : memref<64xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c1120_i64) : memref<64xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c1792_i64) : memref<64xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c1056_i64) : memref<1xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c1760_i64) : memref<1xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c128_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c1632_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %8 = arith.muli %arg12, %arg13 : i32
    %9 = arith.muli %8, %arg14 : i32
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.trunci %10 : i64 to i32
    %12 = arith.muli %arg14, %arg13 : i32
    %13 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8xf32, #hivm.address_space<ub>>
    %subview = memref.subview %13[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %cast = memref.cast %reinterpret_cast : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_2d(%cst_1, %cast) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %14 = arith.extsi %arg10 : i32 to i64
    %15 = arith.extsi %arg7 : i32 to i64
    %16 = arith.muli %arg11, %arg9 : i32
    %17 = arith.extsi %16 : i32 to i64
    %18 = arith.index_cast %arg9 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    scf.for %arg15 = %11 to %9 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %19 = arith.divsi %arg15, %12 : i32
      %20 = arith.remsi %19, %arg12 : i32
      %21 = arith.extsi %20 : i32 to i64
      %22 = arith.muli %21, %14 : i64
      %23 = arith.muli %21, %15 : i64
      %24 = arith.addi %17, %22 : i64
      %25 = arith.index_cast %22 : i64 to index
      scf.for %arg16 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
        %26 = arith.index_cast %arg16 : i32 to index
        %27 = affine.apply #map()[%26]
        %28 = arith.index_cast %27 : index to i64
        %29 = arith.cmpi eq, %28, %c1_i64 : i64
        %30 = arith.select %29, %1, %0 : memref<64xf32, #hivm.address_space<ub>>
        %31 = arith.index_cast %arg16 : i32 to index
        %32 = affine.apply #map()[%31]
        %33 = arith.index_cast %32 : index to i64
        %34 = arith.cmpi eq, %33, %c1_i64 : i64
        %35 = arith.select %34, %3, %2 : memref<64xf32, #hivm.address_space<ub>>
        %36 = arith.index_cast %arg16 : i32 to index
        %37 = affine.apply #map()[%36]
        %38 = arith.index_cast %37 : index to i64
        %39 = arith.cmpi eq, %38, %c1_i64 : i64
        %40 = arith.select %39, %5, %4 : memref<1xf32, #hivm.address_space<ub>>
        %41 = arith.index_cast %arg16 : i32 to index
        %42 = affine.apply #map()[%41]
        %43 = arith.index_cast %42 : index to i64
        %44 = arith.cmpi eq, %43, %c1_i64 : i64
        %45 = arith.select %44, %7, %6 : memref<4x8xf32, #hivm.address_space<ub>>
        %46 = arith.index_cast %arg16 : i32 to index
        %47 = affine.apply #map()[%46]
        %48 = arith.index_cast %47 : index to i1
        %49 = arith.xori %48, %true : i1
        %50 = arith.extui %49 : i1 to i64
        %51 = arith.extsi %arg16 : i32 to i64
        %52 = arith.index_cast %arg16 : i32 to index
        %53 = affine.apply #map1()[%25, %52]
        %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%53], sizes: [4], strides: [%18] : memref<?xf32, #hivm.address_space<gm>> to memref<4xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %subview_3 = memref.subview %45[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
        %cast_4 = memref.cast %reinterpret_cast_2 : memref<4xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_5 = memref.cast %subview_3 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_4, %cast_5, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %54 = hivm.hir.pointer_cast(%c256_i64) : memref<4x256xi1, #hivm.address_space<ub>>
        %subview_6 = memref.subview %54[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
        %base_buffer_7, %offset_8, %sizes_9, %strides_10 = memref.extract_strided_metadata %subview_3 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer_7 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %base_buffer_12, %offset_13, %sizes_14, %strides_15 = memref.extract_strided_metadata %subview_6 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_16 = memref.reinterpret_cast %base_buffer_12 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        scf.for %arg17 = %c0 to %c4 step %c1 {
          %subview_101 = memref.subview %reinterpret_cast_11[%arg17, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_102 = memref.subview %reinterpret_cast_11[%arg17, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_103 = memref.subview %reinterpret_cast_16[%arg17, 0] [1, 1] [1, 1] : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %cast_104 = memref.cast %subview_101 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_105 = memref.cast %subview_102 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_106 = memref.cast %subview_103 : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_104, %cast_105, %cast_106) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        }
        %55 = hivm.hir.pointer_cast(%c256_i64) : memref<4x256xi1, #hivm.address_space<ub>>
        %subview_17 = memref.subview %55[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
        %base_buffer_18, %offset_19, %sizes_20, %strides_21 = memref.extract_strided_metadata %subview_17 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_22 = memref.reinterpret_cast %base_buffer_18 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %56 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_23 = memref.cast %reinterpret_cast_16 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_24 = memref.cast %reinterpret_cast_22 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_25 = memref.cast %56 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_2d_bool(%cast_23, %cast_24, %cast_25) : (memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %57 = hivm.hir.pointer_cast(%c384_i64) : memref<4x256xi1, #hivm.address_space<ub>>
        %subview_26 = memref.subview %57[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
        %base_buffer_27, %offset_28, %sizes_29, %strides_30 = memref.extract_strided_metadata %subview_26 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_31 = memref.reinterpret_cast %base_buffer_27 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
        scf.for %arg17 = %c0 to %c4 step %c1 {
          %subview_101 = memref.subview %reinterpret_cast_11[%arg17, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_102 = memref.subview %reinterpret_cast[%arg17, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_103 = memref.subview %reinterpret_cast_31[%arg17, 0] [1, 1] [1, 1] : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %cast_104 = memref.cast %subview_101 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_105 = memref.cast %subview_102 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_106 = memref.cast %subview_103 : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_104, %cast_105, %cast_106) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        }
        %58 = hivm.hir.pointer_cast(%c256_i64) : memref<4x256xi1, #hivm.address_space<ub>>
        %subview_32 = memref.subview %58[0, 0] [4, 1] [1, 1] : memref<4x256xi1, #hivm.address_space<ub>> to memref<4xi1, strided<[256]>, #hivm.address_space<ub>>
        %base_buffer_33, %offset_34, %sizes_35, %strides_36 = memref.extract_strided_metadata %subview_32 : memref<4xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_37 = memref.reinterpret_cast %base_buffer_33 to offset: [0], sizes: [4, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %59 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_38 = memref.cast %reinterpret_cast_22 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_39 = memref.cast %reinterpret_cast_31 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_40 = memref.cast %reinterpret_cast_37 : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_41 = memref.cast %59 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vor_2d_bool(%cast_38, %cast_39, %cast_40, %cast_41) : (memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %60 = hivm.hir.pointer_cast(%c512_i64) : memref<4x8xf32, #hivm.address_space<ub>>
        %subview_42 = memref.subview %60[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_43, %offset_44, %sizes_45, %strides_46 = memref.extract_strided_metadata %subview_42 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_47 = memref.reinterpret_cast %base_buffer_43 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %61 = hivm.hir.pointer_cast(%c640_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        scf.for %arg17 = %c0 to %c4 step %c1 {
          %subview_101 = memref.subview %reinterpret_cast_37[%arg17, 0] [1, 1] [1, 1] : memref<4x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_102 = memref.subview %reinterpret_cast_11[%arg17, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_103 = memref.subview %reinterpret_cast_47[%arg17, 0] [1, 1] [1, 1] : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %cast_104 = memref.cast %subview_101 : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_105 = memref.cast %subview_102 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_106 = memref.cast %subview_103 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_107 = memref.cast %61 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_sv_1d_float(%cast_104, %cst_0, %cast_105, %cast_106, %cast_107) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        }
        %62 = hivm.hir.pointer_cast(%c704_i64) : memref<1xf32, #hivm.address_space<ub>>
        %base_buffer_48, %offset_49, %sizes_50, %strides_51 = memref.extract_strided_metadata %62 : memref<1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_52 = memref.reinterpret_cast %base_buffer_48 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %63 = hivm.hir.pointer_cast(%c736_i64) : memref<32xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_53 = memref.cast %reinterpret_cast_47 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_54 = memref.cast %reinterpret_cast_52 : memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_55 = memref.cast %63 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_max_ra_float(%cast_53, %cast_54, %cast_55, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %64 = memref.load %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %65 = arith.cmpf oeq, %64, %cst_0 : f32
        %66 = arith.select %65, %cst, %64 : f32
        %67 = hivm.hir.pointer_cast(%c512_i64) : memref<4x8xf32, #hivm.address_space<ub>>
        %subview_56 = memref.subview %67[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
        %68 = arith.subf %cst, %66 : f32
        %base_buffer_57, %offset_58, %sizes_59, %strides_60 = memref.extract_strided_metadata %subview_56 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_61 = memref.reinterpret_cast %base_buffer_57 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %69 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_62 = memref.cast %reinterpret_cast_47 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_63 = memref.cast %reinterpret_cast_61 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_64 = memref.cast %69 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_2d_float(%cast_62, %68, %cast_63, %cast_64) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %70 = hivm.hir.pointer_cast(%c512_i64) : memref<4x8xf32, #hivm.address_space<ub>>
        %subview_65 = memref.subview %70[0, 0] [4, 1] [1, 1] : memref<4x8xf32, #hivm.address_space<ub>> to memref<4xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_66, %offset_67, %sizes_68, %strides_69 = memref.extract_strided_metadata %subview_65 : memref<4xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_70 = memref.reinterpret_cast %base_buffer_66 to offset: [0], sizes: [4, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %71 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_71 = memref.cast %reinterpret_cast_61 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_72 = memref.cast %reinterpret_cast_70 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_73 = memref.cast %71 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vexp_2d_float(%cast_71, %cast_72, %cast_73) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %72 = hivm.hir.pointer_cast(%c864_i64) : memref<1xf32, #hivm.address_space<ub>>
        %base_buffer_74, %offset_75, %sizes_76, %strides_77 = memref.extract_strided_metadata %72 : memref<1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_78 = memref.reinterpret_cast %base_buffer_74 to offset: [0], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>>
        %73 = hivm.hir.pointer_cast(%c896_i64) : memref<32xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_79 = memref.cast %reinterpret_cast_70 : memref<4x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_80 = memref.cast %reinterpret_cast_78 : memref<1x1xf32, strided<[1, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_81 = memref.cast %73 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_ra_float(%cast_79, %cast_80, %cast_81, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %74 = memref.load %72[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %75 = hivm.hir.pointer_cast(%c1024_i64) : memref<1xf32, #hivm.address_space<ub>>
        memref.store %74, %75[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %76 = hivm.hir.pointer_cast(%c1024_i64) : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %77 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_82 = memref.cast %75 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_83 = memref.cast %76 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_84 = memref.cast %77 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vln_1d_float(%cast_82, %cast_83, %cast_84) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %78 = memref.load %76[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %79 = arith.addf %78, %66 : f32
        %80 = arith.addi %22, %51 : i64
        %81 = arith.index_cast %80 : i64 to index
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %50]
        memref.store %79, %40[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast_85 = memref.reinterpret_cast %arg6 to offset: [%81], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_86 = memref.cast %40 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_87 = memref.cast %reinterpret_cast_85 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_86, %cast_87, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %50]
        %82 = arith.muli %arg16, %arg8 : i32
        %83 = arith.extsi %82 : i32 to i64
        %84 = arith.addi %23, %83 : i64
        %85 = arith.addi %24, %51 : i64
        %86 = arith.index_cast %85 : i64 to index
        %reinterpret_cast_88 = memref.reinterpret_cast %arg5 to offset: [%86], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %87 = memref.load %reinterpret_cast_88[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %88 = arith.subf %87, %79 : f32
        %89 = arith.cmpf une, %88, %88 : f32
        %90 = arith.cmpf oeq, %88, %cst_1 : f32
        %91 = arith.ori %89, %90 : i1
        %92 = arith.select %91, %cst_0, %88 : f32
        %93 = hivm.hir.pointer_cast(%c1088_i64) : memref<1xf32, #hivm.address_space<ub>>
        memref.store %92, %93[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %94 = hivm.hir.pointer_cast(%c1088_i64) : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
        %95 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_89 = memref.cast %93 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_90 = memref.cast %94 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_91 = memref.cast %95 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vexp_1d_float(%cast_89, %cast_90, %cast_91) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
        %96 = memref.load %94[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %97 = arith.index_cast %84 : i64 to index
        %reinterpret_cast_92 = memref.reinterpret_cast %arg3 to offset: [%97], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %cast_93 = memref.cast %reinterpret_cast_92 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_94 = memref.cast %35 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_93, %cast_94, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %50]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %98 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_95 = memref.cast %35 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_96 = memref.cast %30 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_97 = memref.cast %98 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_95, %96, %cast_96, %cast_97) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast_98 = memref.reinterpret_cast %arg4 to offset: [%97], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_99 = memref.cast %30 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_100 = memref.cast %reinterpret_cast_98 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_99, %cast_100, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %50]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
