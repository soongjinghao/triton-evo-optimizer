#map = affine_map<()[s0, s1] -> (s1, s0)>
#map1 = affine_map<()[s0, s1] -> (s1 + 16, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map3 = affine_map<()[s0, s1] -> (16, s0 - s1)>
#map4 = affine_map<()[s0] -> (0, s0)>
#map5 = affine_map<()[s0, s1] -> (64, s0 - s1)>
#map6 = affine_map<()[s0] -> (16, s0)>
#map7 = affine_map<()[s0] -> (64, s0)>
#map8 = affine_map<()[s0, s1, s2, s3, s4, s5] -> ((s0 + s2 * s1 + ((s4 - s5) floordiv 48) * (s1 * s3)) mod 2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @eye_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @arange_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_2d_int32_t(memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_int32_t(memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_first_axis_align_2d_int32_t(memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_ss_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @eye_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, false, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %c320_i64 = arith.constant 320 : i64
    %c384_i64 = arith.constant 384 : i64
    %c640_i64 = arith.constant 640 : i64
    %c704_i64 = arith.constant 704 : i64
    %c4800_i64 = arith.constant 4800 : i64
    %c5312_i64 = arith.constant 5312 : i64
    %c5376_i64 = arith.constant 5376 : i64
    %c9472_i64 = arith.constant 9472 : i64
    %c9984_i64 = arith.constant 9984 : i64
    %c14080_i64 = arith.constant 14080 : i64
    %c14592_i64 = arith.constant 14592 : i64
    %c18688_i64 = arith.constant 18688 : i64
    %c22784_i64 = arith.constant 22784 : i64
    %c1_i64 = arith.constant 1 : i64
    %c22880_i64 = arith.constant 22880 : i64
    %c22912_i64 = arith.constant 22912 : i64
    %0 = hivm.hir.pointer_cast(%c22880_i64) : memref<1xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c22912_i64) : memref<1xf32, #hivm.address_space<ub>>
    %2 = arith.muli %arg6, %arg7 : i32
    %3 = arith.muli %2, %arg8 : i32
    %4 = hivm.hir.get_block_idx -> i64
    %5 = arith.trunci %4 : i64 to i32
    %6 = arith.muli %arg8, %arg7 : i32
    %7 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
    %cast = memref.cast %7 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %8 = hivm.hir.pointer_cast(%c64_i64) : memref<64xi32, #hivm.address_space<ub>>
    %cast_1 = memref.cast %8 : memref<64xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast_1, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %9 = arith.remsi %arg5, %c64_i32 : i32
    %10 = arith.cmpi eq, %9, %c0_i32 : i32
    %11 = arith.index_cast %arg4 : i32 to index
    %12 = arith.index_cast %arg5 : i32 to index
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    scf.for %arg9 = %5 to %3 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %13 = arith.divsi %arg9, %arg8 : i32
      %14 = arith.remsi %13, %arg7 : i32
      %15 = arith.divsi %arg9, %6 : i32
      %16 = arith.remsi %15, %arg6 : i32
      %17 = arith.muli %16, %c16_i32 : i32
      %18 = hivm.hir.pointer_cast(%c320_i64) : memref<16xi32, #hivm.address_space<ub>>
      %19 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_2 = memref.cast %7 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_3 = memref.cast %18 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_4 = memref.cast %19 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_int32_t(%cast_2, %17, %cast_3, %cast_4) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %20 = arith.muli %14, %c64_i32 : i32
      %21 = hivm.hir.pointer_cast(%c384_i64) : memref<64xi32, #hivm.address_space<ub>>
      %22 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
      %cast_5 = memref.cast %8 : memref<64xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_6 = memref.cast %21 : memref<64xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_7 = memref.cast %22 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_int32_t(%cast_5, %20, %cast_6, %cast_7) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %23 = scf.if %10 -> (memref<16x64xi32, #hivm.address_space<ub>>) {
        %42 = hivm.hir.pointer_cast(%c640_i64) : memref<16x1xi32, #hivm.address_space<ub>>
        %collapse_shape_23 = memref.collapse_shape %42 [[0, 1]] : memref<16x1xi32, #hivm.address_space<ub>> into memref<16xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %43 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
        %cast_24 = memref.cast %18 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_25 = memref.cast %collapse_shape_23 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_26 = memref.cast %43 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_int32_t(%cast_24, %c64_i32, %cast_25, %cast_26) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape_27 = memref.expand_shape %21 [[0, 1]] output_shape [1, 64] : memref<64xi32, #hivm.address_space<ub>> into memref<1x64xi32, #hivm.address_space<ub>>
        %44 = hivm.hir.pointer_cast(%c704_i64) : memref<16x64xi32, #hivm.address_space<ub>>
        %45 = hivm.hir.pointer_cast(%c4800_i64) : memref<128xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_28 = memref.cast %42 : memref<16x1xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_29 = memref.cast %expand_shape_27 : memref<1x64xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_30 = memref.cast %44 : memref<16x64xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_31 = memref.cast %45 : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_2d_int32_t(%cast_28, %cast_29, %cast_30, %cast_31) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        scf.yield %44 : memref<16x64xi32, #hivm.address_space<ub>>
      } else {
        %42 = hivm.hir.pointer_cast(%c5312_i64) : memref<16x1xi32, #hivm.address_space<ub>>
        %collapse_shape_23 = memref.collapse_shape %42 [[0, 1]] : memref<16x1xi32, #hivm.address_space<ub>> into memref<16xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %43 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi32, #hivm.address_space<ub>>
        %cast_24 = memref.cast %18 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_25 = memref.cast %collapse_shape_23 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_26 = memref.cast %43 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_int32_t(%cast_24, %arg5, %cast_25, %cast_26) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape_27 = memref.expand_shape %21 [[0, 1]] output_shape [1, 64] : memref<64xi32, #hivm.address_space<ub>> into memref<1x64xi32, #hivm.address_space<ub>>
        %44 = hivm.hir.pointer_cast(%c5376_i64) : memref<16x64xi32, #hivm.address_space<ub>>
        %45 = hivm.hir.pointer_cast(%c9472_i64) : memref<128xi32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_28 = memref.cast %42 : memref<16x1xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_29 = memref.cast %expand_shape_27 : memref<1x64xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_30 = memref.cast %44 : memref<16x64xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_31 = memref.cast %45 : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_2d_int32_t(%cast_28, %cast_29, %cast_30, %cast_31) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        scf.yield %44 : memref<16x64xi32, #hivm.address_space<ub>>
      }
      %expand_shape = memref.expand_shape %18 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
      %24 = hivm.hir.pointer_cast(%c9984_i64) : memref<16x64xi32, #hivm.address_space<ub>>
      %25 = hivm.hir.pointer_cast(%c14080_i64) : memref<128xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_8 = memref.cast %expand_shape : memref<16x1xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_9 = memref.cast %24 : memref<16x64xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_10 = memref.cast %25 : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_last_axis_align_2d_int32_t(%cast_8, %cast_9, %cast_10) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %expand_shape_11 = memref.expand_shape %21 [[0, 1]] output_shape [1, 64] : memref<64xi32, #hivm.address_space<ub>> into memref<1x64xi32, #hivm.address_space<ub>>
      %26 = hivm.hir.pointer_cast(%c14592_i64) : memref<16x64xi32, #hivm.address_space<ub>>
      %cast_12 = memref.cast %expand_shape_11 : memref<1x64xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %26 : memref<16x64xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_first_axis_align_2d_int32_t(%cast_12, %cast_13) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %27 = hivm.hir.pointer_cast(%c9984_i64) : memref<16x64xi1, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %24 [[0, 1]] : memref<16x64xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
      %collapse_shape_14 = memref.collapse_shape %26 [[0, 1]] : memref<16x64xi32, #hivm.address_space<ub>> into memref<1024xi32, #hivm.address_space<ub>>
      %collapse_shape_15 = memref.collapse_shape %27 [[0, 1]] : memref<16x64xi1, #hivm.address_space<ub>> into memref<1024xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_16 = memref.cast %collapse_shape : memref<1024xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_17 = memref.cast %collapse_shape_14 : memref<1024xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_18 = memref.cast %collapse_shape_15 : memref<1024xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_int32_t(%cast_16, %cast_17, %cast_18) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %28 = hivm.hir.pointer_cast(%c18688_i64) : memref<16x64xf32, #hivm.address_space<ub>>
      %collapse_shape_19 = memref.collapse_shape %28 [[0, 1]] : memref<16x64xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      %29 = hivm.hir.pointer_cast(%c22784_i64) : memref<24xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_20 = memref.cast %collapse_shape_15 : memref<1024xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_21 = memref.cast %collapse_shape_19 : memref<1024xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_22 = memref.cast %29 : memref<24xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_ss_1d_float(%cast_20, %cst_0, %cst, %cast_21, %cast_22) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %30 = arith.index_cast %17 : i32 to index
      %31 = affine.max #map()[%30, %11]
      %32 = affine.min #map1()[%31, %30]
      %33 = arith.index_cast %20 : i32 to index
      %34 = affine.max #map()[%33, %12]
      %35 = affine.min #map2()[%34, %33]
      %36 = affine.min #map3()[%32, %30]
      %37 = affine.max #map4()[%36]
      %38 = affine.min #map5()[%35, %33]
      %39 = affine.max #map4()[%38]
      %40 = affine.min #map6()[%37]
      %41 = affine.min #map7()[%39]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg10 = %c0 to %40 step %c1 {
        %42 = arith.index_cast %arg10 : index to i32
        %43 = arith.addi %17, %42 : i32
        %44 = arith.cmpi slt, %43, %arg4 : i32
        scf.for %arg11 = %c0 to %41 step %c1 {
          %45 = arith.index_cast %arg9 : i32 to index
          %46 = arith.index_cast %5 : i32 to index
          %47 = affine.apply #map8()[%arg11, %41, %arg10, %40, %45, %46]
          %48 = arith.index_cast %47 : index to i64
          %49 = arith.cmpi eq, %48, %c1_i64 : i64
          %50 = arith.select %49, %1, %0 : memref<1xf32, #hivm.address_space<ub>>
          %51 = arith.index_cast %arg9 : i32 to index
          %52 = arith.index_cast %5 : i32 to index
          %53 = affine.apply #map8()[%arg11, %41, %arg10, %40, %51, %52]
          %54 = arith.index_cast %53 : index to i1
          %55 = arith.xori %54, %true : i1
          %56 = arith.extui %55 : i1 to i64
          %57 = memref.load %23[%arg10, %arg11] {DiscreteMemAccess} : memref<16x64xi32, #hivm.address_space<ub>>
          %58 = memref.load %28[%arg10, %arg11] {DiscreteMemAccess} : memref<16x64xf32, #hivm.address_space<ub>>
          %59 = arith.index_cast %arg11 : index to i32
          %60 = arith.addi %20, %59 : i32
          %61 = arith.cmpi slt, %60, %arg5 : i32
          %62 = arith.andi %44, %61 : i1
          scf.if %62 {
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %56]
            memref.store %58, %50[%c0] : memref<1xf32, #hivm.address_space<ub>>
            hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            %63 = arith.index_cast %57 : i32 to index
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%63], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.pipe_barrier[<PIPE_MTE3>]
            %cast_23 = memref.cast %50 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_24 = memref.cast %reinterpret_cast : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            func.call @store_ubuf_to_gm_1d_float(%cast_23, %cast_24, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %56]
          } {pipeline.veconly}
        } {ExtractedLoadOrStore, pipeline.veconly}
      } {ExtractedLoadOrStore, pipeline.veconly}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
