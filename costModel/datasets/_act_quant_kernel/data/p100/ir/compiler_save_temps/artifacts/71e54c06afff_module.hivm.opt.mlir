#map = affine_map<()[s0, s1] -> (s1, s0)>
#map1 = affine_map<()[s0, s1] -> (s1 + 32, s0)>
#map2 = affine_map<()[s0, s1] -> (32, s0 - s1)>
#map3 = affine_map<()[s0] -> (0, s0)>
#map4 = affine_map<()[s0] -> (s0 mod 2)>
#map5 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
#map6 = affine_map<()[s0, s1] -> (s1 + 128, s0)>
#map7 = affine_map<()[s0, s1] -> (128, s0 - s1)>
#map8 = affine_map<()[s0, s1] -> (s1 * 32, s0)>
#map9 = affine_map<()[s0, s1] -> (s1 * 128, s0)>
#map10 = affine_map<()[s0, s1, s2] -> (s2 * 32, s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @_act_quant_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_2d(f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_2d_bool(memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vabs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_max_ar_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmaxs_vs_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_sv_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vv_1d_bool_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vdiv_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmaxs_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmins_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_half_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_act_quant_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0xFF800000 : f32
    %c2_i32 = arith.constant 2 : i32
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %cst_1 = arith.constant 0.00223214296 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 4.480000e+02 : f32
    %cst_3 = arith.constant -4.480000e+02 : f32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_4 = arith.constant 9.99999974E-5 : f32
    %c67584_i64 = arith.constant 67584 : i64
    %c68608_i64 = arith.constant 68608 : i64
    %c84992_i64 = arith.constant 84992 : i64
    %c101376_i64 = arith.constant 101376 : i64
    %c117760_i64 = arith.constant 117760 : i64
    %c118784_i64 = arith.constant 118784 : i64
    %c102400_i64 = arith.constant 102400 : i64
    %c103424_i64 = arith.constant 103424 : i64
    %c104448_i64 = arith.constant 104448 : i64
    %c101888_i64 = arith.constant 101888 : i64
    %c118272_i64 = arith.constant 118272 : i64
    %c134656_i64 = arith.constant 134656 : i64
    %c151040_i64 = arith.constant 151040 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c0_i64 = arith.constant 0 : i64
    %c33792_i64 = arith.constant 33792 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c34816_i64 = arith.constant 34816 : i64
    %c51200_i64 = arith.constant 51200 : i64
    %0 = hivm.hir.pointer_cast(%c34816_i64) : memref<32x128xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c51200_i64) : memref<32x128xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c32768_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c33792_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c0_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c16384_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c34816_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c51200_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %8 = arith.muli %arg8, %arg9 : i32
    %9 = arith.muli %8, %arg10 : i32
    %10 = hivm.hir.get_block_idx -> i64
    %11 = arith.trunci %10 : i64 to i32
    %12 = arith.muli %arg10, %arg9 : i32
    %13 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview = memref.subview %13[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %cast = memref.cast %reinterpret_cast : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_2d(%cst_4, %cast) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %14 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %14 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %cast_5 = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_3, %cast_5) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %15 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_6 = memref.collapse_shape %15 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %cast_7 = memref.cast %collapse_shape_6 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_2, %cast_7) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %16 = arith.divsi %arg7, %c128_i32 : i32
    %17 = arith.index_cast %arg7 : i32 to index
    %18 = arith.index_cast %arg6 : i32 to index
    %19 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_8 = memref.subview %19[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_9, %offset_10, %sizes_11, %strides_12 = memref.extract_strided_metadata %subview_8 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_13 = memref.reinterpret_cast %base_buffer_9 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    scf.for %arg11 = %c0 to %c32 step %c1 {
      %subview_39 = memref.subview %reinterpret_cast[%arg11, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_40 = memref.subview %reinterpret_cast[%arg11, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_41 = memref.subview %reinterpret_cast_13[%arg11, 0] [1, 1] [1, 1] : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %cast_42 = memref.cast %subview_39 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_43 = memref.cast %subview_40 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_44 = memref.cast %subview_41 : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_float(%cast_42, %cast_43, %cast_44) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    }
    %20 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_14 = memref.subview %20[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_15, %offset_16, %sizes_17, %strides_18 = memref.extract_strided_metadata %subview_14 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_19 = memref.reinterpret_cast %base_buffer_15 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %21 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_20 = memref.cast %reinterpret_cast_13 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_21 = memref.cast %reinterpret_cast_19 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_22 = memref.cast %21 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_2d_bool(%cast_20, %cast_21, %cast_22) : (memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %22 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_23 = memref.collapse_shape %22 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    %cast_24 = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_25 = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_26 = memref.cast %collapse_shape_23 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_float(%cast_24, %cast_25, %cast_26) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %23 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_27 = memref.collapse_shape %23 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %24 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_28 = memref.cast %collapse_shape_23 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_29 = memref.cast %collapse_shape_27 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_30 = memref.cast %24 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_1d_bool(%cast_28, %cast_29, %cast_30) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %25 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_31 = memref.collapse_shape %25 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    %cast_32 = memref.cast %collapse_shape_6 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_33 = memref.cast %collapse_shape_6 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_34 = memref.cast %collapse_shape_31 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_float(%cast_32, %cast_33, %cast_34) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %26 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_35 = memref.collapse_shape %26 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %27 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_36 = memref.cast %collapse_shape_31 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_37 = memref.cast %collapse_shape_35 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_38 = memref.cast %27 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_1d_bool(%cast_36, %cast_37, %cast_38) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %28 = arith.index_cast %16 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    scf.for %arg11 = %11 to %9 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %29 = arith.divsi %arg11, %arg10 : i32
      %30 = arith.remsi %29, %arg9 : i32
      %31 = arith.divsi %arg11, %12 : i32
      %32 = arith.remsi %31, %arg8 : i32
      %33 = arith.muli %32, %c32_i32 : i32
      %34 = arith.muli %30, %c4_i32 : i32
      %35 = arith.index_cast %33 : i32 to index
      %36 = affine.max #map()[%35, %18]
      %37 = affine.min #map1()[%36, %35]
      %38 = affine.min #map2()[%37, %35]
      %39 = affine.max #map3()[%38]
      scf.for %arg12 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %40 = arith.index_cast %arg12 : i32 to index
        %41 = affine.apply #map4()[%40]
        %42 = arith.index_cast %41 : index to i64
        %43 = arith.cmpi eq, %42, %c1_i64 : i64
        %44 = arith.select %43, %1, %0 : memref<32x128xf16, #hivm.address_space<ub>>
        %45 = arith.index_cast %arg12 : i32 to index
        %46 = affine.apply #map4()[%45]
        %47 = arith.index_cast %46 : index to i64
        %48 = arith.cmpi eq, %47, %c1_i64 : i64
        %49 = arith.select %48, %3, %2 : memref<32x8xf32, #hivm.address_space<ub>>
        %50 = arith.index_cast %arg12 : i32 to index
        %51 = affine.apply #map4()[%50]
        %52 = arith.index_cast %51 : index to i64
        %53 = arith.cmpi eq, %52, %c1_i64 : i64
        %54 = arith.select %53, %5, %4 : memref<32x128xf32, #hivm.address_space<ub>>
        %55 = arith.index_cast %arg12 : i32 to index
        %56 = affine.apply #map4()[%55]
        %57 = arith.index_cast %56 : index to i64
        %58 = arith.cmpi eq, %57, %c1_i64 : i64
        %59 = arith.select %58, %7, %6 : memref<32x128xf32, #hivm.address_space<ub>>
        %60 = arith.index_cast %arg12 : i32 to index
        %61 = affine.apply #map4()[%60]
        %62 = arith.index_cast %61 : index to i1
        %63 = arith.select %62, %c1_i64, %c2_i64 : i64
        %64 = arith.index_cast %61 : index to i1
        %65 = arith.xori %64, %true : i1
        %66 = arith.extui %65 : i1 to i64
        %67 = arith.addi %34, %arg12 : i32
        %68 = arith.cmpi slt, %67, %16 : i32
        %69 = arith.muli %67, %c128_i32 : i32
        %70 = arith.index_cast %69 : i32 to index
        %71 = affine.apply #map5()[%70, %35, %17]
        %reinterpret_cast_39 = memref.reinterpret_cast %arg3 to offset: [%71], sizes: [32, 128], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %72 = affine.max #map()[%70, %17]
        %73 = affine.min #map6()[%72, %70]
        %74 = affine.min #map7()[%73, %70]
        %75 = affine.max #map3()[%74]
        %76 = arith.index_castui %68 : i1 to index
        %77 = affine.min #map8()[%39, %76]
        %78 = affine.max #map3()[%77]
        %79 = affine.min #map9()[%75, %76]
        %80 = affine.max #map3()[%79]
        %81 = arith.cmpi slt, %78, %c32 : index
        %82 = arith.cmpi slt, %80, %c128 : index
        %83 = arith.ori %81, %82 : i1
        %subview_40 = memref.subview %reinterpret_cast_39[0, 0] [%78, %80] [1, 1] : memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_41 = memref.subview %54[0, 0] [%78, %80] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
        scf.if %83 {
          %collapse_shape_167 = memref.collapse_shape %54 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
          %cast_168 = memref.cast %collapse_shape_167 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_168) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %63]
        %cast_42 = memref.cast %subview_40 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_43 = memref.cast %subview_41 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_42, %cast_43, %c2_i32, %cst_0, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %84 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_44 = memref.collapse_shape %54 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %collapse_shape_45 = memref.collapse_shape %84 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %85 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_46 = memref.cast %collapse_shape_44 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_47 = memref.cast %collapse_shape_45 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_48 = memref.cast %85 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vabs_1d_float(%cast_46, %cast_47, %cast_48) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %86 = hivm.hir.pointer_cast(%c117760_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
        %subview_49 = memref.subview %86[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %87 = hivm.hir.pointer_cast(%c118784_i64) : memref<2048xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_50 = memref.cast %84 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_51 = memref.cast %subview_49 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_52 = memref.cast %87 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @enablevc_reduce_max_ar_float(%cast_50, %cast_51, %cast_52, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        %collapse_shape_53 = memref.collapse_shape %subview_49 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %88 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x256xi1, #hivm.address_space<ub>>
        %subview_54 = memref.subview %88[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
        %base_buffer_55, %offset_56, %sizes_57, %strides_58 = memref.extract_strided_metadata %collapse_shape_53 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_59 = memref.reinterpret_cast %base_buffer_55 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %base_buffer_60, %offset_61, %sizes_62, %strides_63 = memref.extract_strided_metadata %subview_54 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_64 = memref.reinterpret_cast %base_buffer_60 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        scf.for %arg13 = %c0 to %c32 step %c1 {
          %subview_167 = memref.subview %reinterpret_cast_59[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_168 = memref.subview %reinterpret_cast_59[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_169 = memref.subview %reinterpret_cast_64[%arg13, 0] [1, 1] [1, 1] : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %cast_170 = memref.cast %subview_167 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_171 = memref.cast %subview_168 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_172 = memref.cast %subview_169 : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_170, %cast_171, %cast_172) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        }
        %89 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x256xi1, #hivm.address_space<ub>>
        %subview_65 = memref.subview %89[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
        %base_buffer_66, %offset_67, %sizes_68, %strides_69 = memref.extract_strided_metadata %subview_65 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_70 = memref.reinterpret_cast %base_buffer_66 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %90 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_71 = memref.cast %reinterpret_cast_64 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_72 = memref.cast %reinterpret_cast_70 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_73 = memref.cast %90 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_2d_bool(%cast_71, %cast_72, %cast_73) : (memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %91 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x8xf32, #hivm.address_space<ub>>
        %subview_74 = memref.subview %91[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_75, %offset_76, %sizes_77, %strides_78 = memref.extract_strided_metadata %subview_74 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_79 = memref.reinterpret_cast %base_buffer_75 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %92 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_80 = memref.cast %reinterpret_cast_59 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_81 = memref.cast %reinterpret_cast_79 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_82 = memref.cast %92 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmaxs_vs_2d_float(%cast_80, %cst_4, %cast_81, %cast_82) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %93 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x8xf32, #hivm.address_space<ub>>
        %subview_83 = memref.subview %93[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_84, %offset_85, %sizes_86, %strides_87 = memref.extract_strided_metadata %subview_83 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_88 = memref.reinterpret_cast %base_buffer_84 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %94 = hivm.hir.pointer_cast(%c104448_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        scf.for %arg13 = %c0 to %c32 step %c1 {
          %subview_167 = memref.subview %reinterpret_cast_70[%arg13, 0] [1, 1] [1, 1] : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_168 = memref.subview %reinterpret_cast_79[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_169 = memref.subview %reinterpret_cast_88[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %cast_170 = memref.cast %subview_167 : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_171 = memref.cast %subview_168 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_172 = memref.cast %subview_169 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_173 = memref.cast %94 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_sv_1d_float(%cast_170, %cst_4, %cast_171, %cast_172, %cast_173) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        }
        %subview_89 = memref.subview %49[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_90, %offset_91, %sizes_92, %strides_93 = memref.extract_strided_metadata %subview_89 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_94 = memref.reinterpret_cast %base_buffer_90 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %95 = hivm.hir.pointer_cast(%c101376_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %66]
        hivm.hir.pipe_barrier[<PIPE_V>]
        scf.for %arg13 = %c0 to %c32 step %c1 {
          %subview_167 = memref.subview %reinterpret_cast_19[%arg13, 0] [1, 1] [1, 1] : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_168 = memref.subview %reinterpret_cast_59[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_169 = memref.subview %reinterpret_cast_88[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_170 = memref.subview %reinterpret_cast_94[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %cast_171 = memref.cast %subview_167 : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_172 = memref.cast %subview_168 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_173 = memref.cast %subview_169 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_174 = memref.cast %subview_170 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_175 = memref.cast %95 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vv_1d_bool_float(%cast_171, %cast_172, %cast_173, %cast_174, %cast_175) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        }
        %subview_95 = memref.subview %49[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_96, %offset_97, %sizes_98, %strides_99 = memref.extract_strided_metadata %subview_95 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_100 = memref.reinterpret_cast %base_buffer_96 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %96 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_101 = memref.cast %reinterpret_cast_94 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_102 = memref.cast %reinterpret_cast_100 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_103 = memref.cast %96 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_2d_float(%cast_101, %cst_1, %cast_102, %cast_103) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape = memref.expand_shape %subview_95 [[0, 1]] output_shape [32, 1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
        %97 = hivm.hir.pointer_cast(%c101376_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_104 = memref.cast %54 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_105 = memref.cast %expand_shape : memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_106 = memref.cast %54 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_107 = memref.cast %97 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vdiv_2d_float(%cast_104, %cast_105, %cast_106, %cast_107) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %98 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x128xi1, #hivm.address_space<ub>>
        %collapse_shape_108 = memref.collapse_shape %54 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %collapse_shape_109 = memref.collapse_shape %98 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_110 = memref.cast %collapse_shape_108 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_111 = memref.cast %collapse_shape_108 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_112 = memref.cast %collapse_shape_109 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_float(%cast_110, %cast_111, %cast_112) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %99 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x128xi1, #hivm.address_space<ub>>
        %collapse_shape_113 = memref.collapse_shape %99 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %100 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_114 = memref.cast %collapse_shape_109 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_115 = memref.cast %collapse_shape_113 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_116 = memref.cast %100 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_114, %cast_115, %cast_116) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %101 = hivm.hir.pointer_cast(%c101888_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_117 = memref.collapse_shape %101 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %102 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_118 = memref.cast %collapse_shape_108 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_119 = memref.cast %collapse_shape_117 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_120 = memref.cast %102 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmaxs_vs_1d_float(%cast_118, %cst_3, %cast_119, %cast_120) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %103 = hivm.hir.pointer_cast(%c118272_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_121 = memref.collapse_shape %103 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %104 = hivm.hir.pointer_cast(%c134656_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_122 = memref.cast %collapse_shape_113 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_123 = memref.cast %collapse_shape_117 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_124 = memref.cast %collapse_shape_121 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_125 = memref.cast %104 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_sv_1d_float(%cast_122, %cst_3, %cast_123, %cast_124, %cast_125) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %105 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_126 = memref.collapse_shape %105 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %106 = hivm.hir.pointer_cast(%c117760_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_127 = memref.cast %collapse_shape_27 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_128 = memref.cast %collapse_shape_108 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_129 = memref.cast %collapse_shape_121 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_130 = memref.cast %collapse_shape_126 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_131 = memref.cast %106 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_vv_1d_bool_float(%cast_127, %cast_128, %cast_129, %cast_130, %cast_131) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %63]
        %107 = hivm.hir.pointer_cast(%c117760_i64) : memref<32x128xi1, #hivm.address_space<ub>>
        %collapse_shape_132 = memref.collapse_shape %107 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_133 = memref.cast %collapse_shape_126 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_134 = memref.cast %collapse_shape_126 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_135 = memref.cast %collapse_shape_132 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_float(%cast_133, %cast_134, %cast_135) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %108 = hivm.hir.pointer_cast(%c117760_i64) : memref<32x128xi1, #hivm.address_space<ub>>
        %collapse_shape_136 = memref.collapse_shape %108 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %109 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_137 = memref.cast %collapse_shape_132 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_138 = memref.cast %collapse_shape_136 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_139 = memref.cast %109 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_137, %cast_138, %cast_139) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %110 = hivm.hir.pointer_cast(%c118272_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_140 = memref.collapse_shape %110 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %111 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_141 = memref.cast %collapse_shape_126 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_142 = memref.cast %collapse_shape_140 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_143 = memref.cast %111 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmins_vs_1d_float(%cast_141, %cst_2, %cast_142, %cast_143) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %112 = hivm.hir.pointer_cast(%c134656_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_144 = memref.collapse_shape %112 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %113 = hivm.hir.pointer_cast(%c151040_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_145 = memref.cast %collapse_shape_136 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_146 = memref.cast %collapse_shape_140 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_147 = memref.cast %collapse_shape_144 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_148 = memref.cast %113 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_sv_1d_float(%cast_145, %cst_2, %cast_146, %cast_147, %cast_148) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %collapse_shape_149 = memref.collapse_shape %59 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %114 = hivm.hir.pointer_cast(%c117760_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_150 = memref.cast %collapse_shape_35 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_151 = memref.cast %collapse_shape_126 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_152 = memref.cast %collapse_shape_144 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_153 = memref.cast %collapse_shape_149 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_154 = memref.cast %114 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_vv_1d_bool_float(%cast_150, %cast_151, %cast_152, %cast_153, %cast_154) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %reinterpret_cast_155 = memref.reinterpret_cast %arg4 to offset: [%71], sizes: [32, 128], strides: [%17, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_156 = memref.collapse_shape %44 [[0, 1]] : memref<32x128xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_157 = memref.cast %collapse_shape_149 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_158 = memref.cast %collapse_shape_156 : memref<4096xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_float_to_half_1d_with_mode(%cast_157, %cast_158, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %subview_159 = memref.subview %44[0, 0] [%78, %80] [1, 1] : memref<32x128xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>
        %subview_160 = memref.subview %reinterpret_cast_155[0, 0] [%78, %80] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_161 = memref.cast %subview_159 : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_162 = memref.cast %subview_160 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_half(%cast_161, %cast_162, %c0_i32) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        %115 = arith.index_cast %67 : i32 to index
        %116 = affine.apply #map5()[%115, %35, %28]
        %reinterpret_cast_163 = memref.reinterpret_cast %arg5 to offset: [%116], sizes: [32], strides: [%28] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %117 = affine.min #map10()[%37, %35, %76]
        %118 = affine.max #map3()[%117]
        %subview_164 = memref.subview %subview_95[0] [%118] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
        %subview_165 = memref.subview %reinterpret_cast_163[0] [%118] [1] : memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_166 = memref.cast %subview_164 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @store_ubuf_to_gm_1d_float(%cast_166, %subview_165, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %66]
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
