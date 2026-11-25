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
  func.func private @vln_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vabs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_max_ar_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmaxs_vs_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_sv_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vv_1d_bool_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vdiv_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_float_2d_with_mode(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmaxs_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmins_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_half_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_half(memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_act_quant_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c3_i32 = arith.constant 3 : i32
    %cst = arith.constant 0xFF800000 : f32
    %c2_i32 = arith.constant 2 : i32
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 0.693147182 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %cst_3 = arith.constant 0.00223214296 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_4 = arith.constant 4.480000e+02 : f32
    %cst_5 = arith.constant -4.480000e+02 : f32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_6 = arith.constant 9.99999974E-5 : f32
    %c67584_i64 = arith.constant 67584 : i64
    %c68608_i64 = arith.constant 68608 : i64
    %c84992_i64 = arith.constant 84992 : i64
    %c101376_i64 = arith.constant 101376 : i64
    %c102400_i64 = arith.constant 102400 : i64
    %c118784_i64 = arith.constant 118784 : i64
    %c119808_i64 = arith.constant 119808 : i64
    %c103424_i64 = arith.constant 103424 : i64
    %c104448_i64 = arith.constant 104448 : i64
    %c105472_i64 = arith.constant 105472 : i64
    %c102912_i64 = arith.constant 102912 : i64
    %c119296_i64 = arith.constant 119296 : i64
    %c135680_i64 = arith.constant 135680 : i64
    %c152064_i64 = arith.constant 152064 : i64
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
    call @broadcast_scalar_float_to_2d(%cst_6, %cast) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %14 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %14 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %cast_7 = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_5, %cast_7) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %15 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_8 = memref.collapse_shape %15 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %cast_9 = memref.cast %collapse_shape_8 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_4, %cast_9) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %16 = arith.divsi %arg7, %c128_i32 : i32
    %17 = arith.index_cast %arg7 : i32 to index
    %18 = arith.index_cast %arg6 : i32 to index
    %19 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_10 = memref.subview %19[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %subview_10 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    scf.for %arg11 = %c0 to %c32 step %c1 {
      %subview_57 = memref.subview %reinterpret_cast[%arg11, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_58 = memref.subview %reinterpret_cast[%arg11, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_59 = memref.subview %reinterpret_cast_15[%arg11, 0] [1, 1] [1, 1] : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %cast_60 = memref.cast %subview_57 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_61 = memref.cast %subview_58 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_62 = memref.cast %subview_59 : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_eq_1d_float(%cast_60, %cast_61, %cast_62) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    }
    %20 = hivm.hir.pointer_cast(%c67584_i64) : memref<32x256xi1, #hivm.address_space<ub>>
    %subview_16 = memref.subview %20[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
    %base_buffer_17, %offset_18, %sizes_19, %strides_20 = memref.extract_strided_metadata %subview_16 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_21 = memref.reinterpret_cast %base_buffer_17 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %21 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_22 = memref.cast %reinterpret_cast_15 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_23 = memref.cast %reinterpret_cast_21 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_24 = memref.cast %21 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_2d_bool(%cast_22, %cast_23, %cast_24) : (memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %22 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview_25 = memref.subview %22[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_26, %offset_27, %sizes_28, %strides_29 = memref.extract_strided_metadata %subview_25 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_30 = memref.reinterpret_cast %base_buffer_26 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    %cast_31 = memref.cast %reinterpret_cast_30 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_2d(%cst_1, %cast_31) : (f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %23 = hivm.hir.pointer_cast(%c101376_i64) : memref<32x8xf32, #hivm.address_space<ub>>
    %subview_32 = memref.subview %23[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
    %base_buffer_33, %offset_34, %sizes_35, %strides_36 = memref.extract_strided_metadata %subview_32 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_37 = memref.reinterpret_cast %base_buffer_33 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %24 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
    %cast_38 = memref.cast %reinterpret_cast_30 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_39 = memref.cast %reinterpret_cast_37 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_40 = memref.cast %24 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vln_2d_float(%cast_38, %cast_39, %cast_40) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %25 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_41 = memref.collapse_shape %25 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    %cast_42 = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_43 = memref.cast %collapse_shape : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_44 = memref.cast %collapse_shape_41 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_float(%cast_42, %cast_43, %cast_44) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %26 = hivm.hir.pointer_cast(%c68608_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_45 = memref.collapse_shape %26 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %27 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_46 = memref.cast %collapse_shape_41 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_47 = memref.cast %collapse_shape_45 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_48 = memref.cast %27 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_1d_bool(%cast_46, %cast_47, %cast_48) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %28 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_49 = memref.collapse_shape %28 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    %cast_50 = memref.cast %collapse_shape_8 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_51 = memref.cast %collapse_shape_8 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_52 = memref.cast %collapse_shape_49 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_float(%cast_50, %cast_51, %cast_52) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %29 = hivm.hir.pointer_cast(%c84992_i64) : memref<32x128xi1, #hivm.address_space<ub>>
    %collapse_shape_53 = memref.collapse_shape %29 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %30 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_54 = memref.cast %collapse_shape_49 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_55 = memref.cast %collapse_shape_53 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_56 = memref.cast %30 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_1d_bool(%cast_54, %cast_55, %cast_56) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %31 = arith.index_cast %16 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    scf.for %arg11 = %11 to %9 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %32 = arith.divsi %arg11, %arg10 : i32
      %33 = arith.remsi %32, %arg9 : i32
      %34 = arith.divsi %arg11, %12 : i32
      %35 = arith.remsi %34, %arg8 : i32
      %36 = arith.muli %35, %c32_i32 : i32
      %37 = arith.muli %33, %c4_i32 : i32
      %38 = arith.index_cast %36 : i32 to index
      %39 = affine.max #map()[%38, %18]
      %40 = affine.min #map1()[%39, %38]
      %41 = affine.min #map2()[%40, %38]
      %42 = affine.max #map3()[%41]
      scf.for %arg12 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
        %43 = arith.index_cast %arg12 : i32 to index
        %44 = affine.apply #map4()[%43]
        %45 = arith.index_cast %44 : index to i64
        %46 = arith.cmpi eq, %45, %c1_i64 : i64
        %47 = arith.select %46, %1, %0 : memref<32x128xf16, #hivm.address_space<ub>>
        %48 = arith.index_cast %arg12 : i32 to index
        %49 = affine.apply #map4()[%48]
        %50 = arith.index_cast %49 : index to i64
        %51 = arith.cmpi eq, %50, %c1_i64 : i64
        %52 = arith.select %51, %3, %2 : memref<32x8xf32, #hivm.address_space<ub>>
        %53 = arith.index_cast %arg12 : i32 to index
        %54 = affine.apply #map4()[%53]
        %55 = arith.index_cast %54 : index to i64
        %56 = arith.cmpi eq, %55, %c1_i64 : i64
        %57 = arith.select %56, %5, %4 : memref<32x128xf32, #hivm.address_space<ub>>
        %58 = arith.index_cast %arg12 : i32 to index
        %59 = affine.apply #map4()[%58]
        %60 = arith.index_cast %59 : index to i64
        %61 = arith.cmpi eq, %60, %c1_i64 : i64
        %62 = arith.select %61, %7, %6 : memref<32x128xf32, #hivm.address_space<ub>>
        %63 = arith.index_cast %arg12 : i32 to index
        %64 = affine.apply #map4()[%63]
        %65 = arith.index_cast %64 : index to i1
        %66 = arith.select %65, %c1_i64, %c2_i64 : i64
        %67 = arith.index_cast %64 : index to i1
        %68 = arith.xori %67, %true : i1
        %69 = arith.extui %68 : i1 to i64
        %70 = arith.addi %37, %arg12 : i32
        %71 = arith.cmpi slt, %70, %16 : i32
        %72 = arith.muli %70, %c128_i32 : i32
        %73 = arith.index_cast %72 : i32 to index
        %74 = affine.apply #map5()[%73, %38, %17]
        %reinterpret_cast_57 = memref.reinterpret_cast %arg3 to offset: [%74], sizes: [32, 128], strides: [%17, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %75 = affine.max #map()[%73, %17]
        %76 = affine.min #map6()[%75, %73]
        %77 = affine.min #map7()[%76, %73]
        %78 = affine.max #map3()[%77]
        %79 = arith.index_castui %71 : i1 to index
        %80 = affine.min #map8()[%42, %79]
        %81 = affine.max #map3()[%80]
        %82 = affine.min #map9()[%78, %79]
        %83 = affine.max #map3()[%82]
        %84 = arith.cmpi slt, %81, %c32 : index
        %85 = arith.cmpi slt, %83, %c128 : index
        %86 = arith.ori %84, %85 : i1
        %subview_58 = memref.subview %reinterpret_cast_57[0, 0] [%81, %83] [1, 1] : memref<32x128xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_59 = memref.subview %57[0, 0] [%81, %83] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>>
        scf.if %86 {
          %collapse_shape_230 = memref.collapse_shape %57 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
          %cast_231 = memref.cast %collapse_shape_230 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst_2, %cast_231) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %66]
        %cast_60 = memref.cast %subview_58 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_61 = memref.cast %subview_59 : memref<?x?xf32, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_2d_float(%cast_60, %cast_61, %c2_i32, %cst_2, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %87 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_62 = memref.collapse_shape %57 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %collapse_shape_63 = memref.collapse_shape %87 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %88 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_64 = memref.cast %collapse_shape_62 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_65 = memref.cast %collapse_shape_63 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_66 = memref.cast %88 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vabs_1d_float(%cast_64, %cast_65, %cast_66) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %89 = hivm.hir.pointer_cast(%c118784_i64) : memref<32x8x1xf32, #hivm.address_space<ub>>
        %subview_67 = memref.subview %89[0, 0, 0] [32, 1, 1] [1, 1, 1] : memref<32x8x1xf32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %90 = hivm.hir.pointer_cast(%c119808_i64) : memref<2048xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_68 = memref.cast %87 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_69 = memref.cast %subview_67 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_70 = memref.cast %90 : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @enablevc_reduce_max_ar_float(%cast_68, %cast_69, %cast_70, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        %collapse_shape_71 = memref.collapse_shape %subview_67 [[0, 1]] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %91 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x256xi1, #hivm.address_space<ub>>
        %subview_72 = memref.subview %91[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
        %base_buffer_73, %offset_74, %sizes_75, %strides_76 = memref.extract_strided_metadata %collapse_shape_71 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_77 = memref.reinterpret_cast %base_buffer_73 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %base_buffer_78, %offset_79, %sizes_80, %strides_81 = memref.extract_strided_metadata %subview_72 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_82 = memref.reinterpret_cast %base_buffer_78 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        scf.for %arg13 = %c0 to %c32 step %c1 {
          %subview_230 = memref.subview %reinterpret_cast_77[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_231 = memref.subview %reinterpret_cast_77[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_232 = memref.subview %reinterpret_cast_82[%arg13, 0] [1, 1] [1, 1] : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %cast_233 = memref.cast %subview_230 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_234 = memref.cast %subview_231 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_235 = memref.cast %subview_232 : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_eq_1d_float(%cast_233, %cast_234, %cast_235) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        }
        %92 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x256xi1, #hivm.address_space<ub>>
        %subview_83 = memref.subview %92[0, 0] [32, 1] [1, 1] : memref<32x256xi1, #hivm.address_space<ub>> to memref<32xi1, strided<[256]>, #hivm.address_space<ub>>
        %base_buffer_84, %offset_85, %sizes_86, %strides_87 = memref.extract_strided_metadata %subview_83 : memref<32xi1, strided<[256]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_88 = memref.reinterpret_cast %base_buffer_84 to offset: [0], sizes: [32, 1], strides: [256, 1] : memref<i1, #hivm.address_space<ub>> to memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %93 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_89 = memref.cast %reinterpret_cast_82 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_90 = memref.cast %reinterpret_cast_88 : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_91 = memref.cast %93 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_2d_bool(%cast_89, %cast_90, %cast_91) : (memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %94 = hivm.hir.pointer_cast(%c103424_i64) : memref<32x8xf32, #hivm.address_space<ub>>
        %subview_92 = memref.subview %94[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_93, %offset_94, %sizes_95, %strides_96 = memref.extract_strided_metadata %subview_92 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_97 = memref.reinterpret_cast %base_buffer_93 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %95 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_98 = memref.cast %reinterpret_cast_77 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_99 = memref.cast %reinterpret_cast_97 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_100 = memref.cast %95 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmaxs_vs_2d_float(%cast_98, %cst_6, %cast_99, %cast_100) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %96 = hivm.hir.pointer_cast(%c104448_i64) : memref<32x8xf32, #hivm.address_space<ub>>
        %subview_101 = memref.subview %96[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_102, %offset_103, %sizes_104, %strides_105 = memref.extract_strided_metadata %subview_101 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_106 = memref.reinterpret_cast %base_buffer_102 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %97 = hivm.hir.pointer_cast(%c105472_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        scf.for %arg13 = %c0 to %c32 step %c1 {
          %subview_230 = memref.subview %reinterpret_cast_88[%arg13, 0] [1, 1] [1, 1] : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_231 = memref.subview %reinterpret_cast_97[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_232 = memref.subview %reinterpret_cast_106[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %cast_233 = memref.cast %subview_230 : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_234 = memref.cast %subview_231 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_235 = memref.cast %subview_232 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_236 = memref.cast %97 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_sv_1d_float(%cast_233, %cst_6, %cast_234, %cast_235, %cast_236) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        }
        %subview_107 = memref.subview %52[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_108, %offset_109, %sizes_110, %strides_111 = memref.extract_strided_metadata %subview_107 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_112 = memref.reinterpret_cast %base_buffer_108 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        %98 = hivm.hir.pointer_cast(%c102400_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %69]
        hivm.hir.pipe_barrier[<PIPE_V>]
        scf.for %arg13 = %c0 to %c32 step %c1 {
          %subview_230 = memref.subview %reinterpret_cast_21[%arg13, 0] [1, 1] [1, 1] : memref<32x1xi1, strided<[256, 1]>, #hivm.address_space<ub>> to memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_231 = memref.subview %reinterpret_cast_77[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_232 = memref.subview %reinterpret_cast_106[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_233 = memref.subview %reinterpret_cast_112[%arg13, 0] [1, 1] [1, 1] : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %cast_234 = memref.cast %subview_230 : memref<1xi1, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_235 = memref.cast %subview_231 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_236 = memref.cast %subview_232 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_237 = memref.cast %subview_233 : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_238 = memref.cast %98 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_vv_1d_bool_float(%cast_234, %cast_235, %cast_236, %cast_237, %cast_238) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        }
        %subview_113 = memref.subview %52[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_114, %offset_115, %sizes_116, %strides_117 = memref.extract_strided_metadata %subview_113 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_118 = memref.reinterpret_cast %base_buffer_114 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %99 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_119 = memref.cast %reinterpret_cast_112 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_120 = memref.cast %reinterpret_cast_118 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_121 = memref.cast %99 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_2d_float(%cast_119, %cst_3, %cast_120, %cast_121) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %subview_122 = memref.subview %52[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_123, %offset_124, %sizes_125, %strides_126 = memref.extract_strided_metadata %subview_122 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_127 = memref.reinterpret_cast %base_buffer_123 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %100 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_128 = memref.cast %reinterpret_cast_118 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_129 = memref.cast %reinterpret_cast_127 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_130 = memref.cast %100 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vln_2d_float(%cast_128, %cast_129, %cast_130) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %subview_131 = memref.subview %52[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_132, %offset_133, %sizes_134, %strides_135 = memref.extract_strided_metadata %subview_131 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_136 = memref.reinterpret_cast %base_buffer_132 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %101 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_137 = memref.cast %reinterpret_cast_127 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_138 = memref.cast %reinterpret_cast_37 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_139 = memref.cast %reinterpret_cast_136 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_140 = memref.cast %101 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vdiv_2d_float(%cast_137, %cast_138, %cast_139, %cast_140) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %subview_141 = memref.subview %52[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_142, %offset_143, %sizes_144, %strides_145 = memref.extract_strided_metadata %subview_141 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_146 = memref.reinterpret_cast %base_buffer_142 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_147 = memref.cast %reinterpret_cast_136 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_148 = memref.cast %reinterpret_cast_146 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_float_to_float_2d_with_mode(%cast_147, %cast_148, %c3_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        %subview_149 = memref.subview %52[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_150, %offset_151, %sizes_152, %strides_153 = memref.extract_strided_metadata %subview_149 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_154 = memref.reinterpret_cast %base_buffer_150 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %102 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_155 = memref.cast %reinterpret_cast_146 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_156 = memref.cast %reinterpret_cast_154 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_157 = memref.cast %102 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_2d_float(%cast_155, %cst_0, %cast_156, %cast_157) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %subview_158 = memref.subview %52[0, 0] [32, 1] [1, 1] : memref<32x8xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[8]>, #hivm.address_space<ub>>
        %base_buffer_159, %offset_160, %sizes_161, %strides_162 = memref.extract_strided_metadata %subview_158 : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
        %reinterpret_cast_163 = memref.reinterpret_cast %base_buffer_159 to offset: [0], sizes: [32, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %103 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_164 = memref.cast %reinterpret_cast_154 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_165 = memref.cast %reinterpret_cast_163 : memref<32x1xf32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_166 = memref.cast %103 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vexp_2d_float(%cast_164, %cast_165, %cast_166) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape = memref.expand_shape %subview_158 [[0, 1]] output_shape [32, 1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> into memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>>
        %104 = hivm.hir.pointer_cast(%c102400_i64) : memref<256xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_167 = memref.cast %57 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_168 = memref.cast %expand_shape : memref<32x1xf32, strided<[8, 8]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_169 = memref.cast %57 : memref<32x128xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_170 = memref.cast %104 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vdiv_2d_float(%cast_167, %cast_168, %cast_169, %cast_170) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %105 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xi1, #hivm.address_space<ub>>
        %collapse_shape_171 = memref.collapse_shape %57 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %collapse_shape_172 = memref.collapse_shape %105 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_173 = memref.cast %collapse_shape_171 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_174 = memref.cast %collapse_shape_171 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_175 = memref.cast %collapse_shape_172 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_float(%cast_173, %cast_174, %cast_175) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %106 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xi1, #hivm.address_space<ub>>
        %collapse_shape_176 = memref.collapse_shape %106 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %107 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_177 = memref.cast %collapse_shape_172 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_178 = memref.cast %collapse_shape_176 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_179 = memref.cast %107 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_177, %cast_178, %cast_179) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %108 = hivm.hir.pointer_cast(%c102912_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_180 = memref.collapse_shape %108 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %109 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_181 = memref.cast %collapse_shape_171 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_182 = memref.cast %collapse_shape_180 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_183 = memref.cast %109 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmaxs_vs_1d_float(%cast_181, %cst_5, %cast_182, %cast_183) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %110 = hivm.hir.pointer_cast(%c119296_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_184 = memref.collapse_shape %110 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %111 = hivm.hir.pointer_cast(%c135680_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_185 = memref.cast %collapse_shape_176 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_186 = memref.cast %collapse_shape_180 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_187 = memref.cast %collapse_shape_184 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_188 = memref.cast %111 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_sv_1d_float(%cast_185, %cst_5, %cast_186, %cast_187, %cast_188) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %112 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_189 = memref.collapse_shape %112 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %113 = hivm.hir.pointer_cast(%c118784_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_190 = memref.cast %collapse_shape_45 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_191 = memref.cast %collapse_shape_171 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_192 = memref.cast %collapse_shape_184 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_193 = memref.cast %collapse_shape_189 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_194 = memref.cast %113 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_vv_1d_bool_float(%cast_190, %cast_191, %cast_192, %cast_193, %cast_194) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %66]
        %114 = hivm.hir.pointer_cast(%c118784_i64) : memref<32x128xi1, #hivm.address_space<ub>>
        %collapse_shape_195 = memref.collapse_shape %114 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_196 = memref.cast %collapse_shape_189 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_197 = memref.cast %collapse_shape_189 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_198 = memref.cast %collapse_shape_195 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcmp_eq_1d_float(%cast_196, %cast_197, %cast_198) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %115 = hivm.hir.pointer_cast(%c118784_i64) : memref<32x128xi1, #hivm.address_space<ub>>
        %collapse_shape_199 = memref.collapse_shape %115 [[0, 1]] : memref<32x128xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %116 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
        %cast_200 = memref.cast %collapse_shape_195 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_201 = memref.cast %collapse_shape_199 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_202 = memref.cast %116 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vnot_1d_bool(%cast_200, %cast_201, %cast_202) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %117 = hivm.hir.pointer_cast(%c119296_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_203 = memref.collapse_shape %117 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %118 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_204 = memref.cast %collapse_shape_189 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_205 = memref.cast %collapse_shape_203 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_206 = memref.cast %118 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmins_vs_1d_float(%cast_204, %cst_4, %cast_205, %cast_206) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %119 = hivm.hir.pointer_cast(%c135680_i64) : memref<32x128xf32, #hivm.address_space<ub>>
        %collapse_shape_207 = memref.collapse_shape %119 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %120 = hivm.hir.pointer_cast(%c152064_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_208 = memref.cast %collapse_shape_199 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_209 = memref.cast %collapse_shape_203 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_210 = memref.cast %collapse_shape_207 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_211 = memref.cast %120 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_sv_1d_float(%cast_208, %cst_4, %cast_209, %cast_210, %cast_211) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %collapse_shape_212 = memref.collapse_shape %62 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        %121 = hivm.hir.pointer_cast(%c118784_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_213 = memref.cast %collapse_shape_53 : memref<4096xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_214 = memref.cast %collapse_shape_189 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_215 = memref.cast %collapse_shape_207 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_216 = memref.cast %collapse_shape_212 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_217 = memref.cast %121 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vsel_vv_1d_bool_float(%cast_213, %cast_214, %cast_215, %cast_216, %cast_217) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %reinterpret_cast_218 = memref.reinterpret_cast %arg4 to offset: [%74], sizes: [32, 128], strides: [%17, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_219 = memref.collapse_shape %47 [[0, 1]] : memref<32x128xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_220 = memref.cast %collapse_shape_212 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_221 = memref.cast %collapse_shape_219 : memref<4096xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vcast_float_to_half_1d_with_mode(%cast_220, %cast_221, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %subview_222 = memref.subview %47[0, 0] [%81, %83] [1, 1] : memref<32x128xf16, #hivm.address_space<ub>> to memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>>
        %subview_223 = memref.subview %reinterpret_cast_218[0, 0] [%81, %83] [1, 1] : memref<32x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_224 = memref.cast %subview_222 : memref<?x?xf16, strided<[128, 1]>, #hivm.address_space<ub>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_225 = memref.cast %subview_223 : memref<?x?xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_half(%cast_224, %cast_225, %c0_i32) : (memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        %122 = arith.index_cast %70 : i32 to index
        %123 = affine.apply #map5()[%122, %38, %31]
        %reinterpret_cast_226 = memref.reinterpret_cast %arg5 to offset: [%123], sizes: [32], strides: [%31] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %124 = affine.min #map10()[%40, %38, %79]
        %125 = affine.max #map3()[%124]
        %subview_227 = memref.subview %subview_158[0] [%125] [1] : memref<32xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
        %subview_228 = memref.subview %reinterpret_cast_226[0] [%125] [1] : memref<32xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_229 = memref.cast %subview_227 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @store_ubuf_to_gm_1d_float(%cast_229, %subview_228, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %69]
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
