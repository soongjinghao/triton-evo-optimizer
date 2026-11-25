#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0] -> (0, s0)>
#map2 = affine_map<()[s0] -> (512, s0)>
#map3 = affine_map<()[s0] -> (s0 + 1)>
#map4 = affine_map<()[s0] -> (s0 + 2)>
#map5 = affine_map<()[s0] -> (s0 + 3)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @fill_accepted_out_cache_loc_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_int64_t_to_1d(i64, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_int64_t(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_int8_t_to_half_1d_with_mode(memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_half_to_1d(f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_ne_1d_half(memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bool_to_float_1d_with_temp(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_float_to_int64_t_1d_with_mode(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_int64_t(memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @fill_accepted_out_cache_loc(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant 4 : i32
    %c-1_i64 = arith.constant -1 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %c0_i64 = arith.constant 0 : i64
    %c9216_i64 = arith.constant 9216 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c10240_i64 = arith.constant 10240 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c12384_i64 = arith.constant 12384 : i64
    %c16480_i64 = arith.constant 16480 : i64
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %c16640_i64 = arith.constant 16640 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c20736_i64 = arith.constant 20736 : i64
    %c16512_i64 = arith.constant 16512 : i64
    %c20768_i64 = arith.constant 20768 : i64
    %c16544_i64 = arith.constant 16544 : i64
    %c20800_i64 = arith.constant 20800 : i64
    %c16576_i64 = arith.constant 16576 : i64
    %c16608_i64 = arith.constant 16608 : i64
    %c20832_i64 = arith.constant 20832 : i64
    %0 = hivm.hir.pointer_cast(%c16608_i64) : memref<1xi64, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c20832_i64) : memref<1xi64, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c16576_i64) : memref<1xi64, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c20800_i64) : memref<1xi64, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c16544_i64) : memref<1xi64, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c20768_i64) : memref<1xi64, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c16512_i64) : memref<1xi64, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c20736_i64) : memref<1xi64, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c4096_i64) : memref<512xi64, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c16640_i64) : memref<512xi64, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c4096_i64) : memref<512xi8, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c16640_i64) : memref<512xi8, #hivm.address_space<ub>>
    %12 = arith.muli %arg6, %arg7 : i32
    %13 = arith.muli %12, %arg8 : i32
    %14 = hivm.hir.get_block_idx -> i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = arith.muli %arg8, %arg7 : i32
    %17 = hivm.hir.pointer_cast(%c0_i64) : memref<512xi64, #hivm.address_space<ub>>
    %cast = memref.cast %17 : memref<512xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_int64_t_to_1d(%c-1_i64, %cast) : (i64, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [0], sizes: [512], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<512xi64, strided<[1]>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID6>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID7>]
    scf.for %arg9 = %15 to %13 step %c48_i32  : i32 {
      %18 = arith.index_cast %arg9 : i32 to index
      %19 = arith.index_cast %15 : i32 to index
      %20 = affine.apply #map()[%18, %19]
      %21 = arith.index_cast %20 : index to i64
      %22 = arith.cmpi eq, %21, %c1_i64 : i64
      %23 = arith.select %22, %1, %0 : memref<1xi64, #hivm.address_space<ub>>
      %24 = arith.index_cast %arg9 : i32 to index
      %25 = arith.index_cast %15 : i32 to index
      %26 = affine.apply #map()[%24, %25]
      %27 = arith.index_cast %26 : index to i64
      %28 = arith.cmpi eq, %27, %c1_i64 : i64
      %29 = arith.select %28, %3, %2 : memref<1xi64, #hivm.address_space<ub>>
      %30 = arith.index_cast %arg9 : i32 to index
      %31 = arith.index_cast %15 : i32 to index
      %32 = affine.apply #map()[%30, %31]
      %33 = arith.index_cast %32 : index to i64
      %34 = arith.cmpi eq, %33, %c1_i64 : i64
      %35 = arith.select %34, %5, %4 : memref<1xi64, #hivm.address_space<ub>>
      %36 = arith.index_cast %arg9 : i32 to index
      %37 = arith.index_cast %15 : i32 to index
      %38 = affine.apply #map()[%36, %37]
      %39 = arith.index_cast %38 : index to i64
      %40 = arith.cmpi eq, %39, %c1_i64 : i64
      %41 = arith.select %40, %7, %6 : memref<1xi64, #hivm.address_space<ub>>
      %42 = arith.index_cast %arg9 : i32 to index
      %43 = arith.index_cast %15 : i32 to index
      %44 = affine.apply #map()[%42, %43]
      %45 = arith.index_cast %44 : index to i64
      %46 = arith.cmpi eq, %45, %c1_i64 : i64
      %47 = arith.select %46, %9, %8 : memref<512xi64, #hivm.address_space<ub>>
      %48 = arith.index_cast %arg9 : i32 to index
      %49 = arith.index_cast %15 : i32 to index
      %50 = affine.apply #map()[%48, %49]
      %51 = arith.index_cast %50 : index to i64
      %52 = arith.cmpi eq, %51, %c1_i64 : i64
      %53 = arith.select %52, %11, %10 : memref<512xi8, #hivm.address_space<ub>>
      %54 = arith.index_cast %arg9 : i32 to index
      %55 = arith.index_cast %15 : i32 to index
      %56 = affine.apply #map()[%54, %55]
      %57 = arith.index_cast %56 : index to i1
      %58 = arith.xori %57, %true : i1
      %59 = arith.extui %58 : i1 to i64
      %60 = arith.index_cast %56 : index to i1
      %61 = arith.select %60, %c2_i64, %c3_i64 : i64
      %62 = arith.index_cast %56 : index to i1
      %63 = arith.select %62, %c4_i64, %c5_i64 : i64
      %64 = arith.index_cast %56 : index to i1
      %65 = arith.select %64, %c6_i64, %c7_i64 : i64
      hivm.hir.set_mask_norm
      %66 = arith.divsi %arg9, %16 : i32
      %67 = arith.remsi %66, %arg6 : i32
      %68 = arith.muli %67, %c4_i32 : i32
      %69 = arith.index_cast %68 : i32 to index
      %70 = affine.max #map1()[%69]
      %71 = affine.min #map2()[%70]
      %72 = arith.cmpi slt, %71, %c512 : index
      %subview = memref.subview %reinterpret_cast[0] [%71] [1] : memref<512xi64, strided<[1]>, #hivm.address_space<gm>> to memref<?xi64, strided<[1]>, #hivm.address_space<gm>>
      %subview_0 = memref.subview %47[0] [%71] [1] : memref<512xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1]>, #hivm.address_space<ub>>
      scf.if %72 {
        %cast_21 = memref.cast %47 : memref<512xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_int64_t_to_1d(%c-1_i64, %cast_21) : (i64, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_1 = memref.cast %subview : memref<?xi64, strided<[1]>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_2 = memref.cast %subview_0 : memref<?xi64, strided<[1]>, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_int64_t(%cast_1, %cast_2, %c2_i32, %c-1_i64, %c0) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i64, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %73 = hivm.hir.pointer_cast(%c9216_i64) : memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg10 = %c0 to %c512 step %c1 {
        %94 = memref.load %47[%arg10] : memref<512xi64, #hivm.address_space<ub>>
        %95 = memref.load %17[%arg10] : memref<512xi64, #hivm.address_space<ub>>
        %96 = arith.cmpi eq, %94, %95 : i64
        %97 = arith.extui %96 : i1 to i8
        memref.store %97, %53[%arg10] : memref<512xi8, #hivm.address_space<ub>>
      }
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %74 = hivm.hir.pointer_cast(%c8192_i64) : memref<512xf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %cast_3 = memref.cast %53 : memref<512xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_4 = memref.cast %74 : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_int8_t_to_half_1d_with_mode(%cast_3, %cast_4, %c0_i32) : (memref<?xi8, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %75 = hivm.hir.pointer_cast(%c9216_i64) : memref<512xf16, #hivm.address_space<ub>>
      %cast_5 = memref.cast %75 : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @broadcast_scalar_half_to_1d(%cst, %cast_5) : (f16, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_6 = memref.cast %74 : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_7 = memref.cast %75 : memref<512xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_8 = memref.cast %73 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcmp_ne_1d_half(%cast_6, %cast_7, %cast_8) : (memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %76 = hivm.hir.pointer_cast(%c9216_i64) : memref<512xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %77 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
      %cast_9 = memref.cast %73 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_10 = memref.cast %76 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_11 = memref.cast %77 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vnot_1d_bool(%cast_9, %cast_10, %cast_11) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %78 = hivm.hir.pointer_cast(%c10240_i64) : memref<512xf32, #hivm.address_space<ub>>
      %79 = hivm.hir.pointer_cast(%c12288_i64) : memref<24xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_12 = memref.cast %76 : memref<512xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %78 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %79 : memref<24xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_bool_to_float_1d_with_temp(%cast_12, %cast_13, %cast_14, %c0_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      %80 = hivm.hir.pointer_cast(%c12384_i64) : memref<512xi64, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_15 = memref.cast %78 : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_16 = memref.cast %80 : memref<512xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vcast_float_to_int64_t_1d_with_mode(%cast_15, %cast_16, %c4_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %81 = hivm.hir.pointer_cast(%c16480_i64) : memref<1xi64, #hivm.address_space<ub>>
      memref.store %c0_i64, %81[%c0] : memref<1xi64, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg10 = %c0 to %c512 step %c1 {
        %94 = memref.load %80[%arg10] : memref<512xi64, #hivm.address_space<ub>>
        %95 = arith.cmpi eq, %arg10, %c0 : index
        scf.if %95 {
          memref.store %c0_i64, %81[%c0] : memref<1xi64, #hivm.address_space<ub>>
        }
        %96 = memref.load %81[%c0] : memref<1xi64, #hivm.address_space<ub>>
        %97 = arith.addi %94, %96 : i64
        memref.store %97, %81[%c0] : memref<1xi64, #hivm.address_space<ub>>
      }
      %82 = memref.load %81[%c0] : memref<1xi64, #hivm.address_space<ub>>
      %reinterpret_cast_17 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %83 = memref.load %reinterpret_cast_17[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %84 = arith.cmpi sgt, %83, %c-1_i64 : i64
      scf.if %84 {
        %94 = arith.index_cast %83 : i64 to index
        %reinterpret_cast_21 = memref.reinterpret_cast %arg4 to offset: [%94], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %95 = memref.load %reinterpret_cast_21[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %65]
        memref.store %95, %41[%c0] : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %96 = arith.index_cast %82 : i64 to index
        %reinterpret_cast_22 = memref.reinterpret_cast %arg5 to offset: [%96], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_23 = memref.cast %41 : memref<1xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_24 = memref.cast %reinterpret_cast_22 : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int64_t(%cast_23, %cast_24, %c0_i32) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %65]
      } {pipeline.veconly}
      %85 = affine.apply #map3()[%69]
      %reinterpret_cast_18 = memref.reinterpret_cast %arg3 to offset: [%85], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %86 = memref.load %reinterpret_cast_18[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %87 = arith.cmpi sgt, %86, %c-1_i64 : i64
      scf.if %87 {
        %94 = arith.index_cast %86 : i64 to index
        %reinterpret_cast_21 = memref.reinterpret_cast %arg4 to offset: [%94], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %95 = memref.load %reinterpret_cast_21[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %63]
        memref.store %95, %35[%c0] : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %96 = arith.index_cast %82 : i64 to index
        %97 = affine.apply #map3()[%96]
        %reinterpret_cast_22 = memref.reinterpret_cast %arg5 to offset: [%97], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_23 = memref.cast %35 : memref<1xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_24 = memref.cast %reinterpret_cast_22 : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int64_t(%cast_23, %cast_24, %c0_i32) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %63]
      } {pipeline.veconly}
      %88 = affine.apply #map4()[%69]
      %reinterpret_cast_19 = memref.reinterpret_cast %arg3 to offset: [%88], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %89 = memref.load %reinterpret_cast_19[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %90 = arith.cmpi sgt, %89, %c-1_i64 : i64
      scf.if %90 {
        %94 = arith.index_cast %89 : i64 to index
        %reinterpret_cast_21 = memref.reinterpret_cast %arg4 to offset: [%94], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %95 = memref.load %reinterpret_cast_21[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %61]
        memref.store %95, %29[%c0] : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %96 = arith.index_cast %82 : i64 to index
        %97 = affine.apply #map4()[%96]
        %reinterpret_cast_22 = memref.reinterpret_cast %arg5 to offset: [%97], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_23 = memref.cast %29 : memref<1xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_24 = memref.cast %reinterpret_cast_22 : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int64_t(%cast_23, %cast_24, %c0_i32) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %61]
      } {pipeline.veconly}
      %91 = affine.apply #map5()[%69]
      %reinterpret_cast_20 = memref.reinterpret_cast %arg3 to offset: [%91], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %92 = memref.load %reinterpret_cast_20[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %93 = arith.cmpi sgt, %92, %c-1_i64 : i64
      scf.if %93 {
        %94 = arith.index_cast %92 : i64 to index
        %reinterpret_cast_21 = memref.reinterpret_cast %arg4 to offset: [%94], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %95 = memref.load %reinterpret_cast_21[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %59]
        memref.store %95, %23[%c0] : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %96 = arith.index_cast %82 : i64 to index
        %97 = affine.apply #map5()[%96]
        %reinterpret_cast_22 = memref.reinterpret_cast %arg5 to offset: [%97], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_23 = memref.cast %23 : memref<1xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_24 = memref.cast %reinterpret_cast_22 : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_int64_t(%cast_23, %cast_24, %c0_i32) : (memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi64, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %59]
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
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
