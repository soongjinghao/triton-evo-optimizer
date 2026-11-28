#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 48) mod 2)>
#map1 = affine_map<()[s0] -> (0, s0)>
#map2 = affine_map<()[s0] -> (2, s0)>
#map3 = affine_map<()[s0] -> (s0 * 16)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @reduce_segments_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_max_r_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_sum_r_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_sum_ra_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vdiv_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @reduce_segments(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32, %arg9: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg10: i64 {tt.divisibility = 16 : i32}, %arg11: i64 {tt.divisibility = 16 : i32}, %arg12: i64 {tt.divisibility = 16 : i32}, %arg13: memref<?xi32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, true, false, false, false, true, false, false, false]> : vector<17xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0xFF800000 : f32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c15_i32 = arith.constant 15 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8_i64 = arith.constant 8 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c128_i64 = arith.constant 128 : i64
    %c32_i32 = arith.constant 32 : i32
    %c0_i64 = arith.constant 0 : i64
    %c160_i64 = arith.constant 160 : i64
    %c192_i64 = arith.constant 192 : i64
    %c352_i64 = arith.constant 352 : i64
    %c672_i64 = arith.constant 672 : i64
    %c800_i64 = arith.constant 800 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c832_i64 = arith.constant 832 : i64
    %c64_i64 = arith.constant 64 : i64
    %c864_i64 = arith.constant 864 : i64
    %c96_i64 = arith.constant 96 : i64
    %c896_i64 = arith.constant 896 : i64
    %c224_i64 = arith.constant 224 : i64
    %c608_i64 = arith.constant 608 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %0 = hivm.hir.pointer_cast(%c608_i64) : memref<16xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c1024_i64) : memref<16xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c224_i64) : memref<2x16xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c896_i64) : memref<2x16xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c96_i64) : memref<2xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c864_i64) : memref<2xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c64_i64) : memref<2xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c832_i64) : memref<2xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c608_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c1024_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %10 = arith.muli %arg14, %arg15 : i32
    %11 = arith.muli %10, %arg16 : i32
    %12 = hivm.hir.get_block_idx -> i64
    %13 = arith.trunci %12 : i64 to i32
    %14 = arith.muli %arg16, %arg15 : i32
    %15 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
    %cast = memref.cast %15 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst_0, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    scf.for %arg17 = %13 to %11 step %c48_i32  : i32 {
      %16 = arith.index_cast %arg17 : i32 to index
      %17 = arith.index_cast %13 : i32 to index
      %18 = affine.apply #map()[%16, %17]
      %19 = arith.index_cast %18 : index to i64
      %20 = arith.cmpi eq, %19, %c1_i64 : i64
      %21 = arith.select %20, %1, %0 : memref<16xf32, #hivm.address_space<ub>>
      %22 = arith.index_cast %arg17 : i32 to index
      %23 = arith.index_cast %13 : i32 to index
      %24 = affine.apply #map()[%22, %23]
      %25 = arith.index_cast %24 : index to i64
      %26 = arith.cmpi eq, %25, %c1_i64 : i64
      %27 = arith.select %26, %3, %2 : memref<2x16xf32, #hivm.address_space<ub>>
      %28 = arith.index_cast %arg17 : i32 to index
      %29 = arith.index_cast %13 : i32 to index
      %30 = affine.apply #map()[%28, %29]
      %31 = arith.index_cast %30 : index to i64
      %32 = arith.cmpi eq, %31, %c1_i64 : i64
      %33 = arith.select %32, %5, %4 : memref<2xf32, #hivm.address_space<ub>>
      %34 = arith.index_cast %arg17 : i32 to index
      %35 = arith.index_cast %13 : i32 to index
      %36 = affine.apply #map()[%34, %35]
      %37 = arith.index_cast %36 : index to i64
      %38 = arith.cmpi eq, %37, %c1_i64 : i64
      %39 = arith.select %38, %7, %6 : memref<2xf32, #hivm.address_space<ub>>
      %40 = arith.index_cast %arg17 : i32 to index
      %41 = arith.index_cast %13 : i32 to index
      %42 = affine.apply #map()[%40, %41]
      %43 = arith.index_cast %42 : index to i64
      %44 = arith.cmpi eq, %43, %c1_i64 : i64
      %45 = arith.select %44, %9, %8 : memref<1x16xf32, #hivm.address_space<ub>>
      %46 = arith.index_cast %arg17 : i32 to index
      %47 = arith.index_cast %13 : i32 to index
      %48 = affine.apply #map()[%46, %47]
      %49 = arith.index_cast %48 : index to i1
      %50 = arith.select %49, %c1_i64, %c2_i64 : i64
      %51 = arith.index_cast %48 : index to i1
      %52 = arith.xori %51, %true : i1
      %53 = arith.extui %52 : i1 to i64
      hivm.hir.set_mask_norm
      %54 = arith.divsi %arg17, %arg16 : i32
      %55 = arith.remsi %54, %arg15 : i32
      %56 = arith.divsi %arg17, %14 : i32
      %57 = arith.remsi %56, %arg14 : i32
      %58:2 = scf.while (%arg18 = %c0_i32, %arg19 = %arg8) : (i32, i32) -> (i32, i32) {
        %104 = arith.cmpi slt, %arg18, %arg19 : i32
        scf.condition(%104) %arg18, %arg19 : i32, i32
      } do {
      ^bb0(%arg18: i32, %arg19: i32):
        %104 = arith.addi %arg18, %arg19 : i32
        %105 = arith.shrsi %104, %c1_i32 : i32
        %106 = arith.index_cast %105 : i32 to index
        %reinterpret_cast_50 = memref.reinterpret_cast %arg13 to offset: [%106], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %107 = memref.load %reinterpret_cast_50[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %108 = arith.cmpi sle, %107, %57 : i32
        %109 = arith.addi %105, %c1_i32 : i32
        %110 = arith.select %108, %109, %arg18 : i32
        %111 = arith.select %108, %arg19, %105 : i32
        scf.yield %110, %111 : i32, i32
      }
      %59 = arith.subi %58#0, %c1_i32 : i32
      %60 = arith.index_cast %59 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%60], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %61 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %62 = arith.addi %61, %c15_i32 : i32
      %63 = arith.divsi %62, %c16_i32 : i32
      %64 = arith.muli %63, %c8_i32 : i32
      %65 = arith.addi %61, %64 : i32
      %66 = arith.subi %65, %c1_i32 : i32
      %67 = arith.divsi %66, %64 : i32
      %68 = arith.extsi %57 : i32 to i64
      %69 = arith.muli %68, %c8_i64 : i64
      %70 = arith.muli %55, %c2_i32 : i32
      %71 = arith.extsi %70 : i32 to i64
      %72 = arith.addi %69, %71 : i64
      %73 = arith.index_cast %72 : i64 to index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%73], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %74 = arith.index_cast %67 : i32 to index
      %75 = affine.max #map1()[%74]
      %76 = affine.min #map2()[%75]
      %77 = arith.cmpi slt, %76, %c2 : index
      %subview = memref.subview %reinterpret_cast_1[0] [%76] [1] : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_2 = memref.subview %39[0] [%76] [1] : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %77 {
        %cast_50 = memref.cast %39 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst, %cast_50) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %50]
      %cast_3 = memref.cast %subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_4 = memref.cast %subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_3, %cast_4, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg6 to offset: [%73], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %reinterpret_cast_5[0] [%76] [1] : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_7 = memref.subview %33[0] [%76] [1] : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %77 {
        %cast_50 = memref.cast %33 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_50) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %cast_8 = memref.cast %subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_9 = memref.cast %subview_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_8, %cast_9, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %78 = hivm.hir.pointer_cast(%c128_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
      %79 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_10 = memref.cast %39 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_11 = memref.cast %78 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_12 = memref.cast %79 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @enablevc_reduce_max_r_float(%cast_10, %cast_11, %cast_12, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %80 = memref.load %78[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %81 = arith.subf %cst_0, %80 : f32
      %82 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_13 = memref.cast %39 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_14 = memref.cast %39 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_15 = memref.cast %82 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_13, %81, %cast_14, %cast_15) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %83 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_16 = memref.cast %39 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_17 = memref.cast %39 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_18 = memref.cast %83 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_16, %cast_17, %cast_18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %84 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_19 = memref.cast %33 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %39 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_21 = memref.cast %33 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_22 = memref.cast %84 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_1d_float(%cast_19, %cast_20, %cast_21, %cast_22) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %85 = hivm.hir.pointer_cast(%c160_i64) : memref<1xf32, #hivm.address_space<ub>>
      %86 = hivm.hir.pointer_cast(%c192_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_23 = memref.cast %33 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_24 = memref.cast %85 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_25 = memref.cast %86 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @enablevc_reduce_sum_r_float(%cast_23, %cast_24, %cast_25, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %87 = memref.load %85[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      %88 = arith.muli %68, %c128_i64 : i64
      %89 = arith.muli %55, %c32_i32 : i32
      %90 = arith.extsi %89 : i32 to i64
      %91 = arith.addi %88, %90 : i64
      %92 = arith.index_cast %91 : i64 to index
      %reinterpret_cast_26 = memref.reinterpret_cast %arg4 to offset: [%92], sizes: [2, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<2x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_27 = memref.subview %reinterpret_cast_26[0, 0] [%76, 16] [1, 1] : memref<2x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_28 = memref.subview %27[0, 0] [%76, 16] [1, 1] : memref<2x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
      scf.if %77 {
        %collapse_shape_50 = memref.collapse_shape %27 [[0, 1]] : memref<2x16xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
        %cast_51 = memref.cast %collapse_shape_50 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_51) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %collapse_shape = memref.collapse_shape %subview_27 [[0, 1]] : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %collapse_shape_29 = memref.collapse_shape %subview_28 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %cast_30 = memref.cast %collapse_shape : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_31 = memref.cast %collapse_shape_29 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_float(%cast_30, %cast_31, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %expand_shape = memref.expand_shape %39 [[0, 1]] output_shape [2, 1] : memref<2xf32, #hivm.address_space<ub>> into memref<2x1xf32, #hivm.address_space<ub>>
      %93 = hivm.hir.pointer_cast(%c352_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_32 = memref.cast %27 : memref<2x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_33 = memref.cast %expand_shape : memref<2x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_34 = memref.cast %27 : memref<2x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_35 = memref.cast %93 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_32, %cast_33, %cast_34, %cast_35) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %50]
      %94 = hivm.hir.pointer_cast(%c672_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %53]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_36 = memref.cast %27 : memref<2x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_37 = memref.cast %45 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_38 = memref.cast %94 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_36, %cast_37, %cast_38, %cst_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape_39 = memref.collapse_shape %45 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      %95 = arith.cmpf oeq, %87, %cst_0 : f32
      %96 = hivm.hir.pointer_cast(%c800_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_40 = memref.cast %collapse_shape_39 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_41 = memref.cast %21 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_42 = memref.cast %96 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vdiv_vs_1d_float(%cast_40, %87, %cast_41, %cast_42) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %97 = arith.index_castui %95 : i1 to index
      %98 = affine.apply #map3()[%97]
      %subview_43 = memref.subview %15[0] [%98] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_44 = memref.subview %21[0] [%98] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_45 = memref.cast %subview_43 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_46 = memref.cast %subview_44 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_45, %cast_46, %c0_i32, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %99 = arith.muli %68, %arg10 : i64
      %100 = arith.extsi %55 : i32 to i64
      %101 = arith.muli %100, %arg11 : i64
      %102 = arith.addi %99, %101 : i64
      %103 = arith.index_cast %102 : i64 to index
      %reinterpret_cast_47 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_48 = memref.cast %21 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_49 = memref.cast %reinterpret_cast_47 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_48, %cast_49, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %53]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
