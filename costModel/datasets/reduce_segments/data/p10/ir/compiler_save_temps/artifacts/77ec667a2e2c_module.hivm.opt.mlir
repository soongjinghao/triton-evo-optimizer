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
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c15_i32 = arith.constant 15 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8_i64 = arith.constant 8 : i64
    %cst_0 = arith.constant 0xFF800000 : f32
    %c0_i64 = arith.constant 0 : i64
    %c192_i64 = arith.constant 192 : i64
    %c224_i64 = arith.constant 224 : i64
    %c256_i64 = arith.constant 256 : i64
    %c544_i64 = arith.constant 544 : i64
    %c864_i64 = arith.constant 864 : i64
    %c992_i64 = arith.constant 992 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1056_i64 = arith.constant 1056 : i64
    %c96_i64 = arith.constant 96 : i64
    %c1088_i64 = arith.constant 1088 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1120_i64 = arith.constant 1120 : i64
    %c160_i64 = arith.constant 160 : i64
    %c1152_i64 = arith.constant 1152 : i64
    %c288_i64 = arith.constant 288 : i64
    %c1280_i64 = arith.constant 1280 : i64
    %c416_i64 = arith.constant 416 : i64
    %c800_i64 = arith.constant 800 : i64
    %c1408_i64 = arith.constant 1408 : i64
    %0 = hivm.hir.pointer_cast(%c800_i64) : memref<16xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c1408_i64) : memref<16xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c416_i64) : memref<2x16xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c1280_i64) : memref<2x16xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c288_i64) : memref<2x16xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c1152_i64) : memref<2x16xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c160_i64) : memref<2xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c1120_i64) : memref<2xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c128_i64) : memref<2xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c1088_i64) : memref<2xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c96_i64) : memref<2xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c1056_i64) : memref<2xf32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c64_i64) : memref<2xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c1024_i64) : memref<2xf32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c800_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c1408_i64) : memref<1x16xf32, #hivm.address_space<ub>>
    %16 = arith.muli %arg14, %arg15 : i32
    %17 = arith.muli %16, %arg16 : i32
    %18 = hivm.hir.get_block_idx -> i64
    %19 = arith.trunci %18 : i64 to i32
    %20 = arith.muli %arg16, %arg15 : i32
    %21 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<ub>>
    %cast = memref.cast %21 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_scalar_float_to_1d(%cst, %cast) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
    scf.for %arg17 = %19 to %17 step %c48_i32  : i32 {
      %22 = arith.index_cast %arg17 : i32 to index
      %23 = arith.index_cast %19 : i32 to index
      %24 = affine.apply #map()[%22, %23]
      %25 = arith.index_cast %24 : index to i64
      %26 = arith.cmpi eq, %25, %c1_i64 : i64
      %27 = arith.select %26, %1, %0 : memref<16xf32, #hivm.address_space<ub>>
      %28 = arith.index_cast %arg17 : i32 to index
      %29 = arith.index_cast %19 : i32 to index
      %30 = affine.apply #map()[%28, %29]
      %31 = arith.index_cast %30 : index to i64
      %32 = arith.cmpi eq, %31, %c1_i64 : i64
      %33 = arith.select %32, %3, %2 : memref<2x16xf32, #hivm.address_space<ub>>
      %34 = arith.index_cast %arg17 : i32 to index
      %35 = arith.index_cast %19 : i32 to index
      %36 = affine.apply #map()[%34, %35]
      %37 = arith.index_cast %36 : index to i64
      %38 = arith.cmpi eq, %37, %c1_i64 : i64
      %39 = arith.select %38, %5, %4 : memref<2x16xf32, #hivm.address_space<ub>>
      %40 = arith.index_cast %arg17 : i32 to index
      %41 = arith.index_cast %19 : i32 to index
      %42 = affine.apply #map()[%40, %41]
      %43 = arith.index_cast %42 : index to i64
      %44 = arith.cmpi eq, %43, %c1_i64 : i64
      %45 = arith.select %44, %7, %6 : memref<2xf32, #hivm.address_space<ub>>
      %46 = arith.index_cast %arg17 : i32 to index
      %47 = arith.index_cast %19 : i32 to index
      %48 = affine.apply #map()[%46, %47]
      %49 = arith.index_cast %48 : index to i64
      %50 = arith.cmpi eq, %49, %c1_i64 : i64
      %51 = arith.select %50, %9, %8 : memref<2xf32, #hivm.address_space<ub>>
      %52 = arith.index_cast %arg17 : i32 to index
      %53 = arith.index_cast %19 : i32 to index
      %54 = affine.apply #map()[%52, %53]
      %55 = arith.index_cast %54 : index to i64
      %56 = arith.cmpi eq, %55, %c1_i64 : i64
      %57 = arith.select %56, %11, %10 : memref<2xf32, #hivm.address_space<ub>>
      %58 = arith.index_cast %arg17 : i32 to index
      %59 = arith.index_cast %19 : i32 to index
      %60 = affine.apply #map()[%58, %59]
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.cmpi eq, %61, %c1_i64 : i64
      %63 = arith.select %62, %13, %12 : memref<2xf32, #hivm.address_space<ub>>
      %64 = arith.index_cast %arg17 : i32 to index
      %65 = arith.index_cast %19 : i32 to index
      %66 = affine.apply #map()[%64, %65]
      %67 = arith.index_cast %66 : index to i64
      %68 = arith.cmpi eq, %67, %c1_i64 : i64
      %69 = arith.select %68, %15, %14 : memref<1x16xf32, #hivm.address_space<ub>>
      %70 = arith.index_cast %arg17 : i32 to index
      %71 = arith.index_cast %19 : i32 to index
      %72 = affine.apply #map()[%70, %71]
      %73 = arith.index_cast %72 : index to i1
      %74 = arith.select %73, %c1_i64, %c2_i64 : i64
      %75 = arith.index_cast %72 : index to i1
      %76 = arith.xori %75, %true : i1
      %77 = arith.extui %76 : i1 to i64
      hivm.hir.set_mask_norm
      %78 = arith.divsi %arg17, %arg16 : i32
      %79 = arith.remsi %78, %arg15 : i32
      %80 = arith.divsi %arg17, %20 : i32
      %81 = arith.remsi %80, %arg14 : i32
      %82:2 = scf.while (%arg18 = %c0_i32, %arg19 = %arg8) : (i32, i32) -> (i32, i32) {
        %126 = arith.cmpi slt, %arg18, %arg19 : i32
        scf.condition(%126) %arg18, %arg19 : i32, i32
      } do {
      ^bb0(%arg18: i32, %arg19: i32):
        %126 = arith.addi %arg18, %arg19 : i32
        %127 = arith.divsi %126, %c2_i32 : i32
        %128 = arith.index_cast %127 : i32 to index
        %reinterpret_cast_33 = memref.reinterpret_cast %arg13 to offset: [%128], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %129 = memref.load %reinterpret_cast_33[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %130 = arith.cmpi sle, %129, %81 : i32
        %131 = arith.select %130, %arg19, %127 : i32
        %132 = scf.if %130 -> (i32) {
          %133 = arith.addi %127, %c1_i32 : i32
          scf.yield %133 : i32
        } else {
          scf.yield %arg18 : i32
        }
        scf.yield %132, %131 : i32, i32
      }
      %83 = arith.subi %82#0, %c1_i32 : i32
      %84 = arith.index_cast %83 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%84], sizes: [1], strides: [1] : memref<?xi32, #hivm.address_space<gm>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %85 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %86 = arith.addi %85, %c15_i32 : i32
      %87 = arith.divsi %86, %c16_i32 : i32
      %88 = arith.muli %87, %c8_i32 : i32
      %89 = arith.addi %85, %88 : i32
      %90 = arith.subi %89, %c1_i32 : i32
      %91 = arith.divsi %90, %88 : i32
      %92 = arith.extsi %81 : i32 to i64
      %93 = arith.muli %92, %c8_i64 : i64
      %94 = arith.muli %79, %c2_i32 : i32
      %95 = arith.extsi %94 : i32 to i64
      %96 = arith.addi %93, %95 : i64
      %97 = arith.cmpi eq, %91, %c2_i32 : i32
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %74]
      %98 = arith.select %97, %57, %45 : memref<2xf32, #hivm.address_space<ub>>
      %99 = arith.select %97, %63, %51 : memref<2xf32, #hivm.address_space<ub>>
      scf.if %97 {
        %126 = arith.index_cast %96 : i64 to index
        %reinterpret_cast_33 = memref.reinterpret_cast %arg5 to offset: [%126], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %cast_34 = memref.cast %reinterpret_cast_33 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_35 = memref.cast %63 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_34, %cast_35, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        %reinterpret_cast_36 = memref.reinterpret_cast %arg6 to offset: [%126], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %cast_37 = memref.cast %reinterpret_cast_36 : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_38 = memref.cast %57 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_37, %cast_38, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      } else {
        %126 = arith.index_cast %96 : i64 to index
        %reinterpret_cast_33 = memref.reinterpret_cast %arg5 to offset: [%126], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %127 = arith.index_cast %91 : i32 to index
        %128 = affine.max #map1()[%127]
        %129 = affine.min #map2()[%128]
        %130 = arith.cmpi slt, %129, %c2 : index
        %subview_34 = memref.subview %reinterpret_cast_33[0] [%129] [1] : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_35 = memref.subview %51[0] [%129] [1] : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %130 {
          %cast_43 = memref.cast %51 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_43) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        %cast_36 = memref.cast %subview_34 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_37 = memref.cast %subview_35 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_36, %cast_37, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        %reinterpret_cast_38 = memref.reinterpret_cast %arg6 to offset: [%126], sizes: [2], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_39 = memref.subview %reinterpret_cast_38[0] [%129] [1] : memref<2xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_40 = memref.subview %45[0] [%129] [1] : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        scf.if %130 {
          %cast_43 = memref.cast %45 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_43) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        %cast_41 = memref.cast %subview_39 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_42 = memref.cast %subview_40 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_41, %cast_42, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      }
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %100 = hivm.hir.pointer_cast(%c192_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %101 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_1 = memref.cast %99 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_2 = memref.cast %100 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_3 = memref.cast %101 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @enablevc_reduce_max_r_float(%cast_1, %cast_2, %cast_3, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %102 = memref.load %100[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %103 = arith.subf %cst, %102 : f32
      %104 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_4 = memref.cast %99 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_5 = memref.cast %63 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_6 = memref.cast %104 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_4, %103, %cast_5, %cast_6) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %105 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_7 = memref.cast %63 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_8 = memref.cast %63 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_9 = memref.cast %105 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vexp_1d_float(%cast_7, %cast_8, %cast_9) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]
      %106 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_10 = memref.cast %98 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_11 = memref.cast %63 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_12 = memref.cast %57 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_13 = memref.cast %106 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_1d_float(%cast_10, %cast_11, %cast_12, %cast_13) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %107 = hivm.hir.pointer_cast(%c224_i64) : memref<1xf32, #hivm.address_space<ub>>
      %108 = hivm.hir.pointer_cast(%c256_i64) : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_14 = memref.cast %57 : memref<2xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_15 = memref.cast %107 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_16 = memref.cast %108 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @enablevc_reduce_sum_r_float(%cast_14, %cast_15, %cast_16, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %109 = memref.load %107[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
      %110 = arith.muli %92, %c128_i64 : i64
      %111 = arith.muli %79, %c32_i32 : i32
      %112 = arith.extsi %111 : i32 to i64
      %113 = arith.addi %110, %112 : i64
      %114 = arith.select %97, %39, %33 : memref<2x16xf32, #hivm.address_space<ub>>
      scf.if %97 {
        %126 = arith.index_cast %113 : i64 to index
        %reinterpret_cast_33 = memref.reinterpret_cast %arg4 to offset: [%126], sizes: [2, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<2x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_34 = memref.collapse_shape %reinterpret_cast_33 [[0, 1]] : memref<2x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_35 = memref.collapse_shape %39 [[0, 1]] : memref<2x16xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
        %cast_36 = memref.cast %collapse_shape_34 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_37 = memref.cast %collapse_shape_35 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_36, %cast_37, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      } else {
        %126 = arith.index_cast %113 : i64 to index
        %reinterpret_cast_33 = memref.reinterpret_cast %arg4 to offset: [%126], sizes: [2, 16], strides: [16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<2x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        %127 = arith.index_cast %91 : i32 to index
        %128 = affine.max #map1()[%127]
        %129 = affine.min #map2()[%128]
        %130 = arith.cmpi slt, %129, %c2 : index
        %subview_34 = memref.subview %reinterpret_cast_33[0, 0] [%129, 16] [1, 1] : memref<2x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_35 = memref.subview %33[0, 0] [%129, 16] [1, 1] : memref<2x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
        scf.if %130 {
          %collapse_shape_40 = memref.collapse_shape %33 [[0, 1]] : memref<2x16xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
          %cast_41 = memref.cast %collapse_shape_40 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_scalar_float_to_1d(%cst, %cast_41) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        %collapse_shape_36 = memref.collapse_shape %subview_34 [[0, 1]] : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_37 = memref.collapse_shape %subview_35 [[0, 1]] : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
        %cast_38 = memref.cast %collapse_shape_36 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_39 = memref.cast %collapse_shape_37 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_38, %cast_39, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      }
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %expand_shape = memref.expand_shape %63 [[0, 1]] output_shape [2, 1] : memref<2xf32, #hivm.address_space<ub>> into memref<2x1xf32, #hivm.address_space<ub>>
      %115 = hivm.hir.pointer_cast(%c544_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_17 = memref.cast %114 : memref<2x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_18 = memref.cast %expand_shape : memref<2x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_19 = memref.cast %39 : memref<2x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_20 = memref.cast %115 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmul_2d_float(%cast_17, %cast_18, %cast_19, %cast_20) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %74]
      %116 = hivm.hir.pointer_cast(%c864_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, %77]
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_21 = memref.cast %39 : memref<2x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_22 = memref.cast %69 : memref<1x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_23 = memref.cast %116 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @reduce_sum_ra_float(%cast_21, %cast_22, %cast_23, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
      %collapse_shape = memref.collapse_shape %69 [[0, 1]] : memref<1x16xf32, #hivm.address_space<ub>> into memref<16xf32, #hivm.address_space<ub>>
      %117 = arith.cmpf oeq, %109, %cst : f32
      %118 = hivm.hir.pointer_cast(%c992_i64) : memref<8xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_24 = memref.cast %collapse_shape : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_25 = memref.cast %27 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_26 = memref.cast %118 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vdiv_vs_1d_float(%cast_24, %109, %cast_25, %cast_26) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %119 = arith.index_castui %117 : i1 to index
      %120 = affine.apply #map3()[%119]
      %subview = memref.subview %21[0] [%120] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_27 = memref.subview %27[0] [%120] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %cast_28 = memref.cast %subview : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_29 = memref.cast %subview_27 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @copy_ubuf_to_ubuf_1d_float(%cast_28, %cast_29, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %121 = arith.muli %92, %arg10 : i64
      %122 = arith.extsi %79 : i32 to i64
      %123 = arith.muli %122, %arg11 : i64
      %124 = arith.addi %121, %123 : i64
      %125 = arith.index_cast %124 : i64 to index
      %reinterpret_cast_30 = memref.reinterpret_cast %arg3 to offset: [%125], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_31 = memref.cast %27 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_32 = memref.cast %reinterpret_cast_30 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%cast_31, %cast_32, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, %77]
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
