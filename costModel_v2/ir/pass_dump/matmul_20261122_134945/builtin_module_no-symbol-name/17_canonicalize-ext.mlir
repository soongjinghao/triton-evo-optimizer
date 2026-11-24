// -----// IR Dump After ExtendedCanonicalizer (canonicalize-ext) //----- //
#map = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map3 = affine_map<()[s0, s1] -> (64, s0 - s1)>
#map4 = affine_map<()[s0] -> (0, s0)>
#map5 = affine_map<()[s0, s1] -> (s0 + s1)>
#map6 = affine_map<()[s0] -> (64, s0)>
#map7 = affine_map<()[s0] -> ((s0 + 15) floordiv 16)>
#map8 = affine_map<()[s0] -> ((s0 + 7) floordiv 8)>
#map9 = affine_map<()[s0] -> (s0 + 64)>
#map10 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.3.0">, hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @matmul_kernel_simplified_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c20_i8 = arith.constant 20 : i8
    return %c20_i8 : i8
  }
  func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c63_i32 = arith.constant 63 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1_i32 = arith.constant 1 : i32
    %c24_i32 = arith.constant 24 : i32
    %0 = arith.muli %arg12, %arg13 : i32
    %1 = arith.muli %0, %arg14 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg14, %arg13 : i32
    %5 = arith.addi %arg6, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.addi %arg7, %c63_i32 : i32
    %8 = arith.divsi %7, %c64_i32 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.addi %arg8, %c63_i32 : i32
    %11 = arith.divsi %10, %c64_i32 : i32
    %12 = arith.muli %arg10, %c64_i32 : i32
    %13 = arith.index_cast %arg9 : i32 to index
    %14 = arith.index_cast %arg10 : i32 to index
    %15 = arith.index_cast %arg6 : i32 to index
    %16 = arith.index_cast %arg7 : i32 to index
    %17 = arith.index_cast %12 : i32 to index
    %18 = arith.index_cast %arg11 : i32 to index
    scf.for %arg15 = %3 to %1 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %19 = arith.divsi %arg15, %4 : i32
      %20 = arith.remsi %19, %arg12 : i32
      %21 = arith.divsi %20, %9 : i32
      %22 = arith.muli %21, %c8_i32 : i32
      %23 = arith.subi %6, %22 : i32
      %24 = arith.minsi %23, %c8_i32 : i32
      %25 = arith.remsi %20, %24 : i32
      %26 = arith.addi %22, %25 : i32
      %27 = arith.remsi %20, %9 : i32
      %28 = arith.divsi %27, %24 : i32
      %29 = arith.muli %26, %c64_i32 : i32
      %30 = arith.muli %28, %c64_i32 : i32
      %31 = arith.index_cast %29 : i32 to index
      %32 = affine.apply #map()[%31, %13]
      %33 = arith.index_cast %30 : i32 to index
      %34 = affine.max #map1()[%31, %15]
      %35 = affine.min #map2()[%34, %31]
      %36 = affine.min #map3()[%35, %31]
      %37 = affine.max #map4()[%36]
      %38 = arith.cmpi slt, %37, %c64 : index
      %39 = affine.max #map1()[%33, %16]
      %40 = affine.min #map2()[%39, %33]
      %41 = affine.min #map3()[%40, %33]
      %42 = affine.max #map4()[%41]
      %43 = arith.cmpi slt, %42, %c64 : index
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4x16x16xf32, #hivm.address_space<cc>>
      %cast = memref.cast %alloc : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      %44:2 = scf.for %arg16 = %c0_i32 to %11 step %c1_i32 iter_args(%arg17 = %32, %arg18 = %c0) -> (index, index)  : i32 {
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%arg17], sizes: [64, 64], strides: [%13, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %48 = affine.apply #map5()[%arg18, %33]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [64, 64], strides: [%14, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %49 = arith.muli %arg16, %c64_i32 : i32
        %50 = arith.subi %arg8, %49 : i32
        %alloc_4 = memref.alloc() : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        %cast_5 = memref.cast %alloc_4 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
        %51 = arith.index_cast %50 : i32 to index
        %52 = affine.max #map4()[%51]
        %53 = affine.min #map6()[%52]
        %54 = affine.max #map4()[%53]
        %55 = arith.cmpi slt, %54, %c64 : index
        %56 = arith.ori %38, %55 : i1
        %subview_6 = memref.subview %reinterpret_cast_2[0, 0] [%37, %54] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %57 = affine.apply #map7()[%37]
        %58 = affine.apply #map8()[%54]
        %subview_7 = memref.subview %alloc_4[0, 0, 0, 0] [%58, %57, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
        %cast_8 = memref.cast %subview_7 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %56 {
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>)
        } {hivm.unlikely_condition}
        hivm.hir.nd2nz {dst_continuous} ins(%subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_8 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
        %alloc_9 = memref.alloc() : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        %cast_10 = memref.cast %alloc_9 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
        %59 = arith.ori %55, %43 : i1
        %subview_11 = memref.subview %reinterpret_cast_3[0, 0] [%54, %42] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %60 = affine.apply #map7()[%54]
        %61 = affine.apply #map8()[%42]
        %subview_12 = memref.subview %alloc_9[0, 0, 0, 0] [%61, %60, 16, 8] [1, 1, 1, 1] : memref<8x4x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
        %cast_13 = memref.cast %subview_12 : memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %59 {
          hivm.hir.vbrc ins(%cst : f32) outs(%alloc_9 : memref<8x4x16x8xf32, #hivm.address_space<cbuf>>)
        } {hivm.unlikely_condition}
        hivm.hir.nd2nz {dst_continuous} ins(%subview_11 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_13 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
        %62 = arith.cmpi eq, %arg16, %c0_i32 : i32
        hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_5, %cast_10, %62, %37, %54, %42 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
        %63 = affine.apply #map9()[%arg17]
        %64 = affine.apply #map5()[%arg18, %17]
        scf.yield %63, %64 : index, index
      } {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly}
      %45 = affine.apply #map10()[%33, %31, %18]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%45], sizes: [64, 64], strides: [%18, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %46 = affine.apply #map7()[%37]
      %47 = affine.apply #map7()[%42]
      %subview = memref.subview %alloc[0, 0, 0, 0] [%47, %46, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>>
      %cast_0 = memref.cast %subview : memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>
      %subview_1 = memref.subview %reinterpret_cast[0, 0] [%37, %42] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_0 : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cc>>) outs(%subview_1 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
    } {autoblockify.subloop}
    return
  }
}


