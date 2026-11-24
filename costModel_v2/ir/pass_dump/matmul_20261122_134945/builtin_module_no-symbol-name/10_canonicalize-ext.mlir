// -----// IR Dump After ExtendedCanonicalizer (canonicalize-ext) //----- //
#map = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<()[s0, s1] -> (s1, s0)>
#map3 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map4 = affine_map<()[s0] -> (0, s0)>
#map5 = affine_map<()[s0, s1] -> (64, s0 - s1)>
#map6 = affine_map<()[s0] -> (64, s0)>
#map7 = affine_map<()[s0] -> (s0 + 64)>
#map8 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.3.0">, hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "mix", parallel_mode = "simd"} {
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
    scf.for %arg15 = %3 to %1 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %4 = arith.muli %arg14, %arg13 : i32
      %5 = arith.divsi %arg15, %4 : i32
      %6 = arith.remsi %5, %arg12 : i32
      %7 = arith.addi %arg6, %c63_i32 : i32
      %8 = arith.divsi %7, %c64_i32 : i32
      %9 = arith.addi %arg7, %c63_i32 : i32
      %10 = arith.divsi %9, %c64_i32 : i32
      %11 = arith.muli %10, %c8_i32 : i32
      %12 = arith.divsi %6, %11 : i32
      %13 = arith.muli %12, %c8_i32 : i32
      %14 = arith.subi %8, %13 : i32
      %15 = arith.minsi %14, %c8_i32 : i32
      %16 = arith.remsi %6, %15 : i32
      %17 = arith.addi %13, %16 : i32
      %18 = arith.remsi %6, %11 : i32
      %19 = arith.divsi %18, %15 : i32
      %20 = arith.muli %17, %c64_i32 : i32
      %21 = arith.muli %19, %c64_i32 : i32
      %22 = arith.addi %arg8, %c63_i32 : i32
      %23 = arith.divsi %22, %c64_i32 : i32
      %24 = arith.muli %arg10, %c64_i32 : i32
      %25 = arith.index_cast %20 : i32 to index
      %26 = arith.index_cast %arg9 : i32 to index
      %27 = affine.apply #map()[%25, %26]
      %28 = arith.index_cast %arg10 : i32 to index
      %29 = arith.index_cast %21 : i32 to index
      %30 = tensor.empty() : tensor<64x64xf32>
      %31:3 = scf.for %arg16 = %c0_i32 to %23 step %c1_i32 iter_args(%arg17 = %30, %arg18 = %27, %arg19 = %c0) -> (tensor<64x64xf32>, index, index)  : i32 {
        %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%arg18], sizes: [64, 64], strides: [%26, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
        %44 = affine.apply #map1()[%arg19, %29]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [64, 64], strides: [%28, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
        %45 = arith.muli %arg16, %c64_i32 : i32
        %46 = arith.subi %arg8, %45 : i32
        %alloc = memref.alloc() : memref<64x64xf32>
        %47 = arith.index_cast %arg6 : i32 to index
        %48 = affine.max #map2()[%25, %47]
        %49 = affine.min #map3()[%48, %25]
        %50 = arith.index_cast %46 : i32 to index
        %51 = affine.max #map4()[%50]
        %52 = affine.min #map5()[%49, %25]
        %53 = affine.max #map4()[%52]
        %54 = affine.min #map6()[%51]
        %55 = affine.max #map4()[%54]
        %56 = arith.cmpi slt, %53, %c64 : index
        %57 = arith.cmpi slt, %55, %c64 : index
        %58 = arith.ori %56, %57 : i1
        %subview_2 = memref.subview %reinterpret_cast_0[0, 0] [%53, %55] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_3 = memref.subview %alloc[0, 0] [%53, %55] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
        hivm.hir.load ins(%subview_2 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_3 : memref<?x?xf32, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %58 : i1 may_implicit_transpose_with_last_axis = false core_type = <CUBE>
        %59 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
        %alloc_4 = memref.alloc() : memref<64x64xf32>
        %60 = arith.index_cast %arg7 : i32 to index
        %61 = affine.max #map2()[%29, %60]
        %62 = affine.min #map3()[%61, %29]
        %63 = affine.min #map5()[%62, %29]
        %64 = affine.max #map4()[%63]
        %65 = arith.cmpi slt, %64, %c64 : index
        %66 = arith.ori %57, %65 : i1
        %subview_5 = memref.subview %reinterpret_cast_1[0, 0] [%55, %64] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_6 = memref.subview %alloc_4[0, 0] [%55, %64] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
        hivm.hir.load ins(%subview_5 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_6 : memref<?x?xf32, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %66 : i1 may_implicit_transpose_with_last_axis = false core_type = <CUBE>
        %67 = bufferization.to_tensor %alloc_4 restrict writable : memref<64x64xf32>
        %68 = arith.cmpi eq, %arg16, %c0_i32 : i32
        %69 = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%59, %67, %68, %53, %55, %64 : tensor<64x64xf32>, tensor<64x64xf32>, i1, index, index, index) outs(%arg17 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %70 = affine.apply #map7()[%arg18]
        %71 = arith.index_cast %24 : i32 to index
        %72 = affine.apply #map1()[%arg19, %71]
        scf.yield %69, %70, %72 : tensor<64x64xf32>, index, index
      } {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly}
      %32 = arith.index_cast %arg11 : i32 to index
      %33 = affine.apply #map8()[%29, %25, %32]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%33], sizes: [64, 64], strides: [%32, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %34 = arith.index_cast %arg6 : i32 to index
      %35 = affine.max #map2()[%25, %34]
      %36 = affine.min #map3()[%35, %25]
      %37 = arith.index_cast %arg7 : i32 to index
      %38 = affine.max #map2()[%29, %37]
      %39 = affine.min #map3()[%38, %29]
      %40 = affine.min #map5()[%36, %25]
      %41 = affine.max #map4()[%40]
      %42 = affine.min #map5()[%39, %29]
      %43 = affine.max #map4()[%42]
      %extracted_slice = tensor.extract_slice %31#0[0, 0] [%41, %43] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
      %subview = memref.subview %reinterpret_cast[0, 0] [%41, %43] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%extracted_slice : tensor<?x?xf32>) outs(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>>)
    } {autoblockify.subloop}
    return
  }
}


