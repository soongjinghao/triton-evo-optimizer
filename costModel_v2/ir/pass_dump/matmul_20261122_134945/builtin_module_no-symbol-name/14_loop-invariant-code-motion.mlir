// -----// IR Dump After LoopInvariantCodeMotion (loop-invariant-code-motion) //----- //
#map = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map3 = affine_map<()[s0, s1] -> (64, s0 - s1)>
#map4 = affine_map<()[s0] -> (0, s0)>
#map5 = affine_map<()[s0, s1] -> (s0 + s1)>
#map6 = affine_map<()[s0] -> (64, s0)>
#map7 = affine_map<()[s0] -> (s0 + 64)>
#map8 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.3.0">, hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} {
  func.func @matmul_kernel_simplified_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c20_i8 = arith.constant 20 : i8
    return %c20_i8 : i8
  }
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
    %15 = tensor.empty() : tensor<64x64xf32>
    %16 = arith.index_cast %arg6 : i32 to index
    %17 = arith.index_cast %arg7 : i32 to index
    %18 = arith.index_cast %12 : i32 to index
    %19 = arith.index_cast %arg11 : i32 to index
    %20 = arith.index_cast %arg6 : i32 to index
    %21 = arith.index_cast %arg7 : i32 to index
    scf.for %arg15 = %3 to %1 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %22 = arith.divsi %arg15, %4 : i32
      %23 = arith.remsi %22, %arg12 : i32
      %24 = arith.divsi %23, %9 : i32
      %25 = arith.muli %24, %c8_i32 : i32
      %26 = arith.subi %6, %25 : i32
      %27 = arith.minsi %26, %c8_i32 : i32
      %28 = arith.remsi %23, %27 : i32
      %29 = arith.addi %25, %28 : i32
      %30 = arith.remsi %23, %9 : i32
      %31 = arith.divsi %30, %27 : i32
      %32 = arith.muli %29, %c64_i32 : i32
      %33 = arith.muli %31, %c64_i32 : i32
      %34 = arith.index_cast %32 : i32 to index
      %35 = affine.apply #map()[%34, %13]
      %36 = arith.index_cast %33 : i32 to index
      %37 = affine.max #map1()[%34, %16]
      %38 = affine.min #map2()[%37, %34]
      %39 = affine.min #map3()[%38, %34]
      %40 = affine.max #map4()[%39]
      %41 = arith.cmpi slt, %40, %c64 : index
      %42 = affine.max #map1()[%36, %17]
      %43 = affine.min #map2()[%42, %36]
      %44 = affine.min #map3()[%43, %36]
      %45 = affine.max #map4()[%44]
      %46 = arith.cmpi slt, %45, %c64 : index
      %47:3 = scf.for %arg16 = %c0_i32 to %11 step %c1_i32 iter_args(%arg17 = %15, %arg18 = %35, %arg19 = %c0) -> (tensor<64x64xf32>, index, index)  : i32 {
        %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%arg18], sizes: [64, 64], strides: [%13, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
        %57 = affine.apply #map5()[%arg19, %36]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%57], sizes: [64, 64], strides: [%14, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
        %58 = arith.muli %arg16, %c64_i32 : i32
        %59 = arith.subi %arg8, %58 : i32
        %alloc = memref.alloc() : memref<64x64xf32>
        %60 = arith.index_cast %59 : i32 to index
        %61 = affine.max #map4()[%60]
        %62 = affine.min #map6()[%61]
        %63 = affine.max #map4()[%62]
        %64 = arith.cmpi slt, %63, %c64 : index
        %65 = arith.ori %41, %64 : i1
        %subview_2 = memref.subview %reinterpret_cast_0[0, 0] [%40, %63] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_3 = memref.subview %alloc[0, 0] [%40, %63] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
        hivm.hir.load ins(%subview_2 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_3 : memref<?x?xf32, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %65 : i1 may_implicit_transpose_with_last_axis = false core_type = <CUBE>
        %66 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
        %alloc_4 = memref.alloc() : memref<64x64xf32>
        %67 = arith.ori %64, %46 : i1
        %subview_5 = memref.subview %reinterpret_cast_1[0, 0] [%63, %45] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %subview_6 = memref.subview %alloc_4[0, 0] [%63, %45] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
        hivm.hir.load ins(%subview_5 : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%subview_6 : memref<?x?xf32, strided<[64, 1]>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %67 : i1 may_implicit_transpose_with_last_axis = false core_type = <CUBE>
        %68 = bufferization.to_tensor %alloc_4 restrict writable : memref<64x64xf32>
        %69 = arith.cmpi eq, %arg16, %c0_i32 : i32
        %70 = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%66, %68, %69, %40, %63, %45 : tensor<64x64xf32>, tensor<64x64xf32>, i1, index, index, index) outs(%arg17 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %71 = affine.apply #map7()[%arg18]
        %72 = affine.apply #map5()[%arg19, %18]
        scf.yield %70, %71, %72 : tensor<64x64xf32>, index, index
      } {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly}
      %48 = affine.apply #map8()[%36, %34, %19]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%48], sizes: [64, 64], strides: [%19, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %49 = affine.max #map1()[%34, %20]
      %50 = affine.min #map2()[%49, %34]
      %51 = affine.max #map1()[%36, %21]
      %52 = affine.min #map2()[%51, %36]
      %53 = affine.min #map3()[%50, %34]
      %54 = affine.max #map4()[%53]
      %55 = affine.min #map3()[%52, %36]
      %56 = affine.max #map4()[%55]
      %extracted_slice = tensor.extract_slice %47#0[0, 0] [%54, %56] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
      %subview = memref.subview %reinterpret_cast[0, 0] [%54, %56] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%extracted_slice : tensor<?x?xf32>) outs(%subview : memref<?x?xf32, strided<[?, 1], offset: ?>>)
    } {autoblockify.subloop}
    return
  }
}


