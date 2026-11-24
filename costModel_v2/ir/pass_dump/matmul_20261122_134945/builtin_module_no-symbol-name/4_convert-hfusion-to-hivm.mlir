// -----// IR Dump After ConvertHFusionToHIVM (convert-hfusion-to-hivm) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.3.0">, memref.memref_as_ptr} {
  func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.addi %arg6, %c63_i32 : i32
    %1 = arith.divsi %0, %c64_i32 : i32
    %2 = arith.addi %arg7, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    %4 = arith.muli %3, %c8_i32 : i32
    %5 = arith.divsi %arg15, %4 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.subi %1, %6 : i32
    %8 = arith.minsi %7, %c8_i32 : i32
    %9 = arith.remsi %arg15, %8 : i32
    %10 = arith.addi %6, %9 : i32
    %11 = arith.remsi %arg15, %4 : i32
    %12 = arith.divsi %11, %8 : i32
    %13 = arith.muli %10, %c64_i32 : i32
    %14 = arith.muli %12, %c64_i32 : i32
    %15 = arith.addi %arg8, %c63_i32 : i32
    %16 = arith.divsi %15, %c64_i32 : i32
    %17 = arith.muli %arg10, %c64_i32 : i32
    %18 = arith.index_cast %13 : i32 to index
    %19 = arith.index_cast %arg9 : i32 to index
    %20 = arith.muli %18, %19 : index
    %21 = arith.index_cast %arg10 : i32 to index
    %22 = arith.index_cast %14 : i32 to index
    %23 = tensor.empty() : tensor<64x64xf32>
    %24:3 = scf.for %arg18 = %c0_i32 to %16 step %c1_i32 iter_args(%arg19 = %23, %arg20 = %20, %arg21 = %c0) -> (tensor<64x64xf32>, index, index)  : i32 {
      %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%arg20], sizes: [64, 64], strides: [%19, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %42 = arith.addi %arg21, %22 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%42], sizes: [64, 64], strides: [%21, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
      %43 = arith.muli %arg18, %c64_i32 : i32
      %44 = arith.subi %arg8, %43 : i32
      %alloc = memref.alloc() : memref<64x64xf32>
      %45 = arith.addi %18, %c64 : index
      %46 = arith.index_cast %arg6 : i32 to index
      %47 = arith.maxsi %18, %46 : index
      %48 = arith.minsi %45, %47 : index
      %49 = arith.subi %48, %18 : index
      %50 = arith.index_cast %44 : i32 to index
      %51 = arith.maxsi %50, %c0 : index
      %52 = arith.minsi %51, %c64 : index
      %53 = arith.minsi %49, %c64 : index
      %54 = arith.maxsi %53, %c0 : index
      %55 = arith.minsi %52, %c64 : index
      %56 = arith.maxsi %55, %c0 : index
      %57 = arith.cmpi slt, %54, %c64 : index
      %58 = arith.cmpi slt, %56, %c64 : index
      %59 = arith.ori %57, %58 : i1
      scf.if %59 {
        hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
      } {hivm.unlikely_condition}
      %subview_2 = memref.subview %reinterpret_cast_0[0, 0] [%54, %56] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %alloc[0, 0] [%54, %56] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_2, %subview_3 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      %60 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
      %alloc_4 = memref.alloc() : memref<64x64xf32>
      %61 = arith.addi %22, %c64 : index
      %62 = arith.index_cast %arg7 : i32 to index
      %63 = arith.maxsi %22, %62 : index
      %64 = arith.minsi %61, %63 : index
      %65 = arith.subi %64, %22 : index
      %66 = arith.minsi %65, %c64 : index
      %67 = arith.maxsi %66, %c0 : index
      %68 = arith.cmpi slt, %67, %c64 : index
      %69 = arith.ori %58, %68 : i1
      scf.if %69 {
        hivm.hir.vbrc ins(%cst : f32) outs(%alloc_4 : memref<64x64xf32>)
      } {hivm.unlikely_condition}
      %subview_5 = memref.subview %reinterpret_cast_1[0, 0] [%56, %67] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_6 = memref.subview %alloc_4[0, 0] [%56, %67] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_5, %subview_6 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      %70 = bufferization.to_tensor %alloc_4 restrict writable : memref<64x64xf32>
      %71 = arith.cmpi eq, %arg18, %c0_i32 : i32
      %72 = hivm.hir.mmadL1 ins(%60, %70, %71, %c0, %c0, %c0 : tensor<64x64xf32>, tensor<64x64xf32>, i1, index, index, index) outs(%arg19 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %73 = arith.addi %arg20, %c64 : index
      %74 = arith.index_cast %17 : i32 to index
      %75 = arith.addi %arg21, %74 : index
      scf.yield %72, %73, %75 : tensor<64x64xf32>, index, index
    }
    %25 = arith.index_cast %arg11 : i32 to index
    %26 = arith.muli %18, %25 : index
    %27 = arith.addi %26, %22 : index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%27], sizes: [64, 64], strides: [%25, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
    %28 = arith.addi %18, %c64 : index
    %29 = arith.index_cast %arg6 : i32 to index
    %30 = arith.maxsi %18, %29 : index
    %31 = arith.minsi %28, %30 : index
    %32 = arith.subi %31, %18 : index
    %33 = arith.addi %22, %c64 : index
    %34 = arith.index_cast %arg7 : i32 to index
    %35 = arith.maxsi %22, %34 : index
    %36 = arith.minsi %33, %35 : index
    %37 = arith.subi %36, %22 : index
    %38 = arith.minsi %32, %c64 : index
    %39 = arith.maxsi %38, %c0 : index
    %40 = arith.minsi %37, %c64 : index
    %41 = arith.maxsi %40, %c0 : index
    %extracted_slice = tensor.extract_slice %24#0[0, 0] [%39, %41] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%39, %41] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}


