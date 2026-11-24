// -----// IR Dump After AppendTargetDeviceSpec (hacc-append-device-spec) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hacc.hivmc_compatible_print = false, hacc.hivmc_version = #hacc.hivmc_version<"0.3.0">} {
  func.func @matmul_kernel_simplified(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8_i32 = arith.constant 8 : i32
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = arith.addi %arg5, %c63_i32 : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    %4 = arith.addi %arg6, %c63_i32 : i32
    %5 = arith.divsi %4, %c64_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %arg14, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.minsi %9, %c8_i32 : i32
    %11 = arith.remsi %arg14, %10 : i32
    %12 = arith.addi %8, %11 : i32
    %13 = arith.remsi %arg14, %6 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c64_i32 : i32
    %16 = arith.muli %14, %c64_i32 : i32
    %17 = arith.addi %arg7, %c63_i32 : i32
    %18 = arith.divsi %17, %c64_i32 : i32
    %19 = arith.muli %arg9, %c64_i32 : i32
    %20 = arith.index_cast %15 : i32 to index
    %21 = arith.index_cast %arg8 : i32 to index
    %22 = arith.muli %20, %21 : index
    %23 = arith.index_cast %arg9 : i32 to index
    %24 = arith.index_cast %16 : i32 to index
    %25:3 = scf.for %arg17 = %c0_i32 to %18 step %c1_i32 iter_args(%arg18 = %1, %arg19 = %22, %arg20 = %c0) -> (tensor<64x64xf32>, index, index)  : i32 {
      %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%arg19], sizes: [64, 64], strides: [%21, %c1] : memref<?xf32> to memref<64x64xf32, strided<[?, ?], offset: ?>>
      %43 = arith.addi %arg20, %24 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [64, 64], strides: [%23, %c1] : memref<?xf32> to memref<64x64xf32, strided<[?, ?], offset: ?>>
      %44 = arith.muli %arg17, %c64_i32 : i32
      %45 = arith.subi %arg7, %44 : i32
      %alloc = memref.alloc() : memref<64x64xf32>
      %46 = arith.addi %20, %c64 : index
      %47 = arith.index_cast %arg5 : i32 to index
      %48 = arith.maxsi %20, %47 : index
      %49 = arith.minsi %46, %48 : index
      %50 = arith.subi %49, %20 : index
      %51 = arith.index_cast %45 : i32 to index
      %52 = arith.maxsi %51, %c0 : index
      %53 = arith.minsi %52, %c64 : index
      %54 = arith.minsi %50, %c64 : index
      %55 = arith.maxsi %54, %c0 : index
      %56 = arith.minsi %53, %c64 : index
      %57 = arith.maxsi %56, %c0 : index
      %58 = arith.cmpi slt, %55, %c64 : index
      %59 = arith.cmpi slt, %57, %c64 : index
      %60 = arith.ori %58, %59 : i1
      scf.if %60 {
        linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
      } {hivm.unlikely_condition}
      %subview_2 = memref.subview %reinterpret_cast_0[0, 0] [%55, %57] [1, 1] : memref<64x64xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_3 = memref.subview %alloc[0, 0] [%55, %57] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_2, %subview_3 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      %61 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
      %alloc_4 = memref.alloc() : memref<64x64xf32>
      %62 = arith.addi %24, %c64 : index
      %63 = arith.index_cast %arg6 : i32 to index
      %64 = arith.maxsi %24, %63 : index
      %65 = arith.minsi %62, %64 : index
      %66 = arith.subi %65, %24 : index
      %67 = arith.minsi %66, %c64 : index
      %68 = arith.maxsi %67, %c0 : index
      %69 = arith.cmpi slt, %68, %c64 : index
      %70 = arith.ori %59, %69 : i1
      scf.if %70 {
        linalg.fill ins(%cst : f32) outs(%alloc_4 : memref<64x64xf32>)
      } {hivm.unlikely_condition}
      %subview_5 = memref.subview %reinterpret_cast_1[0, 0] [%57, %68] [1, 1] : memref<64x64xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %subview_6 = memref.subview %alloc_4[0, 0] [%57, %68] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
      memref.copy %subview_5, %subview_6 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
      %71 = bufferization.to_tensor %alloc_4 restrict writable : memref<64x64xf32>
      %72 = linalg.matmul {input_precision = "ieee"} ins(%61, %71 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %73 = arith.addi %arg19, %c64 : index
      %74 = arith.index_cast %19 : i32 to index
      %75 = arith.addi %arg20, %74 : index
      scf.yield %72, %73, %75 : tensor<64x64xf32>, index, index
    }
    %26 = arith.index_cast %arg10 : i32 to index
    %27 = arith.muli %20, %26 : index
    %28 = arith.addi %27, %24 : index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%28], sizes: [64, 64], strides: [%26, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
    %29 = arith.addi %20, %c64 : index
    %30 = arith.index_cast %arg5 : i32 to index
    %31 = arith.maxsi %20, %30 : index
    %32 = arith.minsi %29, %31 : index
    %33 = arith.subi %32, %20 : index
    %34 = arith.addi %24, %c64 : index
    %35 = arith.index_cast %arg6 : i32 to index
    %36 = arith.maxsi %24, %35 : index
    %37 = arith.minsi %34, %36 : index
    %38 = arith.subi %37, %24 : index
    %39 = arith.minsi %33, %c64 : index
    %40 = arith.maxsi %39, %c0 : index
    %41 = arith.minsi %38, %c64 : index
    %42 = arith.maxsi %41, %c0 : index
    %extracted_slice = tensor.extract_slice %25#0[0, 0] [%40, %42] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%40, %42] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}


