// -----// IR Dump After TritonGlobalKernelArgsToHIVMOp (triton-global-kernel-args-to-hivm-op) //----- //
func.func @matmul_kernel_simplified(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix", parallel_mode = "simd"} {
  %0 = arith.muli %arg12, %arg13 : i32
  %1 = arith.muli %0, %arg14 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %c1_i32 = arith.constant 1 : i32
  %4 = arith.divsi %3, %c1_i32 : i32
  %5 = arith.remsi %4, %arg14 : i32
  %6 = arith.muli %c1_i32, %arg14 : i32
  %7 = arith.divsi %3, %6 : i32
  %8 = arith.remsi %7, %arg13 : i32
  %9 = arith.muli %6, %arg13 : i32
  %10 = arith.divsi %3, %9 : i32
  %11 = arith.remsi %10, %arg12 : i32
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c8_i32 = arith.constant 8 : i32
  %c63_i32 = arith.constant 63 : i32
  %c1_i32_0 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %12 = arith.addi %arg6, %c63_i32 : i32
  %13 = arith.divsi %12, %c64_i32 : i32
  %14 = arith.addi %arg7, %c63_i32 : i32
  %15 = arith.divsi %14, %c64_i32 : i32
  %16 = arith.muli %15, %c8_i32 : i32
  %17 = arith.divsi %11, %16 : i32
  %18 = arith.muli %17, %c8_i32 : i32
  %19 = arith.subi %13, %18 : i32
  %20 = arith.minsi %19, %c8_i32 : i32
  %21 = arith.remsi %11, %20 : i32
  %22 = arith.addi %18, %21 : i32
  %23 = arith.remsi %11, %16 : i32
  %24 = arith.divsi %23, %20 : i32
  %25 = arith.muli %22, %c64_i32 : i32
  %26 = arith.muli %24, %c64_i32 : i32
  %27 = arith.addi %arg8, %c63_i32 : i32
  %28 = arith.divsi %27, %c64_i32 : i32
  %29 = arith.muli %arg10, %c64_i32 : i32
  %30 = arith.index_cast %25 : i32 to index
  %31 = arith.index_cast %arg9 : i32 to index
  %32 = arith.muli %30, %31 : index
  %33 = arith.index_cast %arg10 : i32 to index
  %34 = arith.index_cast %26 : i32 to index
  %35 = tensor.empty() : tensor<64x64xf32>
  %36:3 = scf.for %arg15 = %c0_i32 to %28 step %c1_i32_0 iter_args(%arg16 = %35, %arg17 = %32, %arg18 = %c0) -> (tensor<64x64xf32>, index, index)  : i32 {
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%arg17], sizes: [64, 64], strides: [%31, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
    %54 = arith.addi %arg18, %34 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [64, 64], strides: [%33, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
    %55 = arith.muli %arg15, %c64_i32 : i32
    %56 = arith.subi %arg8, %55 : i32
    %alloc = memref.alloc() : memref<64x64xf32>
    %57 = arith.addi %30, %c64 : index
    %58 = arith.index_cast %arg6 : i32 to index
    %59 = arith.maxsi %30, %58 : index
    %60 = arith.minsi %57, %59 : index
    %61 = arith.subi %60, %30 : index
    %62 = arith.index_cast %56 : i32 to index
    %63 = arith.maxsi %62, %c0 : index
    %64 = arith.minsi %63, %c64 : index
    %65 = arith.minsi %61, %c64 : index
    %66 = arith.maxsi %65, %c0 : index
    %67 = arith.minsi %64, %c64 : index
    %68 = arith.maxsi %67, %c0 : index
    %69 = arith.cmpi slt, %66, %c64 : index
    %70 = arith.cmpi slt, %68, %c64 : index
    %71 = arith.ori %69, %70 : i1
    scf.if %71 {
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
    } {hivm.unlikely_condition}
    %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%66, %68] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    %subview_4 = memref.subview %alloc[0, 0] [%66, %68] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    memref.copy %subview_3, %subview_4 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
    %72 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
    %alloc_5 = memref.alloc() : memref<64x64xf32>
    %73 = arith.addi %34, %c64 : index
    %74 = arith.index_cast %arg7 : i32 to index
    %75 = arith.maxsi %34, %74 : index
    %76 = arith.minsi %73, %75 : index
    %77 = arith.subi %76, %34 : index
    %78 = arith.minsi %77, %c64 : index
    %79 = arith.maxsi %78, %c0 : index
    %80 = arith.cmpi slt, %79, %c64 : index
    %81 = arith.ori %70, %80 : i1
    scf.if %81 {
      hivm.hir.vbrc ins(%cst : f32) outs(%alloc_5 : memref<64x64xf32>)
    } {hivm.unlikely_condition}
    %subview_6 = memref.subview %reinterpret_cast_2[0, 0] [%68, %79] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    %subview_7 = memref.subview %alloc_5[0, 0] [%68, %79] [1, 1] : memref<64x64xf32> to memref<?x?xf32, strided<[64, 1]>>
    memref.copy %subview_6, %subview_7 : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
    %82 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x64xf32>
    %83 = arith.cmpi eq, %arg15, %c0_i32 : i32
    %84 = hivm.hir.mmadL1 ins(%72, %82, %83, %c0, %c0, %c0 : tensor<64x64xf32>, tensor<64x64xf32>, i1, index, index, index) outs(%arg16 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %85 = arith.addi %arg17, %c64 : index
    %86 = arith.index_cast %29 : i32 to index
    %87 = arith.addi %arg18, %86 : index
    scf.yield %84, %85, %87 : tensor<64x64xf32>, index, index
  }
  %37 = arith.index_cast %arg11 : i32 to index
  %38 = arith.muli %30, %37 : index
  %39 = arith.addi %38, %34 : index
  %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%39], sizes: [64, 64], strides: [%37, 1] : memref<?xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
  %40 = arith.addi %30, %c64 : index
  %41 = arith.index_cast %arg6 : i32 to index
  %42 = arith.maxsi %30, %41 : index
  %43 = arith.minsi %40, %42 : index
  %44 = arith.subi %43, %30 : index
  %45 = arith.addi %34, %c64 : index
  %46 = arith.index_cast %arg7 : i32 to index
  %47 = arith.maxsi %34, %46 : index
  %48 = arith.minsi %45, %47 : index
  %49 = arith.subi %48, %34 : index
  %50 = arith.minsi %44, %c64 : index
  %51 = arith.maxsi %50, %c0 : index
  %52 = arith.minsi %49, %c64 : index
  %53 = arith.maxsi %52, %c0 : index
  %extracted_slice = tensor.extract_slice %36#0[0, 0] [%51, %53] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
  %subview = memref.subview %reinterpret_cast[0, 0] [%51, %53] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
  bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
  return
}

