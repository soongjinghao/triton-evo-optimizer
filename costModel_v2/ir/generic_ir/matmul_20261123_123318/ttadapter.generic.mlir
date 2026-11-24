#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{}, {}, {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}, {}, {}, {}], function_type = (memref<?xi8>, memref<?xi8>, memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (), sym_name = "matmul_kernel_simplified"}> ({
  ^bb0(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32):
    %0 = "arith.constant"() <{value = 64 : index}> : () -> index
    %1 = "arith.constant"() <{value = 1 : index}> : () -> index
    %2 = "arith.constant"() <{value = 0 : index}> : () -> index
    %3 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 63 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %8 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %9 = "tensor.empty"() : () -> tensor<64x64xf32>
    %10 = "linalg.fill"(%8, %9) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg28: f32, %arg29: f32):
      "linalg.yield"(%arg28) : (f32) -> ()
    }) : (f32, tensor<64x64xf32>) -> tensor<64x64xf32>
    %11 = "arith.addi"(%arg5, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %12 = "arith.divsi"(%11, %7) : (i32, i32) -> i32
    %13 = "arith.addi"(%arg6, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %14 = "arith.divsi"(%13, %7) : (i32, i32) -> i32
    %15 = "arith.muli"(%14, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %16 = "arith.divsi"(%arg14, %15) : (i32, i32) -> i32
    %17 = "arith.muli"(%16, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %18 = "arith.subi"(%12, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %19 = "arith.minsi"(%18, %3) : (i32, i32) -> i32
    %20 = "arith.remsi"(%arg14, %19) : (i32, i32) -> i32
    %21 = "arith.addi"(%17, %20) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %22 = "arith.remsi"(%arg14, %15) : (i32, i32) -> i32
    %23 = "arith.divsi"(%22, %19) : (i32, i32) -> i32
    %24 = "arith.muli"(%21, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %25 = "arith.muli"(%23, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %26 = "arith.addi"(%arg7, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %27 = "arith.divsi"(%26, %7) : (i32, i32) -> i32
    %28 = "arith.muli"(%arg9, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %29 = "arith.index_cast"(%24) : (i32) -> index
    %30 = "arith.index_cast"(%arg8) : (i32) -> index
    %31 = "arith.muli"(%29, %30) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %32 = "arith.index_cast"(%arg9) : (i32) -> index
    %33 = "arith.index_cast"(%25) : (i32) -> index
    %34:3 = "scf.for"(%6, %27, %5, %10, %31, %2) ({
    ^bb0(%arg17: i32, %arg18: tensor<64x64xf32>, %arg19: index, %arg20: index):
      %55 = "memref.reinterpret_cast"(%arg2, %arg19, %30, %1) <{operandSegmentSizes = array<i32: 1, 1, 0, 2>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<?xf32>, index, index, index) -> memref<64x64xf32, strided<[?, ?], offset: ?>>
      %56 = "arith.addi"(%arg20, %33) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %57 = "memref.reinterpret_cast"(%arg3, %56, %32, %1) <{operandSegmentSizes = array<i32: 1, 1, 0, 2>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<?xf32>, index, index, index) -> memref<64x64xf32, strided<[?, ?], offset: ?>>
      %58 = "arith.muli"(%arg17, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %59 = "arith.subi"(%arg7, %58) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %60 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x64xf32>
      %61 = "arith.addi"(%29, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %62 = "arith.index_cast"(%arg5) : (i32) -> index
      %63 = "arith.maxsi"(%29, %62) : (index, index) -> index
      %64 = "arith.minsi"(%61, %63) : (index, index) -> index
      %65 = "arith.subi"(%64, %29) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %66 = "arith.index_cast"(%59) : (i32) -> index
      %67 = "arith.maxsi"(%66, %2) : (index, index) -> index
      %68 = "arith.minsi"(%67, %0) : (index, index) -> index
      %69 = "arith.minsi"(%65, %0) : (index, index) -> index
      %70 = "arith.maxsi"(%69, %2) : (index, index) -> index
      %71 = "arith.minsi"(%68, %0) : (index, index) -> index
      %72 = "arith.maxsi"(%71, %2) : (index, index) -> index
      %73 = "arith.cmpi"(%70, %0) <{predicate = 2 : i64}> : (index, index) -> i1
      %74 = "arith.cmpi"(%72, %0) <{predicate = 2 : i64}> : (index, index) -> i1
      %75 = "arith.ori"(%73, %74) : (i1, i1) -> i1
      "scf.if"(%75) ({
        "linalg.fill"(%8, %60) <{operandSegmentSizes = array<i32: 1, 1>}> ({
        ^bb0(%arg26: f32, %arg27: f32):
          "linalg.yield"(%arg26) : (f32) -> ()
        }) : (f32, memref<64x64xf32>) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) {hivm.unlikely_condition} : (i1) -> ()
      %76 = "memref.subview"(%55, %70, %72) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<64x64xf32, strided<[?, ?], offset: ?>>, index, index) -> memref<?x?xf32, strided<[?, ?], offset: ?>>
      %77 = "memref.subview"(%60, %70, %72) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<64x64xf32>, index, index) -> memref<?x?xf32, strided<[64, 1]>>
      "memref.copy"(%76, %77) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[64, 1]>>) -> ()
      %78 = "bufferization.to_tensor"(%60) <{restrict, writable}> : (memref<64x64xf32>) -> tensor<64x64xf32>
      %79 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x64xf32>
      %80 = "arith.addi"(%33, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %81 = "arith.index_cast"(%arg6) : (i32) -> index
      %82 = "arith.maxsi"(%33, %81) : (index, index) -> index
      %83 = "arith.minsi"(%80, %82) : (index, index) -> index
      %84 = "arith.subi"(%83, %33) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %85 = "arith.minsi"(%84, %0) : (index, index) -> index
      %86 = "arith.maxsi"(%85, %2) : (index, index) -> index
      %87 = "arith.cmpi"(%86, %0) <{predicate = 2 : i64}> : (index, index) -> i1
      %88 = "arith.ori"(%74, %87) : (i1, i1) -> i1
      "scf.if"(%88) ({
        "linalg.fill"(%8, %79) <{operandSegmentSizes = array<i32: 1, 1>}> ({
        ^bb0(%arg24: f32, %arg25: f32):
          "linalg.yield"(%arg24) : (f32) -> ()
        }) : (f32, memref<64x64xf32>) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) {hivm.unlikely_condition} : (i1) -> ()
      %89 = "memref.subview"(%57, %72, %86) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<64x64xf32, strided<[?, ?], offset: ?>>, index, index) -> memref<?x?xf32, strided<[?, ?], offset: ?>>
      %90 = "memref.subview"(%79, %72, %86) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<64x64xf32>, index, index) -> memref<?x?xf32, strided<[64, 1]>>
      "memref.copy"(%89, %90) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[64, 1]>>) -> ()
      %91 = "bufferization.to_tensor"(%79) <{restrict, writable}> : (memref<64x64xf32>) -> tensor<64x64xf32>
      %92 = "linalg.matmul"(%78, %91, %arg18) <{operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%arg21: f32, %arg22: f32, %arg23: f32):
        %96 = "arith.mulf"(%arg21, %arg22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %97 = "arith.addf"(%arg23, %96) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "linalg.yield"(%97) : (f32) -> ()
      }) {input_precision = "ieee", linalg.memoized_indexing_maps = [#map, #map1, #map2]} : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
      %93 = "arith.addi"(%arg19, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %94 = "arith.index_cast"(%28) : (i32) -> index
      %95 = "arith.addi"(%arg20, %94) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "scf.yield"(%92, %93, %95) : (tensor<64x64xf32>, index, index) -> ()
    }) : (i32, i32, i32, tensor<64x64xf32>, index, index) -> (tensor<64x64xf32>, index, index)
    %35 = "arith.index_cast"(%arg10) : (i32) -> index
    %36 = "arith.muli"(%29, %35) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %37 = "arith.addi"(%36, %33) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %38 = "memref.reinterpret_cast"(%arg4, %37, %35) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: -9223372036854775808, 1>}> : (memref<?xf32>, index, index) -> memref<64x64xf32, strided<[?, 1], offset: ?>>
    %39 = "arith.addi"(%29, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %40 = "arith.index_cast"(%arg5) : (i32) -> index
    %41 = "arith.maxsi"(%29, %40) : (index, index) -> index
    %42 = "arith.minsi"(%39, %41) : (index, index) -> index
    %43 = "arith.subi"(%42, %29) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %44 = "arith.addi"(%33, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %45 = "arith.index_cast"(%arg6) : (i32) -> index
    %46 = "arith.maxsi"(%33, %45) : (index, index) -> index
    %47 = "arith.minsi"(%44, %46) : (index, index) -> index
    %48 = "arith.subi"(%47, %33) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    %49 = "arith.minsi"(%43, %0) : (index, index) -> index
    %50 = "arith.maxsi"(%49, %2) : (index, index) -> index
    %51 = "arith.minsi"(%48, %0) : (index, index) -> index
    %52 = "arith.maxsi"(%51, %2) : (index, index) -> index
    %53 = "tensor.extract_slice"(%34#0, %50, %52) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (tensor<64x64xf32>, index, index) -> tensor<?x?xf32>
    %54 = "memref.subview"(%38, %50, %52) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<64x64xf32, strided<[?, 1], offset: ?>>, index, index) -> memref<?x?xf32, strided<[?, 1], offset: ?>>
    "bufferization.materialize_in_destination"(%53, %54) <{writable}> : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    "func.return"() : () -> ()
  }) {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} : () -> ()
}) {hacc.target = #hacc.target<"Ascend910_9382">} : () -> ()

