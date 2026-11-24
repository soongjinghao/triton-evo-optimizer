#map = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<()[s0, s1] -> (s1, s0)>
#map2 = affine_map<()[s0, s1] -> (s1 + 64, s0)>
#map3 = affine_map<()[s0, s1] -> (64, s0 - s1)>
#map4 = affine_map<()[s0] -> (0, s0)>
#map5 = affine_map<()[s0, s1, s2, s3] -> ((s0 + ((s2 - s3) floordiv 24) * s1) mod 2)>
#map6 = affine_map<()[s0, s1, s2, s3] -> (s0 + ((s2 - s3) floordiv 24) * s1)>
#map7 = affine_map<()[s0, s1] -> (s0 + s1)>
#map8 = affine_map<()[s0] -> (64, s0)>
#map9 = affine_map<()[s0] -> ((s0 + 15) floordiv 16)>
#map10 = affine_map<()[s0] -> ((s0 + 7) floordiv 8)>
#map11 = affine_map<()[s0] -> (s0 + 64)>
#map12 = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
"builtin.module"() ({
  "func.func"() <{function_type = () -> i8, sym_name = "matmul_kernel_simplified_infer_task_type_function"}> ({
    %143 = "arith.constant"() <{value = 20 : i8}> : () -> i8
    "func.return"(%143) : (i8) -> ()
  }) {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} : () -> ()
  "func.func"() <{function_type = (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> (), sym_name = "set_l1_2d_float", sym_visibility = "private"}> ({
  }) {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface} : () -> ()
  "func.func"() <{function_type = (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> (), sym_name = "nd2nz_float", sym_visibility = "private"}> ({
  }) {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface} : () -> ()
  "func.func"() <{function_type = (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> (), sym_name = "mma_tile_float_to_float", sym_visibility = "private"}> ({
  }) {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface} : () -> ()
  "func.func"() <{function_type = (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> (), sym_name = "fixpipe_nz2nd_float_to_float_4d_to_2d", sym_visibility = "private"}> ({
  }) {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface} : () -> ()
  "func.func"() <{arg_attrs = [{hacc.arg_type = #hacc.arg_type<ffts_base_address>}, {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, {hacc.arg_type = #hacc.arg_type<workspace>}, {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {}, {}, {}], function_type = (i64, memref<?xi8, #hivm.address_space<gm>>, memref<?xi8, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, memref<?xf32, #hivm.address_space<gm>>, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (), sym_name = "matmul_kernel_simplified"}> ({
  ^bb0(%arg0: i64, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>>, %arg3: memref<?xf32, #hivm.address_space<gm>>, %arg4: memref<?xf32, #hivm.address_space<gm>>, %arg5: memref<?xf32, #hivm.address_space<gm>>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32):
    %0 = "arith.constant"() <{value = false}> : () -> i1
    %1 = "arith.constant"() <{value = 0 : i8}> : () -> i8
    %2 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 64 : index}> : () -> index
    %5 = "arith.constant"() <{value = 0 : index}> : () -> index
    %6 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 63 : i32}> : () -> i32
    %8 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %9 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %11 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %12 = "arith.constant"() <{value = 3 : i64}> : () -> i64
    %13 = "arith.constant"() <{value = 4 : i64}> : () -> i64
    %14 = "arith.constant"() <{value = 2 : i64}> : () -> i64
    %15 = "arith.constant"() <{value = -1 : i64}> : () -> i64
    %16 = "arith.constant"() <{value = 32768 : i64}> : () -> i64
    %17 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %18 = "arith.constant"() <{value = 16384 : i64}> : () -> i64
    %19 = "arith.constant"() <{value = 49152 : i64}> : () -> i64
    %20 = "hivm.hir.pointer_cast"(%18) <{operandSegmentSizes = array<i32: 1, 0>}> : (i64) -> memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %21 = "hivm.hir.pointer_cast"(%19) <{operandSegmentSizes = array<i32: 1, 0>}> : (i64) -> memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %22 = "hivm.hir.pointer_cast"(%17) <{operandSegmentSizes = array<i32: 1, 0>}> : (i64) -> memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %23 = "hivm.hir.pointer_cast"(%16) <{operandSegmentSizes = array<i32: 1, 0>}> : (i64) -> memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
    %24 = "arith.muli"(%arg12, %arg13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %25 = "arith.muli"(%24, %arg14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %26 = "hivm.hir.get_block_idx"() : () -> i64
    %27 = "arith.trunci"(%26) : (i64) -> i32
    %28 = "arith.muli"(%arg14, %arg13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %29 = "arith.addi"(%arg6, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %30 = "arith.divsi"(%29, %9) : (i32, i32) -> i32
    %31 = "arith.addi"(%arg7, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %32 = "arith.divsi"(%31, %9) : (i32, i32) -> i32
    %33 = "arith.muli"(%32, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %34 = "arith.addi"(%arg8, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %35 = "arith.divsi"(%34, %9) : (i32, i32) -> i32
    %36 = "arith.muli"(%arg10, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %37 = "arith.index_cast"(%arg9) : (i32) -> index
    %38 = "arith.index_cast"(%arg10) : (i32) -> index
    %39 = "arith.index_cast"(%arg6) : (i32) -> index
    %40 = "arith.index_cast"(%arg7) : (i32) -> index
    %41 = "arith.index_cast"(%36) : (i32) -> index
    %42 = "arith.index_cast"(%arg11) : (i32) -> index
    "hivm.hir.set_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID1>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
    "hivm.hir.set_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID2>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
    "hivm.hir.set_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID3>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
    "hivm.hir.set_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID4>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
    "hivm.hir.set_flag"() <{set_pipe = #hivm.pipe<PIPE_FIX>, static_event_id = #hivm.event<EVENT_ID0>, wait_pipe = #hivm.pipe<PIPE_M>}> : () -> ()
    "scf.for"(%27, %25, %2) ({
    ^bb0(%arg15: i32):
      "hivm.hir.set_mask_norm"() : () -> ()
      %43 = "arith.divsi"(%arg15, %28) : (i32, i32) -> i32
      %44 = "arith.remsi"(%43, %arg12) : (i32, i32) -> i32
      %45 = "arith.divsi"(%44, %33) : (i32, i32) -> i32
      %46 = "arith.muli"(%45, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %47 = "arith.subi"(%30, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %48 = "arith.minsi"(%47, %6) : (i32, i32) -> i32
      %49 = "arith.remsi"(%44, %48) : (i32, i32) -> i32
      %50 = "arith.addi"(%46, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %51 = "arith.remsi"(%44, %33) : (i32, i32) -> i32
      %52 = "arith.divsi"(%51, %48) : (i32, i32) -> i32
      %53 = "arith.muli"(%50, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %54 = "arith.muli"(%52, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %55 = "arith.index_cast"(%53) : (i32) -> index
      %56 = "affine.apply"(%55, %37) <{map = #map}> : (index, index) -> index
      %57 = "arith.index_cast"(%54) : (i32) -> index
      %58 = "affine.max"(%55, %39) <{map = #map1}> : (index, index) -> index
      %59 = "affine.min"(%58, %55) <{map = #map2}> : (index, index) -> index
      %60 = "affine.min"(%59, %55) <{map = #map3}> : (index, index) -> index
      %61 = "affine.max"(%60) <{map = #map4}> : (index) -> index
      %62 = "arith.cmpi"(%61, %4) <{predicate = 2 : i64}> : (index, index) -> i1
      %63 = "affine.max"(%57, %40) <{map = #map1}> : (index, index) -> index
      %64 = "affine.min"(%63, %57) <{map = #map2}> : (index, index) -> index
      %65 = "affine.min"(%64, %57) <{map = #map3}> : (index, index) -> index
      %66 = "affine.max"(%65) <{map = #map4}> : (index) -> index
      %67 = "arith.cmpi"(%66, %4) <{predicate = 2 : i64}> : (index, index) -> i1
      %68 = "hivm.hir.pointer_cast"(%17) <{operandSegmentSizes = array<i32: 1, 0>}> : (i64) -> memref<4x4x16x16xf32, #hivm.address_space<cc>>
      "hivm.hir.wait_flag"() <{set_pipe = #hivm.pipe<PIPE_FIX>, static_event_id = #hivm.event<EVENT_ID0>, wait_pipe = #hivm.pipe<PIPE_M>}> : () -> ()
      %69:2 = "scf.for"(%8, %35, %3, %56, %5) ({
      ^bb0(%arg16: i32, %arg17: index, %arg18: index):
        %78 = "arith.index_cast"(%arg16) : (i32) -> index
        %79 = "arith.index_cast"(%35) : (i32) -> index
        %80 = "arith.index_cast"(%arg15) : (i32) -> index
        %81 = "arith.index_cast"(%27) : (i32) -> index
        %82 = "affine.apply"(%78, %79, %80, %81) <{map = #map5}> : (index, index, index, index) -> index
        %83 = "arith.index_cast"(%82) : (index) -> i64
        %84 = "arith.cmpi"(%83, %11) <{predicate = 0 : i64}> : (i64, i64) -> i1
        %85 = "arith.select"(%84, %21, %20) : (i1, memref<8x4x16x8xf32, #hivm.address_space<cbuf>>, memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        %86 = "arith.index_cast"(%arg16) : (i32) -> index
        %87 = "arith.index_cast"(%35) : (i32) -> index
        %88 = "arith.index_cast"(%arg15) : (i32) -> index
        %89 = "arith.index_cast"(%27) : (i32) -> index
        %90 = "affine.apply"(%86, %87, %88, %89) <{map = #map5}> : (index, index, index, index) -> index
        %91 = "arith.index_cast"(%90) : (index) -> i64
        %92 = "arith.cmpi"(%91, %11) <{predicate = 0 : i64}> : (i64, i64) -> i1
        %93 = "arith.select"(%92, %23, %22) : (i1, memref<8x4x16x8xf32, #hivm.address_space<cbuf>>, memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<8x4x16x8xf32, #hivm.address_space<cbuf>>
        %94 = "arith.index_cast"(%arg16) : (i32) -> index
        %95 = "arith.index_cast"(%35) : (i32) -> index
        %96 = "arith.index_cast"(%arg15) : (i32) -> index
        %97 = "arith.index_cast"(%27) : (i32) -> index
        %98 = "affine.apply"(%94, %95, %96, %97) <{map = #map6}> : (index, index, index, index) -> index
        %99 = "arith.index_cast"(%98) : (index) -> i64
        %100 = "arith.index_cast"(%arg16) : (i32) -> index
        %101 = "arith.index_cast"(%35) : (i32) -> index
        %102 = "arith.index_cast"(%arg15) : (i32) -> index
        %103 = "arith.index_cast"(%27) : (i32) -> index
        %104 = "affine.apply"(%100, %101, %102, %103) <{map = #map5}> : (index, index, index, index) -> index
        %105 = "arith.index_cast"(%104) : (index) -> i1
        %106 = "arith.select"(%105, %12, %13) : (i1, i64, i64) -> i64
        %107 = "arith.index_cast"(%104) : (index) -> i1
        %108 = "arith.select"(%107, %11, %14) : (i1, i64, i64) -> i64
        %109 = "memref.reinterpret_cast"(%arg3, %arg17, %37) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: -9223372036854775808, 1>}> : (memref<?xf32, #hivm.address_space<gm>>, index, index) -> memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %110 = "affine.apply"(%arg18, %57) <{map = #map7}> : (index, index) -> index
        %111 = "memref.reinterpret_cast"(%arg4, %110, %38) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: -9223372036854775808, 1>}> : (memref<?xf32, #hivm.address_space<gm>>, index, index) -> memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %112 = "arith.muli"(%arg16, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %113 = "arith.subi"(%arg8, %112) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %114 = "arith.index_cast"(%113) : (i32) -> index
        %115 = "affine.max"(%114) <{map = #map4}> : (index) -> index
        %116 = "affine.min"(%115) <{map = #map8}> : (index) -> index
        %117 = "affine.max"(%116) <{map = #map4}> : (index) -> index
        %118 = "arith.cmpi"(%117, %4) <{predicate = 2 : i64}> : (index, index) -> i1
        %119 = "arith.ori"(%62, %118) : (i1, i1) -> i1
        %120 = "memref.subview"(%109, %61, %117) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>, index, index) -> memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %121 = "affine.apply"(%61) <{map = #map9}> : (index) -> index
        %122 = "affine.apply"(%117) <{map = #map10}> : (index) -> index
        %123 = "memref.subview"(%93, %122, %121) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<8x4x16x8xf32, #hivm.address_space<cbuf>>, index, index) -> memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
        "scf.if"(%119) ({
          "hivm.hir.set_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID0>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
          %141 = "memref.collapse_shape"(%93) <{reassociation = [[0, 1, 2, 3]]}> : (memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<4096xf32, #hivm.address_space<cbuf>>
          "hivm.hir.wait_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID0>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
          %142 = "memref.cast"(%141) : (memref<4096xf32, #hivm.address_space<cbuf>>) -> memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          "func.call"(%10, %142) <{callee = @set_l1_2d_float}> : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          "hivm.hir.pipe_barrier"() <{pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
          "scf.yield"() : () -> ()
        }, {
        }) {hivm.unlikely_condition} : (i1) -> ()
        "hivm.hir.wait_flag"(%108) <{set_pipe = #hivm.pipe<PIPE_MTE1>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : (i64) -> ()
        %124 = "memref.cast"(%120) : (memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %125 = "memref.cast"(%123) : (memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>) -> memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        "func.call"(%124, %125) <{callee = @nd2nz_float}> : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        "hivm.hir.set_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE2>, static_event_id = #hivm.event<EVENT_ID0>, wait_pipe = #hivm.pipe<PIPE_MTE1>}> : () -> ()
        %126 = "arith.ori"(%118, %67) : (i1, i1) -> i1
        %127 = "memref.subview"(%111, %117, %66) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>, index, index) -> memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %128 = "affine.apply"(%117) <{map = #map9}> : (index) -> index
        %129 = "affine.apply"(%66) <{map = #map10}> : (index) -> index
        %130 = "memref.subview"(%85, %129, %128) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808, 16, 8>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<8x4x16x8xf32, #hivm.address_space<cbuf>>, index, index) -> memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>
        "scf.if"(%126) ({
          "hivm.hir.set_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID0>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
          %139 = "memref.collapse_shape"(%85) <{reassociation = [[0, 1, 2, 3]]}> : (memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<4096xf32, #hivm.address_space<cbuf>>
          "hivm.hir.wait_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID0>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
          %140 = "memref.cast"(%139) : (memref<4096xf32, #hivm.address_space<cbuf>>) -> memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          "func.call"(%10, %140) <{callee = @set_l1_2d_float}> : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          "hivm.hir.pipe_barrier"() <{pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
          "scf.yield"() : () -> ()
        }, {
        }) {hivm.unlikely_condition} : (i1) -> ()
        "hivm.hir.wait_flag"(%106) <{set_pipe = #hivm.pipe<PIPE_MTE1>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : (i64) -> ()
        %131 = "memref.cast"(%127) : (memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %132 = "memref.cast"(%130) : (memref<?x?x16x8xf32, strided<[512, 128, 8, 1]>, #hivm.address_space<cbuf>>) -> memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        "func.call"(%131, %132) <{callee = @nd2nz_float}> : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        "hivm.hir.set_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE2>, static_event_id = #hivm.event<EVENT_ID1>, wait_pipe = #hivm.pipe<PIPE_MTE1>}> : () -> ()
        %133 = "arith.cmpi"(%arg16, %8) <{predicate = 0 : i64}> : (i32, i32) -> i1
        %134 = "memref.cast"(%93) : (memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %135 = "memref.cast"(%85) : (memref<8x4x16x8xf32, #hivm.address_space<cbuf>>) -> memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %136 = "memref.cast"(%68) : (memref<4x4x16x16xf32, #hivm.address_space<cc>>) -> memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        "func.call"(%134, %135, %133, %61, %117, %66, %136, %17, %11, %108, %106, %99, %15, %15, %1) <{callee = @mma_tile_float_to_float}> : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        %137 = "affine.apply"(%arg17) <{map = #map11}> : (index) -> index
        %138 = "affine.apply"(%arg18, %41) <{map = #map7}> : (index, index) -> index
        "scf.yield"(%137, %138) : (index, index) -> ()
      }) {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly} : (i32, i32, i32, index, index) -> (index, index)
      "hivm.hir.set_flag"() <{set_pipe = #hivm.pipe<PIPE_M>, static_event_id = #hivm.event<EVENT_ID0>, wait_pipe = #hivm.pipe<PIPE_FIX>}> : () -> ()
      %70 = "affine.apply"(%57, %55, %42) <{map = #map12}> : (index, index, index) -> index
      %71 = "memref.reinterpret_cast"(%arg5, %70, %42) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 64, 64>, static_strides = array<i64: -9223372036854775808, 1>}> : (memref<?xf32, #hivm.address_space<gm>>, index, index) -> memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %72 = "affine.apply"(%61) <{map = #map9}> : (index) -> index
      %73 = "affine.apply"(%66) <{map = #map9}> : (index) -> index
      %74 = "memref.subview"(%68, %73, %72) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808, 16, 16>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<4x4x16x16xf32, #hivm.address_space<cc>>, index, index) -> memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>>
      %75 = "memref.subview"(%71, %61, %66) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: 1, 1>}> : (memref<64x64xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>, index, index) -> memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      "hivm.hir.wait_flag"() <{set_pipe = #hivm.pipe<PIPE_M>, static_event_id = #hivm.event<EVENT_ID0>, wait_pipe = #hivm.pipe<PIPE_FIX>}> : () -> ()
      %76 = "memref.cast"(%74) : (memref<?x?x16x16xf32, strided<[1024, 256, 16, 1]>, #hivm.address_space<cc>>) -> memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
      %77 = "memref.cast"(%75) : (memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) -> memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      "func.call"(%76, %77, %17, %17, %0, %1) <{callee = @fixpipe_nz2nd_float_to_float_4d_to_2d}> : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
      "hivm.hir.set_flag"() <{set_pipe = #hivm.pipe<PIPE_FIX>, static_event_id = #hivm.event<EVENT_ID0>, wait_pipe = #hivm.pipe<PIPE_M>}> : () -> ()
      "scf.yield"() : () -> ()
    }) {autoblockify.subloop} : (i32, i32, i32) -> ()
    "hivm.hir.wait_flag"() <{set_pipe = #hivm.pipe<PIPE_FIX>, static_event_id = #hivm.event<EVENT_ID0>, wait_pipe = #hivm.pipe<PIPE_M>}> : () -> ()
    "hivm.hir.wait_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID1>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
    "hivm.hir.wait_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID2>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
    "hivm.hir.wait_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID3>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
    "hivm.hir.wait_flag"() <{set_pipe = #hivm.pipe<PIPE_MTE1>, static_event_id = #hivm.event<EVENT_ID4>, wait_pipe = #hivm.pipe<PIPE_MTE2>}> : () -> ()
    "hivm.hir.pipe_barrier"() <{pipe = #hivm.pipe<PIPE_ALL>}> : () -> ()
    "func.return"() : () -> ()
  }) {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<15xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} : () -> ()
}) {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIC>, memref.memref_as_ptr} : () -> ()

