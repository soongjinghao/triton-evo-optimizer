#map = affine_map<()[s0, s1, s2, s3] -> ((s0 + ((s2 - s3) floordiv 48) * s1) mod 2)>
#map1 = affine_map<()[s0, s1, s2, s3, s4, s5] -> ((s0 + s2 * s1 + ((s4 - s5) floordiv 48) * (s1 * s3)) mod 2)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1)>
#map3 = affine_map<()[s0, s1] -> (s1, s0)>
#map4 = affine_map<()[s0, s1] -> (s1 + 8192, s0)>
#map5 = affine_map<()[s0, s1] -> (s0 - s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @mean_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c10_i8 = arith.constant 10 : i8
    return %c10_i8 : i8
  }
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @enablevc_reduce_sum_r_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @mean_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %true = arith.constant true
    %c48_i32 = arith.constant 48 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8192 = arith.constant 8192 : index
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c8192_i32 = arith.constant 8192 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8191_i32 = arith.constant 8191 : i32
    %c32768_i64 = arith.constant 32768 : i64
    %c32800_i64 = arith.constant 32800 : i64
    %c1_i64 = arith.constant 1 : i64
    %c49216_i64 = arith.constant 49216 : i64
    %c0_i64 = arith.constant 0 : i64
    %c49184_i64 = arith.constant 49184 : i64
    %c81984_i64 = arith.constant 81984 : i64
    %0 = hivm.hir.pointer_cast(%c49184_i64) : memref<1xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c81984_i64) : memref<1xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c49216_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %4 = arith.muli %arg8, %arg9 : i32
    %5 = arith.muli %4, %arg10 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg10, %arg9 : i32
    %9 = arith.addi %arg7, %c8191_i32 : i32
    %10 = arith.divsi %9, %c8192_i32 : i32
    %11 = arith.sitofp %arg7 : i32 to f32
    %12 = arith.index_cast %arg7 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    scf.for %arg11 = %7 to %5 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %13 = arith.divsi %arg11, %8 : i32
      %14 = arith.remsi %13, %arg8 : i32
      %15 = arith.muli %14, %c2_i32 : i32
      %16 = arith.addi %15, %c2_i32 : i32
      %17 = arith.minsi %16, %arg6 : i32
      %18 = arith.subi %17, %15 : i32
      scf.for %arg12 = %c0_i32 to %18 step %c1_i32  : i32 {
        %19 = arith.index_cast %arg12 : i32 to index
        %20 = arith.index_cast %18 : i32 to index
        %21 = arith.index_cast %arg11 : i32 to index
        %22 = arith.index_cast %7 : i32 to index
        %23 = affine.apply #map()[%19, %20, %21, %22]
        %24 = arith.index_cast %23 : index to i64
        %25 = arith.cmpi eq, %24, %c1_i64 : i64
        %26 = arith.select %25, %1, %0 : memref<1xf32, #hivm.address_space<ub>>
        %27 = arith.index_cast %arg12 : i32 to index
        %28 = arith.index_cast %18 : i32 to index
        %29 = arith.index_cast %arg11 : i32 to index
        %30 = arith.index_cast %7 : i32 to index
        %31 = affine.apply #map()[%27, %28, %29, %30]
        %32 = arith.index_cast %31 : index to i1
        %33 = arith.xori %32, %true : i1
        %34 = arith.extui %33 : i1 to i64
        %35 = arith.addi %15, %arg12 : i32
        %36 = arith.muli %35, %arg5 : i32
        %37 = arith.index_cast %36 : i32 to index
        %38 = scf.for %arg13 = %c0_i32 to %10 step %c1_i32 iter_args(%arg14 = %cst) -> (f32)  : i32 {
          %41 = arith.index_cast %arg13 : i32 to index
          %42 = arith.index_cast %10 : i32 to index
          %43 = arith.index_cast %arg12 : i32 to index
          %44 = arith.index_cast %18 : i32 to index
          %45 = arith.index_cast %arg11 : i32 to index
          %46 = arith.index_cast %7 : i32 to index
          %47 = affine.apply #map1()[%41, %42, %43, %44, %45, %46]
          %48 = arith.index_cast %47 : index to i64
          %49 = arith.cmpi eq, %48, %c1_i64 : i64
          %50 = arith.select %49, %3, %2 : memref<8192xf32, #hivm.address_space<ub>>
          %51 = arith.muli %arg13, %c8192_i32 : i32
          %52 = arith.index_cast %51 : i32 to index
          %53 = affine.apply #map2()[%37, %52]
          %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%53], sizes: [8192], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %54 = affine.max #map3()[%52, %12]
          %55 = affine.min #map4()[%54, %52]
          %56 = affine.apply #map5()[%55, %52]
          %57 = arith.cmpi slt, %56, %c8192 : index
          %subview = memref.subview %reinterpret_cast_1[0] [%56] [1] : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_2 = memref.subview %50[0] [%56] [1] : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %57 {
            %cast_8 = memref.cast %50 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst, %cast_8) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %cast_3 = memref.cast %subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_4 = memref.cast %subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_3, %cast_4, %c2_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %58 = hivm.hir.pointer_cast(%c32768_i64) : memref<1xf32, #hivm.address_space<ub>>
          %59 = hivm.hir.pointer_cast(%c32800_i64) : memref<4096xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %cast_5 = memref.cast %50 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_6 = memref.cast %58 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_7 = memref.cast %59 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @enablevc_reduce_sum_r_float(%cast_5, %cast_6, %cast_7, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          %60 = memref.load %58[%c0] : memref<1xf32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %61 = arith.addf %arg14, %60 : f32
          scf.yield %61 : f32
        } {pipeline.veconly}
        %39 = arith.divf %38, %11 : f32
        %40 = arith.index_cast %35 : i32 to index
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %34]
        memref.store %39, %26[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%40], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast = memref.cast %26 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_0 = memref.cast %reinterpret_cast : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast, %cast_0, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %34]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
