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
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c8192_i32 = arith.constant 8192 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8191_i32 = arith.constant 8191 : i32
    %c0_i64 = arith.constant 0 : i64
    %c32800_i64 = arith.constant 32800 : i64
    %c32832_i64 = arith.constant 32832 : i64
    %c1_i64 = arith.constant 1 : i64
    %c49248_i64 = arith.constant 49248 : i64
    %c32_i64 = arith.constant 32 : i64
    %c49216_i64 = arith.constant 49216 : i64
    %c82016_i64 = arith.constant 82016 : i64
    %0 = hivm.hir.pointer_cast(%c49216_i64) : memref<1xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c82016_i64) : memref<1xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c32_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c49248_i64) : memref<8192xf32, #hivm.address_space<ub>>
    %4 = arith.muli %arg8, %arg9 : i32
    %5 = arith.muli %4, %arg10 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = arith.muli %arg10, %arg9 : i32
    %9 = arith.sitofp %arg7 : i32 to f32
    %10 = arith.divf %cst, %9 : f32
    %11 = arith.addi %arg7, %c8191_i32 : i32
    %12 = arith.divsi %11, %c8192_i32 : i32
    %13 = arith.index_cast %arg7 : i32 to index
    %14 = hivm.hir.pointer_cast(%c0_i64) : memref<1xf32, #hivm.address_space<ub>>
    memref.store %10, %14[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    scf.for %arg11 = %7 to %5 step %c48_i32  : i32 {
      hivm.hir.set_mask_norm
      %15 = arith.divsi %arg11, %8 : i32
      %16 = arith.remsi %15, %arg8 : i32
      %17 = arith.muli %16, %c2_i32 : i32
      %18 = arith.addi %17, %c2_i32 : i32
      %19 = arith.minsi %18, %arg6 : i32
      %20 = arith.subi %19, %17 : i32
      scf.for %arg12 = %c0_i32 to %20 step %c1_i32  : i32 {
        %21 = arith.index_cast %arg12 : i32 to index
        %22 = arith.index_cast %20 : i32 to index
        %23 = arith.index_cast %arg11 : i32 to index
        %24 = arith.index_cast %7 : i32 to index
        %25 = affine.apply #map()[%21, %22, %23, %24]
        %26 = arith.index_cast %25 : index to i64
        %27 = arith.cmpi eq, %26, %c1_i64 : i64
        %28 = arith.select %27, %1, %0 : memref<1xf32, #hivm.address_space<ub>>
        %29 = arith.index_cast %arg12 : i32 to index
        %30 = arith.index_cast %20 : i32 to index
        %31 = arith.index_cast %arg11 : i32 to index
        %32 = arith.index_cast %7 : i32 to index
        %33 = affine.apply #map()[%29, %30, %31, %32]
        %34 = arith.index_cast %33 : index to i1
        %35 = arith.xori %34, %true : i1
        %36 = arith.extui %35 : i1 to i64
        %37 = arith.addi %17, %arg12 : i32
        %38 = arith.muli %37, %arg5 : i32
        %39 = arith.index_cast %38 : i32 to index
        %40 = scf.for %arg13 = %c0_i32 to %12 step %c1_i32 iter_args(%arg14 = %cst_0) -> (f32)  : i32 {
          %44 = arith.index_cast %arg13 : i32 to index
          %45 = arith.index_cast %12 : i32 to index
          %46 = arith.index_cast %arg12 : i32 to index
          %47 = arith.index_cast %20 : i32 to index
          %48 = arith.index_cast %arg11 : i32 to index
          %49 = arith.index_cast %7 : i32 to index
          %50 = affine.apply #map1()[%44, %45, %46, %47, %48, %49]
          %51 = arith.index_cast %50 : index to i64
          %52 = arith.cmpi eq, %51, %c1_i64 : i64
          %53 = arith.select %52, %3, %2 : memref<8192xf32, #hivm.address_space<ub>>
          %54 = arith.muli %arg13, %c8192_i32 : i32
          %55 = arith.index_cast %54 : i32 to index
          %56 = affine.apply #map2()[%39, %55]
          %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%56], sizes: [8192], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %57 = affine.max #map3()[%55, %13]
          %58 = affine.min #map4()[%57, %55]
          %59 = affine.apply #map5()[%58, %55]
          %60 = arith.cmpi slt, %59, %c8192 : index
          %subview = memref.subview %reinterpret_cast_2[0] [%59] [1] : memref<8192xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_3 = memref.subview %53[0] [%59] [1] : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %60 {
            %cast_9 = memref.cast %53 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_9) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %cast_4 = memref.cast %subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_5 = memref.cast %subview_3 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_4, %cast_5, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %61 = hivm.hir.pointer_cast(%c32800_i64) : memref<1xf32, #hivm.address_space<ub>>
          %62 = hivm.hir.pointer_cast(%c32832_i64) : memref<4096xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %cast_6 = memref.cast %53 : memref<8192xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_7 = memref.cast %61 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_8 = memref.cast %62 : memref<4096xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @enablevc_reduce_sum_r_float(%cast_6, %cast_7, %cast_8, %cst_0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
          %63 = memref.load %61[%c0] : memref<1xf32, #hivm.address_space<ub>>
          hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
          %64 = arith.addf %arg14, %63 : f32
          scf.yield %64 : f32
        } {pipeline.veconly}
        %41 = memref.load %14[%c0] : memref<1xf32, #hivm.address_space<ub>>
        %42 = arith.mulf %40, %41 : f32
        %43 = arith.index_cast %37 : i32 to index
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, %36]
        memref.store %42, %28[%c0] : memref<1xf32, #hivm.address_space<ub>>
        hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast = memref.cast %28 : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_1 = memref.cast %reinterpret_cast : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast, %cast_1, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, %36]
      } {pipeline.veconly}
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
