#map = affine_map<()[s0, s1] -> (s1, s0)>
#map1 = affine_map<()[s0, s1] -> (s1 + 16, s0)>
#map2 = affine_map<()[s0, s1] -> (16, s0 - s1)>
#map3 = affine_map<()[s0] -> (0, s0)>
#map4 = affine_map<(d0) -> (d0 * 8192)>
#map5 = affine_map<(d0) -> (d0 * 8192 + 4096)>
#map6 = affine_map<(d0, d1) -> ((d0 - d1) ceildiv 16, 4)>
#map7 = affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7] -> ((s0 + (s2 floordiv 64) * s1 + s4 * (s1 * ((s3 + 63) floordiv 64)) + ((s6 - s7) floordiv 24) * ((s1 * ((s3 + 63) floordiv 64)) * s5)) mod 2)>
#map8 = affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7] -> (s0 + (s2 floordiv 64) * s1 + s4 * (s1 * ((s3 + 63) floordiv 64)) + ((s6 - s7) floordiv 24) * ((s1 * ((s3 + 63) floordiv 64)) * s5))>
#map9 = affine_map<()[s0] -> (s0 + 4)>
#map10 = affine_map<(d0)[s0] -> (d0 * 16 + s0)>
#map11 = affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>
#map12 = affine_map<()[s0] -> ((s0 + 15) floordiv 16)>
#map13 = affine_map<()[s0, s1] -> (s0 + s1)>
#map14 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
#map15 = affine_map<()[s0] -> (16, s0)>
#map16 = affine_map<()[s0] -> (s0 + 16)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @_chunk_state_fwd_kernel_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c8192 = arith.constant 8192 : index
    return %c8192 : index
  }
  func.func @_chunk_state_fwd_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @_chunk_state_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32, %arg33: i32, %arg34: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<35xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %true = arith.constant true
    %c64_i64 = arith.constant 64 : i64
    %c1 = arith.constant 1 : index
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c15_i32 = arith.constant 15 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32
    %c24_i32 = arith.constant 24 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c3072_i64 = arith.constant 3072 : i64
    %0 = hivm.hir.pointer_cast(%c1024_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %1 = hivm.hir.pointer_cast(%c3072_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c2048_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_ffts_base_addr %arg0
    %4 = arith.muli %arg32, %arg33 : i32
    %5 = arith.muli %4, %arg34 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7 ffts_base_addr = %arg0
    %8 = arith.muli %arg34, %arg33 : i32
    %9 = arith.extsi %arg11 : i32 to i64
    %10 = arith.addi %arg9, %c15_i32 : i32
    %11 = arith.divsi %10, %c16_i32 : i32
    %12 = arith.extsi %arg10 : i32 to i64
    %13 = arith.extsi %arg12 : i32 to i64
    %14 = arith.extsi %arg20 : i32 to i64
    %15 = arith.extsi %arg21 : i32 to i64
    %16 = arith.index_cast %arg23 : i32 to index
    %17 = arith.index_cast %arg8 : i32 to index
    %18 = arith.index_cast %arg9 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    scf.for %arg35 = %7 to %5 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %19 = arith.remsi %arg35, %arg34 : i32
      %20 = arith.divsi %arg35, %arg34 : i32
      %21 = arith.remsi %20, %arg33 : i32
      %22 = arith.divsi %arg35, %8 : i32
      %23 = arith.remsi %22, %arg32 : i32
      %24 = arith.extsi %21 : i32 to i64
      %25 = arith.divsi %24, %9 : i64
      %26 = arith.muli %25, %9 : i64
      %27 = arith.subi %24, %26 : i64
      %28 = arith.muli %25, %12 : i64
      %29 = arith.muli %23, %c16_i32 : i32
      %30 = arith.subi %13, %28 : i64
      %31 = arith.minsi %12, %30 : i64
      %32 = arith.muli %27, %14 : i64
      %33 = arith.muli %25, %15 : i64
      %34 = arith.addi %32, %33 : i64
      %35 = arith.muli %19, %arg22 : i32
      %36 = arith.extsi %35 : i32 to i64
      %37 = arith.addi %34, %36 : i64
      %38 = arith.index_cast %29 : i32 to index
      %39 = arith.index_cast %31 : i64 to index
      %40 = arith.index_cast %37 : i64 to index
      %41 = affine.max #map()[%38, %17]
      %42 = affine.min #map1()[%41, %38]
      %43 = affine.min #map2()[%42, %38]
      %44 = affine.max #map3()[%43]
      scf.for %arg36 = %c0_i32 to %11 step %c1_i32  : i32 {
        %45 = arith.muli %arg36, %c16_i32 : i32
        %46 = arith.index_cast %45 : i32 to index
        %47 = hivm.hir.get_block_idx -> i64
        %48 = arith.index_cast %47 : i64 to index
        %49 = affine.apply #map4(%48)
        %view = memref.view %arg2[%49][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
        %50 = hivm.hir.get_block_idx -> i64
        %51 = arith.index_cast %50 : i64 to index
        %52 = affine.apply #map5(%51)
        %view_0 = memref.view %arg2[%52][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
        %53 = hivm.hir.pointer_cast(%c0_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        scf.for %arg37 = %c0_i64 to %31 step %c64_i64  : i64 {
          %61 = arith.index_cast %arg37 : i64 to index
          %62 = affine.min #map6(%39, %61)
          scf.for %arg38 = %c0 to %62 step %c1 {
            %63 = arith.index_cast %arg37 : i64 to index
            %64 = arith.index_cast %31 : i64 to index
            %65 = arith.index_cast %arg36 : i32 to index
            %66 = arith.index_cast %11 : i32 to index
            %67 = arith.index_cast %arg35 : i32 to index
            %68 = arith.index_cast %7 : i32 to index
            %69 = affine.apply #map7()[%arg38, %62, %63, %64, %65, %66, %67, %68]
            %70 = arith.index_cast %69 : index to i64
            %71 = arith.cmpi eq, %70, %c1_i64 : i64
            %72 = arith.select %71, %1, %0 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
            %73 = arith.index_cast %arg37 : i64 to index
            %74 = arith.index_cast %31 : i64 to index
            %75 = arith.index_cast %arg36 : i32 to index
            %76 = arith.index_cast %11 : i32 to index
            %77 = arith.index_cast %arg35 : i32 to index
            %78 = arith.index_cast %7 : i32 to index
            %79 = affine.apply #map7()[%arg38, %62, %73, %74, %75, %76, %77, %78]
            %80 = arith.index_cast %79 : index to i64
            %81 = arith.cmpi eq, %80, %c1_i64 : i64
            %82 = arith.select %81, %3, %2 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
            %83 = arith.index_cast %arg37 : i64 to index
            %84 = arith.index_cast %31 : i64 to index
            %85 = arith.index_cast %arg36 : i32 to index
            %86 = arith.index_cast %11 : i32 to index
            %87 = arith.index_cast %arg35 : i32 to index
            %88 = arith.index_cast %7 : i32 to index
            %89 = affine.apply #map8()[%arg38, %62, %83, %84, %85, %86, %87, %88]
            %90 = arith.index_cast %89 : index to i64
            %91 = arith.index_cast %arg37 : i64 to index
            %92 = arith.index_cast %31 : i64 to index
            %93 = arith.index_cast %arg36 : i32 to index
            %94 = arith.index_cast %11 : i32 to index
            %95 = arith.index_cast %arg35 : i32 to index
            %96 = arith.index_cast %7 : i32 to index
            %97 = affine.apply #map7()[%arg38, %62, %91, %92, %93, %94, %95, %96]
            %98 = arith.index_cast %97 : index to i1
            %99 = arith.select %98, %c2_i64, %c3_i64 : i64
            %100 = arith.index_cast %97 : index to i1
            %101 = arith.xori %100, %true : i1
            %102 = arith.extui %101 : i1 to i64
            %103 = arith.index_cast %arg38 : index to i64
            %104 = affine.apply #map9()[%arg38]
            %105 = arith.index_cast %104 : index to i64
            %106 = affine.apply #map10(%arg38)[%61]
            %107 = arith.index_cast %106 : index to i64
            %subview_3 = memref.subview %view[%arg38, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
            hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %102]
            %cast_4 = memref.cast %subview_3 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
            %cast_5 = memref.cast %82 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
            func.call @nd2nz_float(%cast_4, %cast_5) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
            hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %105 ffts_base_addr = %arg0
            %subview_6 = memref.subview %view_0[%arg38, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
            hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %99]
            %cast_7 = memref.cast %subview_6 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
            %cast_8 = memref.cast %72 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
            func.call @nd2nz_float(%cast_7, %cast_8) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
            hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %103 ffts_base_addr = %arg0
            %108 = arith.cmpi eq, %107, %c0_i64 : i64
            %cast_9 = memref.cast %82 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
            %cast_10 = memref.cast %72 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
            %cast_11 = memref.cast %53 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
            func.call @mma_tile_float_to_float(%cast_9, %cast_10, %108, %c16, %c16, %c16, %cast_11, %c0_i64, %c1_i64, %102, %99, %90, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
        }
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %54 = affine.apply #map11()[%46, %40, %38, %16]
        %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [16, 16], strides: [%16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        %55 = affine.max #map()[%46, %18]
        %56 = affine.min #map1()[%55, %46]
        %57 = affine.min #map2()[%56, %46]
        %58 = affine.max #map3()[%57]
        %59 = affine.apply #map12()[%44]
        %60 = affine.apply #map12()[%58]
        %subview = memref.subview %53[0, 0, 0, 0] [%60, %59, 16, 16] [1, 1, 1, 1] : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[256, 256, 16, 1]>, #hivm.address_space<cc>>
        %subview_1 = memref.subview %reinterpret_cast[0, 0] [%44, %58] [1, 1] : memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast = memref.cast %subview : memref<?x?x16x16xf32, strided<[256, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_2 = memref.cast %subview_1 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast, %cast_2, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      }
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func private @nd2nz_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_float_to_float(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_float_to_float_4d_to_2d(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vexp_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @_chunk_state_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32, %arg33: i32, %arg34: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<35xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c2_i32 = arith.constant 2 : i32
    %c2176_i64 = arith.constant 2176 : i64
    %c2112_i64 = arith.constant 2112 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant -1.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c15_i32 = arith.constant 15 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c16_i32 = arith.constant 16 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.muli %arg32, %arg33 : i32
    %1 = arith.muli %0, %arg34 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg34, %arg33 : i32
    %5 = arith.extsi %arg11 : i32 to i64
    %6 = arith.addi %arg9, %c15_i32 : i32
    %7 = arith.divsi %6, %c16_i32 : i32
    %8 = arith.extsi %arg17 : i32 to i64
    %9 = arith.extsi %arg10 : i32 to i64
    %10 = arith.extsi %arg18 : i32 to i64
    %11 = arith.extsi %arg14 : i32 to i64
    %12 = arith.extsi %arg15 : i32 to i64
    %13 = arith.extsi %arg24 : i32 to i64
    %14 = arith.extsi %arg25 : i32 to i64
    %15 = arith.extsi %arg27 : i32 to i64
    %16 = arith.extsi %arg28 : i32 to i64
    %17 = arith.subi %arg10, %c1_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.extsi %arg12 : i32 to i64
    %20 = arith.muli %arg15, %c16_i32 : i32
    %21 = arith.muli %arg18, %c16_i32 : i32
    %22 = arith.index_cast %arg15 : i32 to index
    %23 = arith.index_cast %arg18 : i32 to index
    %24 = arith.index_cast %arg8 : i32 to index
    %25 = hivm.hir.get_sub_block_idx -> i64
    %26 = arith.index_cast %25 : i64 to index
    %27 = arith.cmpi eq, %26, %c0 : index
    %28 = arith.index_cast %arg9 : i32 to index
    %29 = arith.index_cast %20 : i32 to index
    %30 = arith.index_cast %21 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg35 = %3 to %1 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %31 = arith.remsi %arg35, %arg34 : i32
      %32 = arith.divsi %arg35, %arg34 : i32
      %33 = arith.remsi %32, %arg33 : i32
      %34 = arith.divsi %arg35, %4 : i32
      %35 = arith.remsi %34, %arg32 : i32
      %36 = arith.extsi %33 : i32 to i64
      %37 = arith.divsi %36, %5 : i64
      %38 = arith.muli %37, %5 : i64
      %39 = arith.subi %36, %38 : i64
      %40 = arith.muli %39, %8 : i64
      %41 = arith.muli %37, %9 : i64
      %42 = arith.muli %41, %10 : i64
      %43 = arith.addi %40, %42 : i64
      %44 = arith.divsi %31, %arg13 : i32
      %45 = arith.muli %44, %arg19 : i32
      %46 = arith.extsi %45 : i32 to i64
      %47 = arith.addi %43, %46 : i64
      %48 = arith.muli %39, %11 : i64
      %49 = arith.muli %41, %12 : i64
      %50 = arith.addi %48, %49 : i64
      %51 = arith.muli %31, %arg16 : i32
      %52 = arith.extsi %51 : i32 to i64
      %53 = arith.addi %50, %52 : i64
      %54 = arith.muli %39, %13 : i64
      %55 = arith.muli %37, %14 : i64
      %56 = arith.addi %54, %55 : i64
      %57 = arith.muli %31, %arg26 : i32
      %58 = arith.extsi %57 : i32 to i64
      %59 = arith.addi %56, %58 : i64
      %60 = arith.muli %39, %15 : i64
      %61 = arith.muli %37, %16 : i64
      %62 = arith.addi %60, %61 : i64
      %63 = arith.muli %31, %arg29 : i32
      %64 = arith.extsi %63 : i32 to i64
      %65 = arith.addi %62, %64 : i64
      %66 = arith.muli %35, %c16_i32 : i32
      %67 = arith.index_cast %65 : i64 to index
      %68 = affine.apply #map13()[%67, %18]
      %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%68], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %69 = memref.load %reinterpret_cast[%c0] {disableDCache = 0 : i32, markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %70 = arith.subi %19, %41 : i64
      %71 = arith.minsi %9, %70 : i64
      %72 = arith.index_cast %66 : i32 to index
      %73 = arith.index_cast %71 : i64 to index
      %74 = arith.index_cast %53 : i64 to index
      %75 = arith.index_cast %47 : i64 to index
      %76 = arith.index_cast %59 : i64 to index
      %77 = affine.max #map()[%72, %24]
      %78 = affine.min #map1()[%77, %72]
      %79 = affine.min #map2()[%78, %72]
      %80 = affine.max #map3()[%79]
      %81 = arith.cmpi slt, %80, %c16 : index
      scf.for %arg36 = %c0_i32 to %7 step %c1_i32  : i32 {
        %82 = arith.muli %arg36, %c16_i32 : i32
        %83 = arith.index_cast %82 : i32 to index
        %84 = hivm.hir.get_block_idx -> i64
        %85 = arith.index_cast %84 : i64 to index
        %86 = affine.apply #map4(%85)
        %view = memref.view %arg2[%86][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
        %87 = hivm.hir.get_block_idx -> i64
        %88 = arith.index_cast %87 : i64 to index
        %89 = affine.apply #map5(%88)
        %view_1 = memref.view %arg2[%89][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
        %90 = affine.max #map()[%83, %28]
        %91 = affine.min #map1()[%90, %83]
        %92 = affine.min #map2()[%91, %83]
        %93 = affine.max #map3()[%92]
        %94 = arith.cmpi slt, %93, %c16 : index
        %95:4 = scf.for %arg37 = %c0_i64 to %71 step %c64_i64 iter_args(%arg38 = %72, %arg39 = %c0, %arg40 = %c0, %arg41 = %c0) -> (index, index, index, index)  : i64 {
          %96 = arith.index_cast %arg37 : i64 to index
          %97 = affine.min #map6(%73, %96)
          %98:4 = scf.for %arg42 = %c0 to %97 step %c1 iter_args(%arg43 = %arg38, %arg44 = %arg40, %arg45 = %arg41, %arg46 = %arg39) -> (index, index, index, index) {
            %99 = arith.index_cast %arg42 : index to i64
            %100 = affine.apply #map9()[%arg42]
            %101 = arith.index_cast %100 : index to i64
            %102 = affine.apply #map10(%arg42)[%96]
            %103 = arith.index_cast %102 : index to i64
            %104 = affine.apply #map13()[%74, %arg43]
            %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%104], sizes: [16, 16], strides: [1, %22] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>>
            %105 = affine.apply #map14()[%83, %75, %arg46]
            %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%105], sizes: [16, 16], strides: [%23, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
            %106 = affine.apply #map13()[%76, %arg44]
            %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%106], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %107 = affine.apply #map13()[%67, %arg45]
            %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%107], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %108 = arith.subi %71, %103 : i64
            %109 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16xf32, #hivm.address_space<ub>>
            %110 = arith.index_cast %108 : i64 to index
            %111 = affine.max #map3()[%110]
            %112 = affine.min #map15()[%111]
            %113 = affine.max #map3()[%112]
            %114 = arith.cmpi slt, %113, %c16 : index
            %115 = arith.ori %81, %114 : i1
            %subview = memref.subview %reinterpret_cast_2[0, 0] [%80, %113] [1, 1] : memref<16x16xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>>
            %subview_6 = memref.subview %109[0, 0] [%80, %113] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
            scf.if %115 {
              hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
              %collapse_shape_40 = memref.collapse_shape %109 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
              hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
              %cast_41 = memref.cast %collapse_shape_40 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
              func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_41) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
              hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            } {hivm.unlikely_condition}
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
            %cast = memref.cast %subview : memref<?x?xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
            %cast_7 = memref.cast %subview_6 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            func.call @load_gm_to_ubuf_2d_float(%cast, %cast_7, %c2_i32, %cst_0, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
            %subview_8 = memref.subview %view[%arg42, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape = memref.collapse_shape %subview_8 [[0, 1], [2]] : memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %101
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
            scf.if %27 {
              %collapse_shape_40 = memref.collapse_shape %109 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
              %collapse_shape_41 = memref.collapse_shape %collapse_shape [[0, 1]] : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
              %cast_42 = memref.cast %collapse_shape_40 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
              %cast_43 = memref.cast %collapse_shape_41 : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
              func.call @store_ubuf_to_gm_1d_float(%cast_42, %cast_43, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
            } {limit_sub_block_id0}
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8 ffts_base_addr = %arg0
            %116 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x16xf32, #hivm.address_space<ub>>
            %117 = arith.ori %114, %94 : i1
            %subview_9 = memref.subview %reinterpret_cast_3[0, 0] [%113, %93] [1, 1] : memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
            %subview_10 = memref.subview %116[0, 0] [%113, %93] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
            scf.if %117 {
              hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
              %collapse_shape_40 = memref.collapse_shape %116 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
              hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
              %cast_41 = memref.cast %collapse_shape_40 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
              func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_41) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
              hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            } {hivm.unlikely_condition}
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
            %cast_11 = memref.cast %subview_9 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
            %cast_12 = memref.cast %subview_10 : memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            func.call @load_gm_to_ubuf_2d_float(%cast_11, %cast_12, %c2_i32, %cst_0, %c0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
            %118 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
            %119 = arith.cmpi slt, %112, %c16 : index
            %subview_13 = memref.subview %reinterpret_cast_5[0] [%112] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_14 = memref.subview %118[0] [%112] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
            scf.if %119 {
              hivm.hir.pipe_barrier[<PIPE_V>]
              %cast_40 = memref.cast %118 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
              func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_40) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
              hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            } {hivm.unlikely_condition}
            %cast_15 = memref.cast %subview_13 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %cast_16 = memref.cast %subview_14 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @load_gm_to_ubuf_1d_float(%cast_15, %cast_16, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
            %120 = hivm.hir.pointer_cast(%c2112_i64) : memref<16xf32, #hivm.address_space<ub>>
            %subview_17 = memref.subview %reinterpret_cast_4[0] [%112] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_18 = memref.subview %120[0] [%112] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
            scf.if %119 {
              %cast_40 = memref.cast %120 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
              func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_40) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
              hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            } {hivm.unlikely_condition}
            %cast_19 = memref.cast %subview_17 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            %cast_20 = memref.cast %subview_18 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @load_gm_to_ubuf_1d_float(%cast_19, %cast_20, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
            hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            %121 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
            %122 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_21 = memref.cast %118 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_22 = memref.cast %121 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_23 = memref.cast %122 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmuls_vs_1d_float(%cast_21, %cst, %cast_22, %cast_23) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %123 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %124 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_24 = memref.cast %121 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_25 = memref.cast %123 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_26 = memref.cast %124 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vadds_vs_1d_float(%cast_24, %69, %cast_25, %cast_26) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %125 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %126 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_27 = memref.cast %123 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_28 = memref.cast %125 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_29 = memref.cast %126 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vexp_1d_float(%cast_27, %cast_28, %cast_29) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %127 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
            hivm.hir.pipe_barrier[<PIPE_V>]
            %128 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
            %cast_30 = memref.cast %125 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_31 = memref.cast %120 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_32 = memref.cast %127 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_33 = memref.cast %128 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmul_1d_float(%cast_30, %cast_31, %cast_32, %cast_33) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            %expand_shape = memref.expand_shape %127 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
            %129 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x16xf32, #hivm.address_space<ub>>
            %130 = hivm.hir.pointer_cast(%c2176_i64) : memref<128xf32, #hivm.address_space<ub>>
            hivm.hir.pipe_barrier[<PIPE_V>]
            %cast_34 = memref.cast %116 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_35 = memref.cast %expand_shape : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_36 = memref.cast %129 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
            %cast_37 = memref.cast %130 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @vmul_2d_float(%cast_34, %cast_35, %cast_36, %cast_37) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            %subview_38 = memref.subview %view_1[%arg42, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>>
            %collapse_shape_39 = memref.collapse_shape %subview_38 [[0, 1], [2]] : memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %99
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            scf.if %27 {
              %collapse_shape_40 = memref.collapse_shape %129 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
              %collapse_shape_41 = memref.collapse_shape %collapse_shape_39 [[0, 1]] : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
              %cast_42 = memref.cast %collapse_shape_40 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
              %cast_43 = memref.cast %collapse_shape_41 : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
              func.call @store_ubuf_to_gm_1d_float(%cast_42, %cast_43, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
            } {limit_sub_block_id0}
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8 ffts_base_addr = %arg0
            %131 = affine.apply #map13()[%arg43, %29]
            %132 = affine.apply #map13()[%arg46, %30]
            %133 = affine.apply #map16()[%arg44]
            %134 = affine.apply #map16()[%arg45]
            scf.yield %131, %133, %134, %132 : index, index, index, index
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
          scf.yield %98#0, %98#3, %98#1, %98#2 : index, index, index, index
        }
      }
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
