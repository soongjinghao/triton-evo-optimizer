#map = affine_map<(d0) -> (d0 * 8192)>
#map1 = affine_map<(d0) -> (d0 * 8192 + 4096)>
#map2 = affine_map<(d0, d1) -> ((d0 - d1) ceildiv 16, 4)>
#map3 = affine_map<()[s0, s1, s2, s3, s4, s5] -> ((s0 + (s2 floordiv 64) * s1 + ((s4 - s5) floordiv 24) * (s1 * ((s3 + 63) floordiv 64))) mod 2)>
#map4 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 + (s2 floordiv 64) * s1 + ((s4 - s5) floordiv 24) * (s1 * ((s3 + 63) floordiv 64)))>
#map5 = affine_map<()[s0] -> (s0 + 4)>
#map6 = affine_map<(d0)[s0] -> (d0 * 16 + s0)>
#map7 = affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>
#map8 = affine_map<()[s0, s1] -> (s1, s0)>
#map9 = affine_map<()[s0, s1] -> (s1 + 16, s0)>
#map10 = affine_map<()[s0, s1] -> (16, s0 - s1)>
#map11 = affine_map<()[s0] -> (0, s0)>
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
  func.func @_chunk_state_fwd_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32, %arg38: i32, %arg39: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<40xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %true = arith.constant true
    %c64_i64 = arith.constant 64 : i64
    %c1 = arith.constant 1 : index
    %c16_i32 = arith.constant 16 : i32
    %c15_i32 = arith.constant 15 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
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
    %4 = arith.muli %arg37, %arg38 : i32
    %5 = arith.muli %4, %arg39 : i32
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
    %8 = arith.muli %arg39, %arg38 : i32
    %9 = arith.extsi %arg12 : i32 to i64
    %10 = arith.addi %arg10, %c15_i32 : i32
    %11 = arith.divsi %10, %c16_i32 : i32
    %12 = arith.extsi %arg11 : i32 to i64
    %13 = arith.extsi %arg13 : i32 to i64
    %14 = arith.extsi %arg21 : i32 to i64
    %15 = arith.extsi %arg22 : i32 to i64
    %16 = arith.index_cast %arg24 : i32 to index
    %17 = arith.index_cast %arg9 : i32 to index
    %18 = arith.index_cast %arg10 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    scf.for %arg40 = %7 to %5 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %19 = arith.remsi %arg40, %arg39 : i32
      %20 = arith.divsi %arg40, %arg39 : i32
      %21 = arith.remsi %20, %arg38 : i32
      %22 = arith.divsi %arg40, %8 : i32
      %23 = arith.remsi %22, %arg37 : i32
      %24 = arith.extsi %21 : i32 to i64
      %25 = arith.divsi %24, %9 : i64
      %26 = arith.muli %25, %9 : i64
      %27 = arith.subi %24, %26 : i64
      %28 = arith.divsi %23, %11 : i32
      %29 = arith.remsi %23, %11 : i32
      %30 = arith.muli %25, %12 : i64
      %31 = arith.muli %28, %c16_i32 : i32
      %32 = arith.muli %29, %c16_i32 : i32
      %33 = arith.subi %13, %30 : i64
      %34 = arith.minsi %12, %33 : i64
      %35 = arith.index_cast %31 : i32 to index
      %36 = arith.index_cast %32 : i32 to index
      %37 = hivm.hir.get_block_idx -> i64
      %38 = arith.index_cast %37 : i64 to index
      %39 = affine.apply #map(%38)
      %view = memref.view %arg2[%39][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
      %40 = hivm.hir.get_block_idx -> i64
      %41 = arith.index_cast %40 : i64 to index
      %42 = affine.apply #map1(%41)
      %view_0 = memref.view %arg2[%42][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
      %43 = arith.index_cast %34 : i64 to index
      %44 = hivm.hir.pointer_cast(%c0_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
      scf.for %arg41 = %c0_i64 to %34 step %c64_i64  : i64 {
        %63 = arith.index_cast %arg41 : i64 to index
        %64 = affine.min #map2(%43, %63)
        scf.for %arg42 = %c0 to %64 step %c1 {
          %65 = arith.index_cast %arg41 : i64 to index
          %66 = arith.index_cast %34 : i64 to index
          %67 = arith.index_cast %arg40 : i32 to index
          %68 = arith.index_cast %7 : i32 to index
          %69 = affine.apply #map3()[%arg42, %64, %65, %66, %67, %68]
          %70 = arith.index_cast %69 : index to i64
          %71 = arith.cmpi eq, %70, %c1_i64 : i64
          %72 = arith.select %71, %1, %0 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
          %73 = arith.index_cast %arg41 : i64 to index
          %74 = arith.index_cast %34 : i64 to index
          %75 = arith.index_cast %arg40 : i32 to index
          %76 = arith.index_cast %7 : i32 to index
          %77 = affine.apply #map3()[%arg42, %64, %73, %74, %75, %76]
          %78 = arith.index_cast %77 : index to i64
          %79 = arith.cmpi eq, %78, %c1_i64 : i64
          %80 = arith.select %79, %3, %2 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
          %81 = arith.index_cast %arg41 : i64 to index
          %82 = arith.index_cast %34 : i64 to index
          %83 = arith.index_cast %arg40 : i32 to index
          %84 = arith.index_cast %7 : i32 to index
          %85 = affine.apply #map4()[%arg42, %64, %81, %82, %83, %84]
          %86 = arith.index_cast %85 : index to i64
          %87 = arith.index_cast %arg41 : i64 to index
          %88 = arith.index_cast %34 : i64 to index
          %89 = arith.index_cast %arg40 : i32 to index
          %90 = arith.index_cast %7 : i32 to index
          %91 = affine.apply #map3()[%arg42, %64, %87, %88, %89, %90]
          %92 = arith.index_cast %91 : index to i1
          %93 = arith.select %92, %c2_i64, %c3_i64 : i64
          %94 = arith.index_cast %91 : index to i1
          %95 = arith.xori %94, %true : i1
          %96 = arith.extui %95 : i1 to i64
          %97 = arith.index_cast %arg42 : index to i64
          %98 = affine.apply #map5()[%arg42]
          %99 = arith.index_cast %98 : index to i64
          %100 = affine.apply #map6(%arg42)[%63]
          %101 = arith.index_cast %100 : index to i64
          %subview_3 = memref.subview %view_0[%arg42, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %96]
          %cast_4 = memref.cast %subview_3 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_5 = memref.cast %80 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @nd2nz_float(%cast_4, %cast_5) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %99 ffts_base_addr = %arg0
          %subview_6 = memref.subview %view[%arg42, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 8
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %93]
          %cast_7 = memref.cast %subview_6 : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
          %cast_8 = memref.cast %72 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @nd2nz_float(%cast_7, %cast_8) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %97 ffts_base_addr = %arg0
          %102 = arith.cmpi eq, %101, %c0_i64 : i64
          %cast_9 = memref.cast %80 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_10 = memref.cast %72 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
          %cast_11 = memref.cast %44 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
          func.call @mma_tile_float_to_float(%cast_9, %cast_10, %102, %c16, %c16, %c16, %cast_11, %c0_i64, %c1_i64, %96, %93, %86, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
      }
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %45 = arith.muli %27, %14 : i64
      %46 = arith.muli %25, %15 : i64
      %47 = arith.addi %45, %46 : i64
      %48 = arith.muli %19, %arg23 : i32
      %49 = arith.extsi %48 : i32 to i64
      %50 = arith.addi %47, %49 : i64
      %51 = arith.index_cast %50 : i64 to index
      %52 = affine.apply #map7()[%36, %51, %35, %16]
      %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [16, 16], strides: [%16, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %53 = affine.max #map8()[%35, %17]
      %54 = affine.min #map9()[%53, %35]
      %55 = affine.max #map8()[%36, %18]
      %56 = affine.min #map9()[%55, %36]
      %57 = affine.min #map10()[%54, %35]
      %58 = affine.max #map11()[%57]
      %59 = affine.min #map10()[%56, %36]
      %60 = affine.max #map11()[%59]
      %61 = affine.apply #map12()[%58]
      %62 = affine.apply #map12()[%60]
      %subview = memref.subview %44[0, 0, 0, 0] [%62, %61, 16, 16] [1, 1, 1, 1] : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x16x16xf32, strided<[256, 256, 16, 1]>, #hivm.address_space<cc>>
      %subview_1 = memref.subview %reinterpret_cast[0, 0] [%58, %60] [1, 1] : memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %cast = memref.cast %subview : memref<?x?x16x16xf32, strided<[256, 256, 16, 1]>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
      %cast_2 = memref.cast %subview_1 : memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast, %cast_2, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
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
  func.func @_chunk_state_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32 {tt.divisibility = 16 : i32}, %arg36: i32 {tt.divisibility = 16 : i32}, %arg37: i32, %arg38: i32, %arg39: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : vector<40xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c2176_i64 = arith.constant 2176 : i64
    %c2112_i64 = arith.constant 2112 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c24_i32 = arith.constant 24 : i32
    %cst = arith.constant -1.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c15_i32 = arith.constant 15 : i32
    %c0_i64 = arith.constant 0 : i64
    %c16_i32 = arith.constant 16 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    hivm.hir.set_ffts_base_addr %arg0
    %0 = arith.muli %arg37, %arg38 : i32
    %1 = arith.muli %0, %arg39 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg39, %arg38 : i32
    %5 = arith.extsi %arg12 : i32 to i64
    %6 = arith.addi %arg10, %c15_i32 : i32
    %7 = arith.divsi %6, %c16_i32 : i32
    %8 = arith.extsi %arg18 : i32 to i64
    %9 = arith.extsi %arg11 : i32 to i64
    %10 = arith.extsi %arg19 : i32 to i64
    %11 = arith.extsi %arg15 : i32 to i64
    %12 = arith.extsi %arg16 : i32 to i64
    %13 = arith.extsi %arg25 : i32 to i64
    %14 = arith.extsi %arg26 : i32 to i64
    %15 = arith.extsi %arg28 : i32 to i64
    %16 = arith.extsi %arg29 : i32 to i64
    %17 = arith.extsi %arg33 : i32 to i64
    %18 = arith.extsi %arg34 : i32 to i64
    %19 = arith.extsi %arg13 : i32 to i64
    %20 = arith.muli %arg16, %c16_i32 : i32
    %21 = arith.muli %arg19, %c16_i32 : i32
    %22 = arith.index_cast %arg16 : i32 to index
    %23 = arith.index_cast %arg19 : i32 to index
    %24 = arith.index_cast %arg9 : i32 to index
    %25 = hivm.hir.get_sub_block_idx -> i64
    %26 = arith.index_cast %25 : i64 to index
    %27 = arith.cmpi eq, %26, %c0 : index
    %28 = arith.index_cast %arg10 : i32 to index
    %29 = arith.index_cast %20 : i32 to index
    %30 = arith.index_cast %21 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg40 = %3 to %1 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %31 = arith.remsi %arg40, %arg39 : i32
      %32 = arith.divsi %arg40, %arg39 : i32
      %33 = arith.remsi %32, %arg38 : i32
      %34 = arith.divsi %arg40, %4 : i32
      %35 = arith.remsi %34, %arg37 : i32
      %36 = arith.extsi %33 : i32 to i64
      %37 = arith.divsi %36, %5 : i64
      %38 = arith.muli %37, %5 : i64
      %39 = arith.subi %36, %38 : i64
      %40 = arith.divsi %35, %7 : i32
      %41 = arith.remsi %35, %7 : i32
      %42 = arith.muli %39, %8 : i64
      %43 = arith.muli %37, %9 : i64
      %44 = arith.muli %43, %10 : i64
      %45 = arith.addi %42, %44 : i64
      %46 = arith.divsi %31, %arg14 : i32
      %47 = arith.muli %46, %arg20 : i32
      %48 = arith.extsi %47 : i32 to i64
      %49 = arith.addi %45, %48 : i64
      %50 = arith.muli %39, %11 : i64
      %51 = arith.muli %43, %12 : i64
      %52 = arith.addi %50, %51 : i64
      %53 = arith.muli %31, %arg17 : i32
      %54 = arith.extsi %53 : i32 to i64
      %55 = arith.addi %52, %54 : i64
      %56 = arith.muli %39, %13 : i64
      %57 = arith.muli %37, %14 : i64
      %58 = arith.addi %56, %57 : i64
      %59 = arith.muli %31, %arg27 : i32
      %60 = arith.extsi %59 : i32 to i64
      %61 = arith.addi %58, %60 : i64
      %62 = arith.muli %39, %15 : i64
      %63 = arith.muli %37, %16 : i64
      %64 = arith.addi %62, %63 : i64
      %65 = arith.muli %31, %arg30 : i32
      %66 = arith.extsi %65 : i32 to i64
      %67 = arith.addi %64, %66 : i64
      %68 = arith.muli %40, %c16_i32 : i32
      %69 = arith.muli %41, %c16_i32 : i32
      %70 = arith.muli %39, %17 : i64
      %71 = arith.muli %37, %18 : i64
      %72 = arith.addi %70, %71 : i64
      %73 = arith.index_cast %72 : i64 to index
      %74 = arith.index_cast %31 : i32 to index
      %75 = affine.apply #map13()[%73, %74]
      %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%75], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %76 = memref.load %reinterpret_cast[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %77 = arith.subi %19, %43 : i64
      %78 = arith.minsi %9, %77 : i64
      %79 = arith.index_cast %68 : i32 to index
      %80 = arith.index_cast %69 : i32 to index
      %81 = hivm.hir.get_block_idx -> i64
      %82 = arith.index_cast %81 : i64 to index
      %83 = affine.apply #map(%82)
      %view = memref.view %arg2[%83][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
      %84 = hivm.hir.get_block_idx -> i64
      %85 = arith.index_cast %84 : i64 to index
      %86 = affine.apply #map1(%85)
      %view_1 = memref.view %arg2[%86][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x16x16xf32, #hivm.address_space<gm>>
      %87 = arith.index_cast %78 : i64 to index
      %88 = arith.index_cast %55 : i64 to index
      %89 = arith.index_cast %49 : i64 to index
      %90 = arith.index_cast %61 : i64 to index
      %91 = arith.index_cast %67 : i64 to index
      %92 = affine.max #map8()[%79, %24]
      %93 = affine.min #map9()[%92, %79]
      %94 = affine.min #map10()[%93, %79]
      %95 = affine.max #map11()[%94]
      %96 = arith.cmpi slt, %95, %c16 : index
      %97 = affine.max #map8()[%80, %28]
      %98 = affine.min #map9()[%97, %80]
      %99 = affine.min #map10()[%98, %80]
      %100 = affine.max #map11()[%99]
      %101 = arith.cmpi slt, %100, %c16 : index
      %102:4 = scf.for %arg41 = %c0_i64 to %78 step %c64_i64 iter_args(%arg42 = %79, %arg43 = %c0, %arg44 = %c0, %arg45 = %c0) -> (index, index, index, index)  : i64 {
        %103 = arith.index_cast %arg41 : i64 to index
        %104 = affine.min #map2(%87, %103)
        %105:4 = scf.for %arg46 = %c0 to %104 step %c1 iter_args(%arg47 = %arg45, %arg48 = %arg44, %arg49 = %arg43, %arg50 = %arg42) -> (index, index, index, index) {
          %106 = arith.index_cast %arg46 : index to i64
          %107 = affine.apply #map5()[%arg46]
          %108 = arith.index_cast %107 : index to i64
          %109 = affine.apply #map6(%arg46)[%103]
          %110 = arith.index_cast %109 : index to i64
          %111 = affine.apply #map13()[%88, %arg50]
          %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%111], sizes: [16, 16], strides: [1, %22] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>>
          %112 = affine.apply #map14()[%80, %89, %arg49]
          %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%112], sizes: [16, 16], strides: [%23, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %113 = affine.apply #map13()[%90, %arg48]
          %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%113], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %114 = affine.apply #map13()[%91, %arg47]
          %reinterpret_cast_5 = memref.reinterpret_cast %arg7 to offset: [%114], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %115 = arith.subi %78, %110 : i64
          %116 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16xf32, #hivm.address_space<ub>>
          %117 = arith.index_cast %115 : i64 to index
          %118 = affine.max #map11()[%117]
          %119 = affine.min #map15()[%118]
          %120 = affine.max #map11()[%119]
          %121 = arith.cmpi slt, %120, %c16 : index
          %122 = arith.ori %96, %121 : i1
          %subview = memref.subview %reinterpret_cast_2[0, 0] [%95, %120] [1, 1] : memref<16x16xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[1, ?], offset: ?>, #hivm.address_space<gm>>
          %subview_6 = memref.subview %116[0, 0] [%95, %120] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          scf.if %122 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape_40 = memref.collapse_shape %116 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
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
          %subview_8 = memref.subview %view_1[%arg46, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape = memref.collapse_shape %subview_8 [[0, 1], [2]] : memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %108
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
          scf.if %27 {
            %collapse_shape_40 = memref.collapse_shape %116 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
            %collapse_shape_41 = memref.collapse_shape %collapse_shape [[0, 1]] : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %cast_42 = memref.cast %collapse_shape_40 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_43 = memref.cast %collapse_shape_41 : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            func.call @store_ubuf_to_gm_1d_float(%cast_42, %cast_43, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          } {limit_sub_block_id0}
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8 ffts_base_addr = %arg0
          %123 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x16xf32, #hivm.address_space<ub>>
          %124 = arith.ori %121, %101 : i1
          %subview_9 = memref.subview %reinterpret_cast_3[0, 0] [%120, %100] [1, 1] : memref<16x16xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_10 = memref.subview %123[0, 0] [%120, %100] [1, 1] : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[16, 1]>, #hivm.address_space<ub>>
          scf.if %124 {
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
            %collapse_shape_40 = memref.collapse_shape %123 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
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
          %125 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
          %126 = arith.cmpi slt, %119, %c16 : index
          %subview_13 = memref.subview %reinterpret_cast_5[0] [%119] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_14 = memref.subview %125[0] [%119] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %126 {
            hivm.hir.pipe_barrier[<PIPE_V>]
            %cast_40 = memref.cast %125 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_40) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %cast_15 = memref.cast %subview_13 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_16 = memref.cast %subview_14 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_15, %cast_16, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
          %127 = hivm.hir.pointer_cast(%c2112_i64) : memref<16xf32, #hivm.address_space<ub>>
          %subview_17 = memref.subview %reinterpret_cast_4[0] [%119] [1] : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %subview_18 = memref.subview %127[0] [%119] [1] : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          scf.if %126 {
            %cast_40 = memref.cast %127 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            func.call @broadcast_scalar_float_to_1d(%cst_0, %cast_40) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          } {hivm.unlikely_condition}
          %cast_19 = memref.cast %subview_17 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          %cast_20 = memref.cast %subview_18 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @load_gm_to_ubuf_1d_float(%cast_19, %cast_20, %c2_i32, %cst_0, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          %128 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
          %129 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_21 = memref.cast %125 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_22 = memref.cast %128 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_23 = memref.cast %129 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmuls_vs_1d_float(%cast_21, %cst, %cast_22, %cast_23) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %130 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %131 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_24 = memref.cast %128 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_25 = memref.cast %130 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_26 = memref.cast %131 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vadds_vs_1d_float(%cast_24, %76, %cast_25, %cast_26) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %132 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %133 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_27 = memref.cast %130 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_28 = memref.cast %132 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_29 = memref.cast %133 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vexp_1d_float(%cast_27, %cast_28, %cast_29) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %134 = hivm.hir.pointer_cast(%c2048_i64) : memref<16xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %135 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
          %cast_30 = memref.cast %132 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_31 = memref.cast %127 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_32 = memref.cast %134 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_33 = memref.cast %135 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_1d_float(%cast_30, %cast_31, %cast_32, %cast_33) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %expand_shape = memref.expand_shape %134 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
          %136 = hivm.hir.pointer_cast(%c1024_i64) : memref<16x16xf32, #hivm.address_space<ub>>
          %137 = hivm.hir.pointer_cast(%c2176_i64) : memref<128xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_34 = memref.cast %123 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_35 = memref.cast %expand_shape : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_36 = memref.cast %136 : memref<16x16xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_37 = memref.cast %137 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vmul_2d_float(%cast_34, %cast_35, %cast_36, %cast_37) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %subview_38 = memref.subview %view[%arg46, 0, 0] [1, 16, 16] [1, 1, 1] : memref<4x16x16xf32, #hivm.address_space<gm>> to memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_39 = memref.collapse_shape %subview_38 [[0, 1], [2]] : memref<1x16x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<gm>> into memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %106
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          scf.if %27 {
            %collapse_shape_40 = memref.collapse_shape %136 [[0, 1]] : memref<16x16xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
            %collapse_shape_41 = memref.collapse_shape %collapse_shape_39 [[0, 1]] : memref<16x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %cast_42 = memref.cast %collapse_shape_40 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
            %cast_43 = memref.cast %collapse_shape_41 : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
            func.call @store_ubuf_to_gm_1d_float(%cast_42, %cast_43, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          } {limit_sub_block_id0}
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 8 ffts_base_addr = %arg0
          %138 = affine.apply #map13()[%arg50, %29]
          %139 = affine.apply #map13()[%arg49, %30]
          %140 = affine.apply #map16()[%arg48]
          %141 = affine.apply #map16()[%arg47]
          scf.yield %141, %140, %139, %138 : index, index, index, index
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
        scf.yield %105#3, %105#2, %105#1, %105#0 : index, index, index, index
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
