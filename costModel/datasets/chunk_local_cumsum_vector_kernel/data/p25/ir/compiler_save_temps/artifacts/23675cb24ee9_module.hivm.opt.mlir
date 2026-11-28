#map = affine_map<()[s0] -> (s0 * 16384)>
#map1 = affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 24, 4)>
#map2 = affine_map<()[s0, s1, s2, s3] -> ((s0 + ((s2 - s3) floordiv 96) * s1) mod 2)>
#map3 = affine_map<(d0)[s0] -> (d0 * 24 + s0)>
#map4 = affine_map<()[s0] -> (s0 mod 2)>
#map5 = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 + s1 * 2 + s2 * 8 + ((s4 - s5) floordiv 96) * (s3 * 8))>
#map6 = affine_map<()[s0, s1] -> (s0 + s1 * 1024)>
#map7 = affine_map<()[s0, s1] -> (0, s0 - s1)>
#map8 = affine_map<()[s0] -> (32, s0)>
#map9 = affine_map<()[s0, s1] -> (32, s1, s0)>
#map10 = affine_map<()[s0, s1] -> (s0 - s1)>
#map11 = affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>
#map12 = affine_map<()[s0] -> (s0 floordiv 16)>
#map13 = affine_map<()[s0] -> (s0 mod 16)>
#map14 = affine_map<()[s0] -> (s0 * 8)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @chunk_local_cumsum_vector_kernel_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c16384 = arith.constant 16384 : index
    return %c16384 : index
  }
  func.func @chunk_local_cumsum_vector_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @chunk_local_cumsum_vector_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c96_i32 = arith.constant 96 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c0_i64 = arith.constant 0 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    %1 = hivm.hir.pointer_cast(%c8192_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    %2 = hivm.hir.pointer_cast(%c4096_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c12288_i64) : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_ffts_base_addr %arg0
    %4 = arith.muli %arg6, %arg7 : i32
    %5 = arith.muli %4, %arg8 : i32
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.index_cast %8 : i64 to index
    %10 = affine.apply #map()[%9]
    %view = memref.view %arg2[%10][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x32x32xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3 ffts_base_addr = %arg0
    %11 = arith.index_cast %5 : i32 to index
    %12 = arith.muli %arg8, %arg7 : i32
    %13 = arith.addi %arg5, %c31_i32 : i32
    %14 = arith.divsi %13, %c32_i32 : i32
    %15 = arith.addi %14, %c1_i32 : i32
    %16 = arith.divsi %15, %c2_i32 : i32
    %17 = arith.muli %16, %c16_i32 : i32
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    scf.for %arg9 = %7 to %5 step %c96_i32  : i32 {
      %18 = arith.index_cast %arg9 : i32 to index
      %19 = affine.min #map1(%18)[%11]
      scf.for %arg10 = %c0 to %19 step %c1 {
        %20 = arith.index_cast %arg9 : i32 to index
        %21 = arith.index_cast %7 : i32 to index
        %22 = affine.apply #map2()[%arg10, %19, %20, %21]
        %23 = arith.index_cast %22 : index to i64
        %24 = arith.cmpi eq, %23, %c1_i64 : i64
        %25 = arith.select %24, %1, %0 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
        %26 = arith.index_cast %arg9 : i32 to index
        %27 = arith.index_cast %7 : i32 to index
        %28 = affine.apply #map2()[%arg10, %19, %26, %27]
        %29 = arith.index_cast %28 : index to i1
        %30 = arith.xori %29, %true : i1
        %31 = arith.extui %30 : i1 to i64
        %32 = arith.index_cast %arg10 : index to i64
        %33 = affine.apply #map3(%arg10)[%18]
        %34 = arith.index_cast %33 : index to i32
        %35 = arith.divsi %34, %12 : i32
        %36 = arith.remsi %35, %arg6 : i32
        %subview = memref.subview %view[%arg10, 0, 0] [1, 32, 32] [1, 1, 1] : memref<4x32x32xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 4
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %31]
        %cast = memref.cast %subview : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_0 = memref.cast %25 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast, %cast_0) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = %32 ffts_base_addr = %arg0
        %37 = arith.cmpi slt, %36, %17 : i32
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        scf.if %37 {
          %38 = arith.remsi %36, %16 : i32
          %39 = arith.divsi %36, %16 : i32
          %40 = arith.muli %39, %c4_i32 : i32
          %41 = arith.addi %40, %c4_i32 : i32
          %42 = arith.minsi %41, %c64_i32 : i32
          %43 = arith.muli %38, %c2_i32 : i32
          %44 = arith.addi %43, %c2_i32 : i32
          %45 = arith.minsi %44, %14 : i32
          scf.for %arg11 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
            %46 = arith.addi %40, %arg11 : i32
            %47 = arith.cmpi slt, %46, %42 : i32
            scf.if %47 {
              %48 = arith.divsi %46, %c16_i32 : i32
              %49 = arith.remsi %46, %c16_i32 : i32
              hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
              hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
              hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
              scf.for %arg12 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
                %50 = arith.index_cast %arg12 : i32 to index
                %51 = affine.apply #map4()[%50]
                %52 = arith.index_cast %51 : index to i64
                %53 = arith.cmpi eq, %52, %c1_i64 : i64
                %54 = arith.select %53, %3, %2 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
                %55 = arith.index_cast %arg12 : i32 to index
                %56 = arith.index_cast %arg11 : i32 to index
                %57 = arith.index_cast %arg9 : i32 to index
                %58 = arith.index_cast %7 : i32 to index
                %59 = affine.apply #map5()[%55, %56, %arg10, %19, %57, %58]
                %60 = arith.index_cast %59 : index to i64
                %61 = arith.index_cast %arg12 : i32 to index
                %62 = affine.apply #map4()[%61]
                %63 = arith.index_cast %62 : index to i1
                %64 = arith.select %63, %c2_i64, %c3_i64 : i64
                %65 = arith.addi %43, %arg12 : i32
                %66 = arith.cmpi slt, %65, %45 : i32
                scf.if %66 {
                  %67 = arith.muli %48, %arg5 : i32
                  %68 = arith.muli %67, %c16_i32 : i32
                  %69 = arith.addi %68, %49 : i32
                  %70 = arith.muli %69, %c64_i32 : i32
                  %71 = arith.index_cast %70 : i32 to index
                  %72 = arith.muli %65, %c32_i32 : i32
                  %73 = arith.maxsi %72, %c0_i32 : i32
                  %74 = arith.index_cast %73 : i32 to index
                  %75 = affine.apply #map6()[%71, %74]
                  %76 = arith.index_cast %arg5 : i32 to index
                  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%75], sizes: [32, 32], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%75], sizes: [32, 32], strides: [1024, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<32x32xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
                  %77 = arith.cmpi slt, %72, %arg5 : i32
                  scf.if %77 {
                    %78 = affine.max #map7()[%76, %74]
                    %79 = affine.min #map8()[%78]
                    %80 = arith.subi %c0_i32, %72 : i32
                    %81 = arith.maxsi %80, %c0_i32 : i32
                    %82 = arith.index_cast %81 : i32 to index
                    %83 = affine.min #map9()[%82, %78]
                    %84 = affine.apply #map10()[%79, %83]
                    %85 = arith.cmpi slt, %84, %c32 : index
                    %subview_2 = memref.subview %reinterpret_cast[0, 0] [%84, 32] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
                    %86 = affine.apply #map11()[%79, %83]
                    %87 = affine.apply #map12()[%83]
                    %88 = affine.apply #map13()[%83]
                    %subview_3 = memref.subview %54[0, %87, %88, 0] [4, %86, 16, 8] [1, 1, 1, 1] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<4x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
                    scf.if %85 {
                      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
                      %collapse_shape = memref.collapse_shape %54 [[0, 1, 2, 3]] : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> into memref<1024xf32, #hivm.address_space<cbuf>>
                      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
                      %cast_13 = memref.cast %collapse_shape : memref<1024xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
                      func.call @set_l1_2d_float(%cst, %cast_13) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
                      hivm.hir.pipe_barrier[<PIPE_MTE2>]
                    } {hivm.unlikely_condition}
                    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %64]
                    %cast_4 = memref.cast %subview_2 : memref<?x32xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
                    %cast_5 = memref.cast %subview_3 : memref<4x?x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
                    func.call @nd2nz_float(%cast_4, %cast_5) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
                    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
                    %89 = hivm.hir.pointer_cast(%c0_i64) : memref<2x2x16x16xf32, #hivm.address_space<cc>>
                    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
                    %cast_6 = memref.cast %25 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
                    %cast_7 = memref.cast %54 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
                    %cast_8 = memref.cast %89 : memref<2x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
                    func.call @mma_tile_float_to_float(%cast_6, %cast_7, %true, %c32, %c32, %c32, %cast_8, %c-1_i64, %c0_i64, %c-1_i64, %64, %60, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
                    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
                    %subview_9 = memref.subview %89[0, %87, %88, 0] [2, %86, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf32, #hivm.address_space<cc>> to memref<2x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>>
                    %subview_10 = memref.subview %reinterpret_cast_1[0, 0] [%84, 32] [1, 1] : memref<32x32xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x32xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>>
                    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
                    %cast_11 = memref.cast %subview_9 : memref<2x?x16x16xf32, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
                    %cast_12 = memref.cast %subview_10 : memref<?x32xf32, strided<[1024, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
                    func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_11, %cast_12, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
                    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
                  }
                }
              } {fixpipe_for_mmad_result_already_inserted = true}
              hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
              hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
              hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
            }
          }
        } {pipeline.cubeonly}
        hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %31]
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
    }
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func private @nd2nz_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @set_l1_2d_float(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_float_to_float(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_float_to_float_4d_to_2d(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @arange_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_int32_t_to_float_1d_with_mode(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_first_axis_align_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_ge_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_ss_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @chunk_local_cumsum_vector_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c3616_i64 = arith.constant 3616 : i64
    %c2592_i64 = arith.constant 2592 : i64
    %c2336_i64 = arith.constant 2336 : i64
    %c1312_i64 = arith.constant 1312 : i64
    %c1280_i64 = arith.constant 1280 : i64
    %c256_i64 = arith.constant 256 : i64
    %c128_i64 = arith.constant 128 : i64
    %c0_i64 = arith.constant 0 : i64
    %c16 = arith.constant 16 : index
    %c96_i32 = arith.constant 96 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.index_cast %1 : i32 to index
    %5 = hivm.hir.pointer_cast(%c0_i64) : memref<32xi32, #hivm.address_space<ub>>
    %cast = memref.cast %5 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %6 = hivm.hir.pointer_cast(%c128_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_1 = memref.cast %5 : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_2 = memref.cast %6 : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_int32_t_to_float_1d_with_mode(%cast_1, %cast_2, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    %expand_shape = memref.expand_shape %6 [[0, 1]] output_shape [1, 32] {vector_producer_to_fuse_0} : memref<32xf32, #hivm.address_space<ub>> into memref<1x32xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c256_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_3 = memref.cast %expand_shape : memref<1x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_4 = memref.cast %7 : memref<8x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_first_axis_align_2d_float(%cast_3, %cast_4) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %8 = hivm.hir.get_sub_block_idx -> i64
    %9 = arith.index_cast %8 : i64 to index
    %10 = affine.apply #map14()[%9]
    hivm.hir.set_ffts_base_addr %arg0
    %11 = hivm.hir.get_block_idx -> i64
    %12 = arith.index_cast %11 : i64 to index
    %13 = affine.apply #map()[%12]
    %view = memref.view %arg2[%13][] : memref<?xi8, #hivm.address_space<gm>> to memref<4x32x32xf32, #hivm.address_space<gm>>
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    scf.for %arg9 = %3 to %1 step %c96_i32  : i32 {
      %14 = arith.index_cast %arg9 : i32 to index
      %15 = affine.min #map1(%14)[%4]
      scf.for %arg10 = %c0 to %15 step %c1 {
        %16 = arith.index_cast %arg10 : index to i64
        %subview = memref.subview %view[%arg10, 0, 0] [1, 32, 32] [1, 1, 1] : memref<4x32x32xf32, #hivm.address_space<gm>> to memref<1x32x32xf32, strided<[1024, 32, 1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x32x32xf32, strided<[1024, 32, 1], offset: ?>, #hivm.address_space<gm>> into memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = %16
        scf.for %arg11 = %c0 to %c32 step %c16 {
          %subview_5 = memref.subview %5[%arg11] [16] [1] : memref<32xi32, #hivm.address_space<ub>> to memref<16xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %subview_6 = memref.subview %subview_5[%10] [8] [1] {to_be_bubbled_slice} : memref<16xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
          %17 = hivm.hir.pointer_cast(%c1280_i64) : memref<8xf32, #hivm.address_space<ub>>
          %cast_7 = memref.cast %subview_6 : memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_8 = memref.cast %17 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcast_int32_t_to_float_1d_with_mode(%cast_7, %cast_8, %c0_i32) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
          %expand_shape_9 = memref.expand_shape %17 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
          %18 = hivm.hir.pointer_cast(%c1312_i64) : memref<8x32xf32, #hivm.address_space<ub>>
          %19 = hivm.hir.pointer_cast(%c2336_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_10 = memref.cast %expand_shape_9 : memref<8x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_11 = memref.cast %18 : memref<8x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
          %cast_12 = memref.cast %19 : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @broadcast_last_axis_align_2d_float(%cast_10, %cast_11, %cast_12) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %20 = hivm.hir.pointer_cast(%c1312_i64) : memref<8x32xi1, #hivm.address_space<ub>>
          %collapse_shape_13 = memref.collapse_shape %18 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
          %collapse_shape_14 = memref.collapse_shape %7 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
          %collapse_shape_15 = memref.collapse_shape %20 [[0, 1]] : memref<8x32xi1, #hivm.address_space<ub>> into memref<256xi1, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_16 = memref.cast %collapse_shape_13 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_17 = memref.cast %collapse_shape_14 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_18 = memref.cast %collapse_shape_15 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vcmp_ge_1d_float(%cast_16, %cast_17, %cast_18) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          %21 = hivm.hir.pointer_cast(%c2592_i64) : memref<8x32xf32, #hivm.address_space<ub>>
          %collapse_shape_19 = memref.collapse_shape %21 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
          %22 = hivm.hir.pointer_cast(%c3616_i64) : memref<24xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_20 = memref.cast %collapse_shape_15 : memref<256xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_21 = memref.cast %collapse_shape_19 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_22 = memref.cast %22 : memref<24xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @vsel_ss_1d_float(%cast_20, %cst, %cst_0, %cast_21, %cast_22) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %subview_23 = memref.subview %collapse_shape[%arg11, 0] [16, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_24 = memref.subview %subview_23[%10, 0] [8, 32] [1, 1] {to_be_bubbled_slice} : memref<16x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>>
          %collapse_shape_25 = memref.collapse_shape %subview_24 [[0, 1]] : memref<8x32xf32, strided<[32, 1], offset: ?>, #hivm.address_space<gm>> into memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %cast_26 = memref.cast %collapse_shape_19 : memref<256xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_27 = memref.cast %collapse_shape_25 : memref<256xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
          func.call @store_ubuf_to_gm_1d_float(%cast_26, %cast_27, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        }
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 4 ffts_base_addr = %arg0
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
    }
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
