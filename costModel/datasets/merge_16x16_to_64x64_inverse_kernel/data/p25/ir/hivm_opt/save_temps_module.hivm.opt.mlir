#map = affine_map<()[s0, s1] -> (((s0 - s1) floordiv 24) mod 2)>
#map1 = affine_map<()[s0, s1] -> ((s0 - s1) floordiv 24)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1 * 256)>
#map3 = affine_map<()[s0, s1] -> (s0 + s1 * 256 + 16)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1 * 256 + 32)>
#map5 = affine_map<()[s0, s1] -> (0, s0 - s1)>
#map6 = affine_map<()[s0] -> (16, s0)>
#map7 = affine_map<()[s0, s1] -> (16, s1, s0)>
#map8 = affine_map<()[s0, s1] -> (s0 - s1)>
#map9 = affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>
#map10 = affine_map<()[s0] -> (s0 floordiv 16)>
#map11 = affine_map<()[s0] -> (s0 mod 16)>
#map12 = affine_map<()[s0, s1] -> (s0 + s1 * 256 + 48)>
#map13 = affine_map<(d0) -> (d0 * 16384)>
#map14 = affine_map<(d0) -> (d0 * 16384 + 1024)>
#map15 = affine_map<(d0) -> (d0 * 16384 + 2048)>
#map16 = affine_map<(d0) -> (d0 * 16384 + 3072)>
#map17 = affine_map<(d0) -> (d0 * 16384 + 4096)>
#map18 = affine_map<(d0) -> (d0 * 16384 + 5120)>
#map19 = affine_map<(d0) -> (d0 * 16384 + 6144)>
#map20 = affine_map<(d0) -> (d0 * 16384 + 7168)>
#map21 = affine_map<(d0) -> (d0 * 16384 + 8192)>
#map22 = affine_map<(d0) -> (d0 * 16384 + 9216)>
#map23 = affine_map<(d0) -> (d0 * 16384 + 10240)>
#map24 = affine_map<(d0) -> (d0 * 16384 + 11264)>
#map25 = affine_map<(d0) -> (d0 * 16384 + 12288)>
#map26 = affine_map<(d0) -> (d0 * 16384 + 13312)>
#map27 = affine_map<(d0) -> (d0 * 16384 + 14336)>
#map28 = affine_map<(d0) -> (d0 * 16384 + 15360)>
#map29 = affine_map<()[s0] -> (s0 * 8)>
#map30 = affine_map<()[s0] -> (-s0 + (s0 floordiv 256) * 256 + 64, 0)>
#map31 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 256)>
#map32 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 256, 0)>
#map33 = affine_map<()[s0, s1, s2] -> (s1 - s2, s0 * 8)>
#map34 = affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2, 8)>
#map35 = affine_map<()[s0, s1] -> (s1 * 8, s0)>
#map36 = affine_map<()[s0, s1] -> (16, s1 * 8 + 8, s0)>
#map37 = affine_map<()[s0, s1, s2] -> (s2 * 8, s1, s0)>
#map38 = affine_map<()[s0, s1] -> (s0 - s1 * 8)>
#map39 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
#map40 = affine_map<()[s0] -> (s0 + 1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>,   hivm.aic_bitcode = #hivm.aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aic.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.aiv.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aic.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/lib/meta_op.mix.aiv.bc">, hivm.module_core_type = #hivm.module_core_type<MIX>, memref.memref_as_ptr} {
  func.func @merge_16x16_to_64x64_inverse_kernel_infer_workspace_shape_function() -> index attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_workspace_shape_function>} {
    %c16384 = arith.constant 16384 : index
    return %c16384 : index
  }
  func.func @merge_16x16_to_64x64_inverse_kernel_infer_task_type_function() -> i8 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<infer_task_type_function>} {
    %c32_i8 = arith.constant 32 : i8
    return %c32_i8 : i8
  }
  func.func @merge_16x16_to_64x64_inverse_kernel_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i8 = arith.constant 0 : i8
    %cst = arith.constant 0.000000e+00 : f32
    %c48_i32 = arith.constant 48 : i32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16 = arith.constant 16 : index
    %c-16_i32 = arith.constant -16 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c-48_i32 = arith.constant -48 : i32
    %true = arith.constant true
    %false = arith.constant false
    %c24_i32 = arith.constant 24 : i32
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c5_i64 = arith.constant 5 : i64
    %c6_i64 = arith.constant 6 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c20480_i64 = arith.constant 20480 : i64
    %c0_i64 = arith.constant 0 : i64
    %c21504_i64 = arith.constant 21504 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c22528_i64 = arith.constant 22528 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c23552_i64 = arith.constant 23552 : i64
    %c3072_i64 = arith.constant 3072 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c25600_i64 = arith.constant 25600 : i64
    %c5120_i64 = arith.constant 5120 : i64
    %c26624_i64 = arith.constant 26624 : i64
    %c6144_i64 = arith.constant 6144 : i64
    %c27648_i64 = arith.constant 27648 : i64
    %c7168_i64 = arith.constant 7168 : i64
    %c28672_i64 = arith.constant 28672 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c29696_i64 = arith.constant 29696 : i64
    %c9216_i64 = arith.constant 9216 : i64
    %c30720_i64 = arith.constant 30720 : i64
    %c10240_i64 = arith.constant 10240 : i64
    %c31744_i64 = arith.constant 31744 : i64
    %c11264_i64 = arith.constant 11264 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c33792_i64 = arith.constant 33792 : i64
    %c13312_i64 = arith.constant 13312 : i64
    %c34816_i64 = arith.constant 34816 : i64
    %c14336_i64 = arith.constant 14336 : i64
    %c35840_i64 = arith.constant 35840 : i64
    %c15360_i64 = arith.constant 15360 : i64
    %c36864_i64 = arith.constant 36864 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c37888_i64 = arith.constant 37888 : i64
    %c17408_i64 = arith.constant 17408 : i64
    %c38912_i64 = arith.constant 38912 : i64
    %c18432_i64 = arith.constant 18432 : i64
    %c19456_i64 = arith.constant 19456 : i64
    %c39936_i64 = arith.constant 39936 : i64
    %0 = hivm.hir.pointer_cast(%c19456_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %1 = hivm.hir.pointer_cast(%c39936_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %2 = hivm.hir.pointer_cast(%c18432_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c38912_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %4 = hivm.hir.pointer_cast(%c17408_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %5 = hivm.hir.pointer_cast(%c37888_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %6 = hivm.hir.pointer_cast(%c16384_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %7 = hivm.hir.pointer_cast(%c36864_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %8 = hivm.hir.pointer_cast(%c15360_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %9 = hivm.hir.pointer_cast(%c35840_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %10 = hivm.hir.pointer_cast(%c14336_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %11 = hivm.hir.pointer_cast(%c34816_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %12 = hivm.hir.pointer_cast(%c13312_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %13 = hivm.hir.pointer_cast(%c33792_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %14 = hivm.hir.pointer_cast(%c12288_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %15 = hivm.hir.pointer_cast(%c32768_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %16 = hivm.hir.pointer_cast(%c11264_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %17 = hivm.hir.pointer_cast(%c31744_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %18 = hivm.hir.pointer_cast(%c10240_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %19 = hivm.hir.pointer_cast(%c30720_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %20 = hivm.hir.pointer_cast(%c9216_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %21 = hivm.hir.pointer_cast(%c29696_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %22 = hivm.hir.pointer_cast(%c8192_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %23 = hivm.hir.pointer_cast(%c28672_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %24 = hivm.hir.pointer_cast(%c7168_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %25 = hivm.hir.pointer_cast(%c27648_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %26 = hivm.hir.pointer_cast(%c6144_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %27 = hivm.hir.pointer_cast(%c26624_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %28 = hivm.hir.pointer_cast(%c5120_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %29 = hivm.hir.pointer_cast(%c25600_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %30 = hivm.hir.pointer_cast(%c4096_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %31 = hivm.hir.pointer_cast(%c24576_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %32 = hivm.hir.pointer_cast(%c3072_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %33 = hivm.hir.pointer_cast(%c23552_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %34 = hivm.hir.pointer_cast(%c2048_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %35 = hivm.hir.pointer_cast(%c22528_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %36 = hivm.hir.pointer_cast(%c1024_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %37 = hivm.hir.pointer_cast(%c21504_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %38 = hivm.hir.pointer_cast(%c0_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    %39 = hivm.hir.pointer_cast(%c20480_i64) : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
    hivm.hir.set_ffts_base_addr %arg0
    %40 = arith.muli %arg6, %arg7 : i32
    %41 = arith.muli %40, %arg8 : i32
    %42 = hivm.hir.get_block_idx -> i64
    %43 = arith.trunci %42 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 8 ffts_base_addr = %arg0
    %44 = arith.muli %arg8, %arg7 : i32
    %45 = arith.index_cast %arg5 : i32 to index
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID6>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
    scf.for %arg9 = %43 to %41 step %c24_i32  : i32 {
      %46 = arith.index_cast %arg9 : i32 to index
      %47 = arith.index_cast %43 : i32 to index
      %48 = affine.apply #map()[%46, %47]
      %49 = arith.index_cast %48 : index to i64
      %50 = arith.cmpi eq, %49, %c1_i64 : i64
      %51 = arith.select %50, %1, %0 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %52 = arith.index_cast %arg9 : i32 to index
      %53 = arith.index_cast %43 : i32 to index
      %54 = affine.apply #map()[%52, %53]
      %55 = arith.index_cast %54 : index to i64
      %56 = arith.cmpi eq, %55, %c1_i64 : i64
      %57 = arith.select %56, %3, %2 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %58 = arith.index_cast %arg9 : i32 to index
      %59 = arith.index_cast %43 : i32 to index
      %60 = affine.apply #map()[%58, %59]
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.cmpi eq, %61, %c1_i64 : i64
      %63 = arith.select %62, %5, %4 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %64 = arith.index_cast %arg9 : i32 to index
      %65 = arith.index_cast %43 : i32 to index
      %66 = affine.apply #map()[%64, %65]
      %67 = arith.index_cast %66 : index to i64
      %68 = arith.cmpi eq, %67, %c1_i64 : i64
      %69 = arith.select %68, %7, %6 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %70 = arith.index_cast %arg9 : i32 to index
      %71 = arith.index_cast %43 : i32 to index
      %72 = affine.apply #map()[%70, %71]
      %73 = arith.index_cast %72 : index to i64
      %74 = arith.cmpi eq, %73, %c1_i64 : i64
      %75 = arith.select %74, %9, %8 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %76 = arith.index_cast %arg9 : i32 to index
      %77 = arith.index_cast %43 : i32 to index
      %78 = affine.apply #map()[%76, %77]
      %79 = arith.index_cast %78 : index to i64
      %80 = arith.cmpi eq, %79, %c1_i64 : i64
      %81 = arith.select %80, %11, %10 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %82 = arith.index_cast %arg9 : i32 to index
      %83 = arith.index_cast %43 : i32 to index
      %84 = affine.apply #map()[%82, %83]
      %85 = arith.index_cast %84 : index to i64
      %86 = arith.cmpi eq, %85, %c1_i64 : i64
      %87 = arith.select %86, %13, %12 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %88 = arith.index_cast %arg9 : i32 to index
      %89 = arith.index_cast %43 : i32 to index
      %90 = affine.apply #map()[%88, %89]
      %91 = arith.index_cast %90 : index to i64
      %92 = arith.cmpi eq, %91, %c1_i64 : i64
      %93 = arith.select %92, %15, %14 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %94 = arith.index_cast %arg9 : i32 to index
      %95 = arith.index_cast %43 : i32 to index
      %96 = affine.apply #map()[%94, %95]
      %97 = arith.index_cast %96 : index to i64
      %98 = arith.cmpi eq, %97, %c1_i64 : i64
      %99 = arith.select %98, %17, %16 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %100 = arith.index_cast %arg9 : i32 to index
      %101 = arith.index_cast %43 : i32 to index
      %102 = affine.apply #map()[%100, %101]
      %103 = arith.index_cast %102 : index to i64
      %104 = arith.cmpi eq, %103, %c1_i64 : i64
      %105 = arith.select %104, %19, %18 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %106 = arith.index_cast %arg9 : i32 to index
      %107 = arith.index_cast %43 : i32 to index
      %108 = affine.apply #map()[%106, %107]
      %109 = arith.index_cast %108 : index to i64
      %110 = arith.cmpi eq, %109, %c1_i64 : i64
      %111 = arith.select %110, %21, %20 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %112 = arith.index_cast %arg9 : i32 to index
      %113 = arith.index_cast %43 : i32 to index
      %114 = affine.apply #map()[%112, %113]
      %115 = arith.index_cast %114 : index to i64
      %116 = arith.cmpi eq, %115, %c1_i64 : i64
      %117 = arith.select %116, %23, %22 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %118 = arith.index_cast %arg9 : i32 to index
      %119 = arith.index_cast %43 : i32 to index
      %120 = affine.apply #map()[%118, %119]
      %121 = arith.index_cast %120 : index to i64
      %122 = arith.cmpi eq, %121, %c1_i64 : i64
      %123 = arith.select %122, %25, %24 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %124 = arith.index_cast %arg9 : i32 to index
      %125 = arith.index_cast %43 : i32 to index
      %126 = affine.apply #map()[%124, %125]
      %127 = arith.index_cast %126 : index to i64
      %128 = arith.cmpi eq, %127, %c1_i64 : i64
      %129 = arith.select %128, %27, %26 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %130 = arith.index_cast %arg9 : i32 to index
      %131 = arith.index_cast %43 : i32 to index
      %132 = affine.apply #map()[%130, %131]
      %133 = arith.index_cast %132 : index to i64
      %134 = arith.cmpi eq, %133, %c1_i64 : i64
      %135 = arith.select %134, %29, %28 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %136 = arith.index_cast %arg9 : i32 to index
      %137 = arith.index_cast %43 : i32 to index
      %138 = affine.apply #map()[%136, %137]
      %139 = arith.index_cast %138 : index to i64
      %140 = arith.cmpi eq, %139, %c1_i64 : i64
      %141 = arith.select %140, %31, %30 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %142 = arith.index_cast %arg9 : i32 to index
      %143 = arith.index_cast %43 : i32 to index
      %144 = affine.apply #map()[%142, %143]
      %145 = arith.index_cast %144 : index to i64
      %146 = arith.cmpi eq, %145, %c1_i64 : i64
      %147 = arith.select %146, %33, %32 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %148 = arith.index_cast %arg9 : i32 to index
      %149 = arith.index_cast %43 : i32 to index
      %150 = affine.apply #map()[%148, %149]
      %151 = arith.index_cast %150 : index to i64
      %152 = arith.cmpi eq, %151, %c1_i64 : i64
      %153 = arith.select %152, %35, %34 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %154 = arith.index_cast %arg9 : i32 to index
      %155 = arith.index_cast %43 : i32 to index
      %156 = affine.apply #map()[%154, %155]
      %157 = arith.index_cast %156 : index to i64
      %158 = arith.cmpi eq, %157, %c1_i64 : i64
      %159 = arith.select %158, %37, %36 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %160 = arith.index_cast %arg9 : i32 to index
      %161 = arith.index_cast %43 : i32 to index
      %162 = affine.apply #map()[%160, %161]
      %163 = arith.index_cast %162 : index to i64
      %164 = arith.cmpi eq, %163, %c1_i64 : i64
      %165 = arith.select %164, %39, %38 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>>
      %166 = arith.index_cast %arg9 : i32 to index
      %167 = arith.index_cast %43 : i32 to index
      %168 = affine.apply #map1()[%166, %167]
      %169 = arith.index_cast %168 : index to i64
      %170 = arith.index_cast %arg9 : i32 to index
      %171 = arith.index_cast %43 : i32 to index
      %172 = affine.apply #map()[%170, %171]
      %173 = arith.index_cast %172 : index to i1
      %174 = arith.select %173, %c2_i64, %c3_i64 : i64
      %175 = arith.index_cast %172 : index to i1
      %176 = arith.select %175, %c5_i64, %c6_i64 : i64
      %177 = arith.index_cast %172 : index to i1
      %178 = arith.xori %177, %true : i1
      %179 = arith.extui %178 : i1 to i64
      hivm.hir.set_mask_norm
      %180 = arith.remsi %arg9, %arg8 : i32
      %181 = arith.divsi %arg9, %arg8 : i32
      %182 = arith.remsi %181, %arg7 : i32
      %183 = arith.divsi %arg9, %44 : i32
      %184 = arith.remsi %183, %arg6 : i32
      %185 = arith.divsi %182, %c4_i32 : i32
      %186 = arith.remsi %182, %c4_i32 : i32
      %187 = arith.muli %185, %arg5 : i32
      %188 = arith.muli %187, %c4_i32 : i32
      %189 = arith.addi %188, %186 : i32
      %190 = arith.muli %189, %c64_i32 : i32
      %191 = arith.index_cast %190 : i32 to index
      %192 = arith.cmpi eq, %180, %c0_i32 : i32
      %193 = arith.muli %184, %c64_i32 : i32
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10
      scf.if %192 {
        %194 = arith.addi %193, %c16_i32 : i32
        %195 = arith.maxsi %194, %c0_i32 : i32
        %196 = arith.index_cast %195 : i32 to index
        %197 = affine.apply #map2()[%191, %196]
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%197], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %198 = arith.addi %193, %c32_i32 : i32
        %199 = arith.maxsi %198, %c0_i32 : i32
        %200 = arith.index_cast %199 : i32 to index
        %201 = affine.apply #map2()[%191, %200]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%201], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %202 = affine.apply #map3()[%191, %200]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%202], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %203 = arith.addi %193, %c48_i32 : i32
        %204 = arith.maxsi %203, %c0_i32 : i32
        %205 = arith.index_cast %204 : i32 to index
        %206 = affine.apply #map2()[%191, %205]
        %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%206], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %207 = affine.apply #map3()[%191, %205]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%207], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %208 = affine.apply #map4()[%191, %205]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%208], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %209 = affine.max #map5()[%45, %196]
        %210 = affine.min #map6()[%209]
        %211 = arith.subi %c-16_i32, %193 : i32
        %212 = arith.maxsi %211, %c0_i32 : i32
        %213 = arith.index_cast %212 : i32 to index
        %214 = affine.min #map7()[%213, %209]
        %215 = affine.apply #map8()[%210, %214]
        %216 = arith.cmpi slt, %215, %c16 : index
        %subview = memref.subview %reinterpret_cast[0, 0] [%215, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %217 = affine.apply #map9()[%210, %214]
        %218 = affine.apply #map10()[%214]
        %219 = affine.apply #map11()[%214]
        %subview_5 = memref.subview %165[0, %218, %219, 0] [2, %217, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %216 {
          hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %collapse_shape = memref.collapse_shape %165 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %cast_154 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %176]
        %cast = memref.cast %subview : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_6 = memref.cast %subview_5 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast, %cast_6) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        %220 = affine.max #map5()[%45, %200]
        %221 = affine.min #map6()[%220]
        %222 = arith.subi %c-32_i32, %193 : i32
        %223 = arith.maxsi %222, %c0_i32 : i32
        %224 = arith.index_cast %223 : i32 to index
        %225 = affine.min #map7()[%224, %220]
        %226 = affine.apply #map8()[%221, %225]
        %227 = arith.cmpi slt, %226, %c16 : index
        %subview_7 = memref.subview %reinterpret_cast_0[0, 0] [%226, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %228 = affine.apply #map9()[%221, %225]
        %229 = affine.apply #map10()[%225]
        %230 = affine.apply #map11()[%225]
        %subview_8 = memref.subview %159[0, %229, %230, 0] [2, %228, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %227 {
          hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %collapse_shape = memref.collapse_shape %159 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %cast_154 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        %cast_9 = memref.cast %subview_7 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_10 = memref.cast %subview_8 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_9, %cast_10) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        %subview_11 = memref.subview %reinterpret_cast_1[0, 0] [%226, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_12 = memref.subview %153[0, %229, %230, 0] [2, %228, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %227 {
          hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %collapse_shape = memref.collapse_shape %153 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %cast_154 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        %cast_13 = memref.cast %subview_11 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_14 = memref.cast %subview_12 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_13, %cast_14) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        %231 = affine.max #map5()[%45, %205]
        %232 = affine.min #map6()[%231]
        %233 = arith.subi %c-48_i32, %193 : i32
        %234 = arith.maxsi %233, %c0_i32 : i32
        %235 = arith.index_cast %234 : i32 to index
        %236 = affine.min #map7()[%235, %231]
        %237 = affine.apply #map8()[%232, %236]
        %238 = arith.cmpi slt, %237, %c16 : index
        %subview_15 = memref.subview %reinterpret_cast_2[0, 0] [%237, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %239 = affine.apply #map9()[%232, %236]
        %240 = affine.apply #map10()[%236]
        %241 = affine.apply #map11()[%236]
        %subview_16 = memref.subview %147[0, %240, %241, 0] [2, %239, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %238 {
          hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %collapse_shape = memref.collapse_shape %147 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %cast_154 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        %cast_17 = memref.cast %subview_15 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_18 = memref.cast %subview_16 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_17, %cast_18) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        %subview_19 = memref.subview %reinterpret_cast_3[0, 0] [%237, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_20 = memref.subview %141[0, %240, %241, 0] [2, %239, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %238 {
          hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %collapse_shape = memref.collapse_shape %141 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %cast_154 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        %cast_21 = memref.cast %subview_19 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_22 = memref.cast %subview_20 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_21, %cast_22) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        %subview_23 = memref.subview %reinterpret_cast_4[0, 0] [%237, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_24 = memref.subview %135[0, %240, %241, 0] [2, %239, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %238 {
          hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %collapse_shape = memref.collapse_shape %135 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %cast_154 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        %cast_25 = memref.cast %subview_23 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_26 = memref.cast %subview_24 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_25, %cast_26) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        %242 = arith.maxsi %193, %c0_i32 : i32
        %243 = arith.index_cast %242 : i32 to index
        %244 = affine.apply #map2()[%191, %243]
        %reinterpret_cast_27 = memref.reinterpret_cast %arg4 to offset: [%244], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %245 = affine.apply #map3()[%191, %196]
        %reinterpret_cast_28 = memref.reinterpret_cast %arg4 to offset: [%245], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %246 = affine.apply #map4()[%191, %200]
        %reinterpret_cast_29 = memref.reinterpret_cast %arg4 to offset: [%246], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %247 = affine.apply #map12()[%191, %205]
        %reinterpret_cast_30 = memref.reinterpret_cast %arg4 to offset: [%247], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %248 = affine.max #map5()[%45, %243]
        %249 = affine.min #map6()[%248]
        %250 = arith.subi %c0_i32, %193 : i32
        %251 = arith.maxsi %250, %c0_i32 : i32
        %252 = arith.index_cast %251 : i32 to index
        %253 = affine.min #map7()[%252, %248]
        %254 = affine.apply #map8()[%249, %253]
        %255 = arith.cmpi slt, %254, %c16 : index
        %subview_31 = memref.subview %reinterpret_cast_27[0, 0] [%254, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %256 = affine.apply #map9()[%249, %253]
        %257 = affine.apply #map10()[%253]
        %258 = affine.apply #map11()[%253]
        %subview_32 = memref.subview %129[0, %257, %258, 0] [2, %256, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %255 {
          %collapse_shape = memref.collapse_shape %129 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
          %cast_154 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        %cast_33 = memref.cast %subview_31 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_34 = memref.cast %subview_32 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_33, %cast_34) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        %subview_35 = memref.subview %reinterpret_cast_28[0, 0] [%215, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_36 = memref.subview %123[0, %218, %219, 0] [2, %217, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %216 {
          %collapse_shape = memref.collapse_shape %123 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
          %cast_154 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        %cast_37 = memref.cast %subview_35 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_38 = memref.cast %subview_36 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_37, %cast_38) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        %subview_39 = memref.subview %reinterpret_cast_29[0, 0] [%226, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_40 = memref.subview %117[0, %229, %230, 0] [2, %228, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %227 {
          %collapse_shape = memref.collapse_shape %117 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
          %cast_154 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        %cast_41 = memref.cast %subview_39 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_42 = memref.cast %subview_40 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_41, %cast_42) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        %subview_43 = memref.subview %reinterpret_cast_30[0, 0] [%237, 16] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_44 = memref.subview %111[0, %240, %241, 0] [2, %239, 16, 8] [1, 1, 1, 1] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %238 {
          hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %collapse_shape = memref.collapse_shape %111 [[0, 1, 2, 3]] : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> into memref<256xf32, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
          %cast_154 = memref.cast %collapse_shape : memref<256xf32, #hivm.address_space<cbuf>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>
          func.call @set_l1_2d_float(%cst, %cast_154) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %174]
        %cast_45 = memref.cast %subview_43 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_46 = memref.cast %subview_44 : memref<2x?x16x8xf32, strided<[128, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_45, %cast_46) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 11 ffts_base_addr = %arg0
        %259 = hivm.hir.pointer_cast(%c0_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
        %cast_47 = memref.cast %123 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_48 = memref.cast %165 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_49 = memref.cast %259 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_47, %cast_48, %true, %215, %c16, %c16, %cast_49, %c0_i64, %c-1_i64, %c-1_i64, %c-1_i64, %169, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %260 = hivm.hir.get_block_idx -> i64
        %261 = arith.index_cast %260 : i64 to index
        %262 = affine.apply #map13(%261)
        %view = memref.view %arg2[%262][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_50 = memref.cast %259 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_51 = memref.cast %view : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_50, %cast_51, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_52 = memref.cast %view : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_53 = memref.cast %105 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_52, %cast_53) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
        %263 = hivm.hir.pointer_cast(%c1024_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        %cast_54 = memref.cast %105 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_55 = memref.cast %129 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_56 = memref.cast %263 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_54, %cast_55, %true, %c16, %c16, %c16, %cast_56, %c0_i64, %c-1_i64, %c-1_i64, %c-1_i64, %169, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %264 = hivm.hir.get_block_idx -> i64
        %265 = arith.index_cast %264 : i64 to index
        %266 = affine.apply #map14(%265)
        %view_57 = memref.view %arg2[%266][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 9
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_58 = memref.cast %263 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_59 = memref.cast %view_57 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_58, %cast_59, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12 ffts_base_addr = %arg0
        %267 = hivm.hir.get_block_idx -> i64
        %268 = arith.index_cast %267 : i64 to index
        %269 = affine.apply #map15(%268)
        %view_60 = memref.view %arg2[%269][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10
        %cast_61 = memref.cast %view_60 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_62 = memref.cast %99 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_61, %cast_62) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 8 ffts_base_addr = %arg0
        %270 = hivm.hir.get_block_idx -> i64
        %271 = arith.index_cast %270 : i64 to index
        %272 = affine.apply #map16(%271)
        %view_63 = memref.view %arg2[%272][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10
        %cast_64 = memref.cast %view_63 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_65 = memref.cast %93 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_64, %cast_65) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7 ffts_base_addr = %arg0
        %273 = hivm.hir.pointer_cast(%c2048_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        %cast_66 = memref.cast %117 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_67 = memref.cast %153 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_68 = memref.cast %273 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_66, %cast_67, %true, %226, %c16, %c16, %cast_68, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %274 = hivm.hir.get_block_idx -> i64
        %275 = arith.index_cast %274 : i64 to index
        %276 = affine.apply #map17(%275)
        %view_69 = memref.view %arg2[%276][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_70 = memref.cast %273 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_71 = memref.cast %view_69 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_70, %cast_71, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_72 = memref.cast %view_69 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_73 = memref.cast %87 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_72, %cast_73) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
        %277 = hivm.hir.pointer_cast(%c3072_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        %cast_74 = memref.cast %87 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_75 = memref.cast %123 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_76 = memref.cast %277 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_74, %cast_75, %true, %c16, %c16, %c16, %cast_76, %c0_i64, %c-1_i64, %c-1_i64, %c-1_i64, %169, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %278 = hivm.hir.get_block_idx -> i64
        %279 = arith.index_cast %278 : i64 to index
        %280 = affine.apply #map18(%279)
        %view_77 = memref.view %arg2[%280][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_78 = memref.cast %277 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_79 = memref.cast %view_77 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_78, %cast_79, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12 ffts_base_addr = %arg0
        %281 = hivm.hir.get_block_idx -> i64
        %282 = arith.index_cast %281 : i64 to index
        %283 = affine.apply #map19(%282)
        %view_80 = memref.view %arg2[%283][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10
        %cast_81 = memref.cast %view_80 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_82 = memref.cast %81 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_81, %cast_82) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5 ffts_base_addr = %arg0
        %284 = hivm.hir.pointer_cast(%c4096_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        %cast_83 = memref.cast %111 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_84 = memref.cast %135 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_85 = memref.cast %284 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_83, %cast_84, %true, %237, %c16, %c16, %cast_85, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %285 = hivm.hir.get_block_idx -> i64
        %286 = arith.index_cast %285 : i64 to index
        %287 = affine.apply #map20(%286)
        %view_86 = memref.view %arg2[%287][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_87 = memref.cast %284 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_88 = memref.cast %view_86 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_87, %cast_88, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_89 = memref.cast %view_86 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_90 = memref.cast %75 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_89, %cast_90) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
        %288 = hivm.hir.pointer_cast(%c5120_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        %cast_91 = memref.cast %75 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_92 = memref.cast %117 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_93 = memref.cast %288 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_91, %cast_92, %true, %c16, %c16, %c16, %cast_93, %c0_i64, %c-1_i64, %c-1_i64, %c-1_i64, %169, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %289 = hivm.hir.get_block_idx -> i64
        %290 = arith.index_cast %289 : i64 to index
        %291 = affine.apply #map21(%290)
        %view_94 = memref.view %arg2[%291][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_95 = memref.cast %288 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_96 = memref.cast %view_94 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_95, %cast_96, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12 ffts_base_addr = %arg0
        %292 = hivm.hir.pointer_cast(%c6144_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
        %cast_97 = memref.cast %159 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_98 = memref.cast %129 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_99 = memref.cast %292 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_97, %cast_98, %true, %226, %c16, %c16, %cast_99, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        %cast_100 = memref.cast %153 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_101 = memref.cast %99 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_102 = memref.cast %292 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_100, %cast_101, %false, %226, %c16, %c16, %cast_102, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %293 = hivm.hir.get_block_idx -> i64
        %294 = arith.index_cast %293 : i64 to index
        %295 = affine.apply #map22(%294)
        %view_103 = memref.view %arg2[%295][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_104 = memref.cast %292 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_105 = memref.cast %view_103 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_104, %cast_105, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_106 = memref.cast %view_103 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_107 = memref.cast %69 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_106, %cast_107) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
        %296 = hivm.hir.pointer_cast(%c7168_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
        %cast_108 = memref.cast %117 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_109 = memref.cast %69 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_110 = memref.cast %296 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_108, %cast_109, %true, %226, %c16, %c16, %cast_110, %c-1_i64, %c0_i64, %c-1_i64, %c-1_i64, %169, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %297 = hivm.hir.get_block_idx -> i64
        %298 = arith.index_cast %297 : i64 to index
        %299 = affine.apply #map23(%298)
        %view_111 = memref.view %arg2[%299][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_112 = memref.cast %296 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_113 = memref.cast %view_111 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_112, %cast_113, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12 ffts_base_addr = %arg0
        %300 = hivm.hir.get_block_idx -> i64
        %301 = arith.index_cast %300 : i64 to index
        %302 = affine.apply #map24(%301)
        %view_114 = memref.view %arg2[%302][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 10
        %cast_115 = memref.cast %view_114 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_116 = memref.cast %63 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_115, %cast_116) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2 ffts_base_addr = %arg0
        %303 = hivm.hir.pointer_cast(%c8192_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
        %cast_117 = memref.cast %141 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_118 = memref.cast %123 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_119 = memref.cast %303 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_117, %cast_118, %true, %237, %c16, %c16, %cast_119, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        %cast_120 = memref.cast %135 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_121 = memref.cast %81 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_122 = memref.cast %303 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_120, %cast_121, %false, %237, %c16, %c16, %cast_122, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %304 = hivm.hir.get_block_idx -> i64
        %305 = arith.index_cast %304 : i64 to index
        %306 = affine.apply #map25(%305)
        %view_123 = memref.view %arg2[%306][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_124 = memref.cast %303 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_125 = memref.cast %view_123 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_124, %cast_125, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_126 = memref.cast %view_123 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_127 = memref.cast %57 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_126, %cast_127) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
        %307 = hivm.hir.pointer_cast(%c9216_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
        %cast_128 = memref.cast %111 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_129 = memref.cast %57 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_130 = memref.cast %307 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_128, %cast_129, %true, %237, %c16, %c16, %cast_130, %c-1_i64, %c0_i64, %c-1_i64, %c-1_i64, %169, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %308 = hivm.hir.get_block_idx -> i64
        %309 = arith.index_cast %308 : i64 to index
        %310 = affine.apply #map26(%309)
        %view_131 = memref.view %arg2[%310][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_132 = memref.cast %307 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_133 = memref.cast %view_131 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_132, %cast_133, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12 ffts_base_addr = %arg0
        %311 = hivm.hir.pointer_cast(%c10240_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
        %cast_134 = memref.cast %147 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_135 = memref.cast %129 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_136 = memref.cast %311 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_134, %cast_135, %true, %237, %c16, %c16, %cast_136, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        %cast_137 = memref.cast %141 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_138 = memref.cast %93 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_139 = memref.cast %311 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_137, %cast_138, %false, %237, %c16, %c16, %cast_139, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        %cast_140 = memref.cast %135 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_141 = memref.cast %63 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_142 = memref.cast %311 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_140, %cast_141, %false, %237, %c16, %c16, %cast_142, %c-1_i64, %c-1_i64, %176, %c-1_i64, %169, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %312 = hivm.hir.get_block_idx -> i64
        %313 = arith.index_cast %312 : i64 to index
        %314 = affine.apply #map27(%313)
        %view_143 = memref.view %arg2[%314][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_144 = memref.cast %311 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_145 = memref.cast %view_143 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_144, %cast_145, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %179]
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_146 = memref.cast %view_143 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        %cast_147 = memref.cast %51 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        func.call @nd2nz_float(%cast_146, %cast_147) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
        %315 = hivm.hir.pointer_cast(%c11264_i64) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        %cast_148 = memref.cast %111 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_149 = memref.cast %51 : memref<2x1x16x8xf32, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>
        %cast_150 = memref.cast %315 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        func.call @mma_tile_float_to_float(%cast_148, %cast_149, %true, %237, %c16, %c16, %cast_150, %c-1_i64, %c0_i64, %174, %179, %169, %c-1_i64, %c-1_i64, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %316 = hivm.hir.get_block_idx -> i64
        %317 = arith.index_cast %316 : i64 to index
        %318 = affine.apply #map28(%317)
        %view_151 = memref.view %arg2[%318][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %cast_152 = memref.cast %315 : memref<1x1x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>
        %cast_153 = memref.cast %view_151 : memref<16x16xf32, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @fixpipe_nz2nd_float_to_float_4d_to_2d(%cast_152, %cast_153, %c0_i64, %c0_i64, %false, %c0_i8) : (memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) -> ()
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 12 ffts_base_addr = %arg0
      }
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID6>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID3>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID4>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_FIX>, <EVENT_ID5>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 9
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func private @set_l1_2d_float(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @nd2nz_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @mma_tile_float_to_float(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cbuf>>, i1, index, index, index, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64, i64, i64, i64, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @fixpipe_nz2nd_float_to_float_4d_to_2d(memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i64, i64, i1, i8) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIC>, llvm.emit_c_interface}
  func.func private @arange_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_last_axis_align_2d_int32_t(memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_first_axis_align_2d_int32_t(memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmax_2d_int32_t(memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcmp_eq_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vnot_1d_bool(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vcast_bool_to_float_1d_with_temp(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_scalar_float_to_1d(f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vsel_vs_1d_float(memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmuls_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadds_vs_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @load_gm_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vmul_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @reduce_sum_ra_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @vadd_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @broadcast_first_axis_align_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @copy_ubuf_to_ubuf_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_2d_float(memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface}
  func.func @merge_16x16_to_64x64_inverse_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
    %c8544_i64 = arith.constant 8544 : i64
    %c8032_i64 = arith.constant 8032 : i64
    %c7520_i64 = arith.constant 7520 : i64
    %c7008_i64 = arith.constant 7008 : i64
    %c6496_i64 = arith.constant 6496 : i64
    %c5984_i64 = arith.constant 5984 : i64
    %c5472_i64 = arith.constant 5472 : i64
    %c5440_i64 = arith.constant 5440 : i64
    %c4928_i64 = arith.constant 4928 : i64
    %c4896_i64 = arith.constant 4896 : i64
    %c4384_i64 = arith.constant 4384 : i64
    %c3872_i64 = arith.constant 3872 : i64
    %c3808_i64 = arith.constant 3808 : i64
    %c3744_i64 = arith.constant 3744 : i64
    %c3232_i64 = arith.constant 3232 : i64
    %c2720_i64 = arith.constant 2720 : i64
    %c2624_i64 = arith.constant 2624 : i64
    %c2112_i64 = arith.constant 2112 : i64
    %c1600_i64 = arith.constant 1600 : i64
    %c1088_i64 = arith.constant 1088 : i64
    %c576_i64 = arith.constant 576 : i64
    %c64_i64 = arith.constant 64 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c48_i32 = arith.constant 48 : i32
    %c32_i32 = arith.constant 32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16 = arith.constant 16 : index
    %c256_i32 = arith.constant 256 : i32
    %c-16_i32 = arith.constant -16 : i32
    %c-32_i32 = arith.constant -32 : i32
    %c-48_i32 = arith.constant -48 : i32
    %cst_0 = arith.constant -1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c24_i32 = arith.constant 24 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg8, %arg7 : i32
    %5 = hivm.hir.pointer_cast(%c0_i64) : memref<16xi32, #hivm.address_space<ub>>
    %cast = memref.cast %5 : memref<16xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @arange_1d_int32_t(%cast, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    %expand_shape = memref.expand_shape %5 [[0, 1]] output_shape [16, 1] : memref<16xi32, #hivm.address_space<ub>> into memref<16x1xi32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c64_i64) : memref<16x8xi32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c576_i64) : memref<0xi32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_1 = memref.cast %expand_shape : memref<16x1xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_2 = memref.cast %6 : memref<16x8xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_3 = memref.cast %7 : memref<0xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_last_axis_align_2d_int32_t(%cast_1, %cast_2, %cast_3) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %8 = arith.index_cast %arg5 : i32 to index
    %9 = hivm.hir.get_sub_block_idx -> i64
    %10 = arith.index_cast %9 : i64 to index
    %11 = affine.apply #map29()[%10]
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6 ffts_base_addr = %arg0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 9 ffts_base_addr = %arg0
    %subview = memref.subview %5[%11] [8] [1] {to_be_bubbled_slice} : memref<16xi32, #hivm.address_space<ub>> to memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %expand_shape_4 = memref.expand_shape %subview [[0, 1]] output_shape [1, 8] : memref<8xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c576_i64) : memref<16x8xi32, #hivm.address_space<ub>>
    %cast_5 = memref.cast %expand_shape_4 : memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_6 = memref.cast %12 : memref<16x8xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    call @broadcast_first_axis_align_2d_int32_t(%cast_5, %cast_6) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %13 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c1600_i64) : memref<128xi32, #hivm.address_space<ub>>
    %cast_7 = memref.cast %expand_shape : memref<16x1xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_8 = memref.cast %expand_shape_4 : memref<1x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_9 = memref.cast %13 : memref<16x8xi32, #hivm.address_space<ub>> to memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
    %cast_10 = memref.cast %14 : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vmax_2d_int32_t(%cast_7, %cast_8, %cast_9, %cast_10) : (memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xi32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %15 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape = memref.collapse_shape %13 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
    %collapse_shape_11 = memref.collapse_shape %12 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
    %collapse_shape_12 = memref.collapse_shape %15 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_13 = memref.cast %collapse_shape : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_14 = memref.cast %collapse_shape_11 : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_15 = memref.cast %collapse_shape_12 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_int32_t(%cast_13, %cast_14, %cast_15) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %16 = hivm.hir.pointer_cast(%c1088_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_16 = memref.collapse_shape %16 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %17 = hivm.hir.pointer_cast(%c0_i64) : memref<0xi1, #hivm.address_space<ub>>
    %cast_17 = memref.cast %collapse_shape_12 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_18 = memref.cast %collapse_shape_16 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_19 = memref.cast %17 : memref<0xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vnot_1d_bool(%cast_17, %cast_18, %cast_19) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %18 = hivm.hir.pointer_cast(%c576_i64) : memref<16x8xi1, #hivm.address_space<ub>>
    %collapse_shape_20 = memref.collapse_shape %6 [[0, 1]] : memref<16x8xi32, #hivm.address_space<ub>> into memref<128xi32, #hivm.address_space<ub>>
    %collapse_shape_21 = memref.collapse_shape %18 [[0, 1]] : memref<16x8xi1, #hivm.address_space<ub>> into memref<128xi1, #hivm.address_space<ub>>
    %cast_22 = memref.cast %collapse_shape_20 : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_23 = memref.cast %collapse_shape_11 : memref<128xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_24 = memref.cast %collapse_shape_21 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcmp_eq_1d_int32_t(%cast_22, %cast_23, %cast_24) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    %19 = hivm.hir.pointer_cast(%c2112_i64) : memref<16x8xf32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %19 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c2624_i64) : memref<24xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    %cast_26 = memref.cast %collapse_shape_21 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_27 = memref.cast %collapse_shape_25 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %cast_28 = memref.cast %20 : memref<24xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @vcast_bool_to_float_1d_with_temp(%cast_26, %cast_27, %cast_28, %c0_i32) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    scf.for %arg9 = %3 to %1 step %c24_i32  : i32 {
      hivm.hir.set_mask_norm
      %21 = arith.remsi %arg9, %arg8 : i32
      %22 = arith.divsi %arg9, %arg8 : i32
      %23 = arith.remsi %22, %arg7 : i32
      %24 = arith.divsi %arg9, %4 : i32
      %25 = arith.remsi %24, %arg6 : i32
      %26 = arith.divsi %23, %c4_i32 : i32
      %27 = arith.remsi %23, %c4_i32 : i32
      %28 = arith.muli %26, %arg5 : i32
      %29 = arith.muli %28, %c4_i32 : i32
      %30 = arith.addi %29, %27 : i32
      %31 = arith.muli %30, %c64_i32 : i32
      %32 = arith.index_cast %31 : i32 to index
      %33 = arith.cmpi eq, %21, %c0_i32 : i32
      %34 = scf.if %33 -> (i32) {
        scf.yield %c0_i32 : i32
      } else {
        %80 = arith.cmpi eq, %21, %c1_i32 : i32
        %81 = scf.if %80 -> (i32) {
          scf.yield %c16_i32 : i32
        } else {
          %82 = arith.cmpi eq, %21, %c2_i32 : i32
          %83 = arith.select %82, %c32_i32, %c48_i32 : i32
          scf.yield %83 : i32
        }
        scf.yield %81 : i32
      }
      %35 = arith.muli %25, %c64_i32 : i32
      %36 = arith.addi %35, %34 : i32
      %37 = arith.maxsi %36, %c0_i32 : i32
      %38 = arith.index_cast %37 : i32 to index
      %39 = arith.maxsi %34, %c0_i32 : i32
      %40 = arith.index_cast %39 : i32 to index
      %41 = affine.max #map30()[%40]
      %42 = affine.min #map6()[%41]
      %43 = affine.apply #map31()[%40, %32, %38]
      %44 = affine.max #map32()[%8, %40, %38]
      %45 = affine.min #map6()[%44]
      %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %46 = hivm.hir.pointer_cast(%c2720_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %47 = arith.subi %c0_i32, %36 : i32
      %48 = arith.maxsi %47, %c0_i32 : i32
      %49 = arith.index_cast %48 : i32 to index
      %50 = affine.min #map7()[%49, %44]
      %51 = affine.apply #map8()[%45, %50]
      %52 = arith.subi %c0_i32, %34 : i32
      %53 = arith.maxsi %52, %c0_i32 : i32
      %54 = arith.index_cast %53 : i32 to index
      %55 = affine.min #map7()[%54, %41]
      %56 = affine.apply #map8()[%42, %55]
      %57 = arith.cmpi slt, %51, %c16 : index
      %58 = arith.cmpi slt, %56, %c16 : index
      %59 = arith.ori %57, %58 : i1
      %60 = affine.min #map33()[%10, %42, %55]
      %61 = affine.min #map34()[%60, %42, %55]
      %subview_29 = memref.subview %reinterpret_cast[0, %11] [16, 8] [1, 1] : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_30 = memref.subview %subview_29[0, 0] [%51, %61] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %62 = affine.max #map35()[%55, %10]
      %63 = affine.min #map36()[%41, %10]
      %64 = affine.max #map37()[%63, %55, %10]
      %65 = affine.apply #map8()[%64, %62]
      %66 = affine.apply #map38()[%62, %10]
      %subview_31 = memref.subview %46[%50, %66] [%51, %65] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %59 {
        %collapse_shape_59 = memref.collapse_shape %46 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        %cast_60 = memref.cast %collapse_shape_59 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_scalar_float_to_1d(%cst, %cast_60) : (f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      %cast_32 = memref.cast %subview_30 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_33 = memref.cast %subview_31 : memref<?x?xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_2d_float(%cast_32, %cast_33, %c2_i32, %cst, %55) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %67 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_34 = memref.collapse_shape %46 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %collapse_shape_35 = memref.collapse_shape %67 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      %68 = hivm.hir.pointer_cast(%c3744_i64) : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %cast_36 = memref.cast %collapse_shape_16 : memref<128xi1, #hivm.address_space<ub>> to memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_37 = memref.cast %collapse_shape_34 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_38 = memref.cast %collapse_shape_35 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_39 = memref.cast %68 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vsel_vs_1d_float(%cast_36, %cast_37, %cst, %cast_38, %cast_39) : (memref<?xi1, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      %69 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_40 = memref.collapse_shape %69 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %70 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_41 = memref.cast %collapse_shape_35 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_42 = memref.cast %collapse_shape_40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_43 = memref.cast %70 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vmuls_vs_1d_float(%cast_41, %cst_0, %cast_42, %cast_43) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %71 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_44 = memref.collapse_shape %71 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %72 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_45 = memref.cast %collapse_shape_40 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_46 = memref.cast %collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_47 = memref.cast %72 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadds_vs_1d_float(%cast_45, %cst, %cast_46, %cast_47) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %73 = arith.addi %34, %c2_i32 : i32
      %74 = arith.addi %34, %c16_i32 : i32
      %75 = arith.subi %arg5, %35 : i32
      %76 = arith.minsi %74, %75 : i32
      %77 = arith.index_cast %34 : i32 to index
      scf.for %arg10 = %73 to %76 step %c1_i32  : i32 {
        %80 = arith.addi %35, %arg10 : i32
        %81 = arith.muli %80, %c256_i32 : i32
        %82 = arith.index_cast %81 : i32 to index
        %83 = affine.apply #map39()[%77, %32, %82]
        %reinterpret_cast_59 = memref.reinterpret_cast %arg3 to offset: [%83], sizes: [16], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %84 = hivm.hir.pointer_cast(%c3808_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %cast_60 = memref.cast %reinterpret_cast_59 : memref<16xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_61 = memref.cast %84 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_60, %cast_61, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %85 = hivm.hir.pointer_cast(%c3808_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %86 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_62 = memref.cast %84 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_63 = memref.cast %85 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_64 = memref.cast %86 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_62, %cst_0, %cast_63, %cast_64) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %87 = hivm.hir.pointer_cast(%c3808_i64) : memref<16xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %88 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_65 = memref.cast %85 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_66 = memref.cast %87 : memref<16xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_67 = memref.cast %88 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_65, %cst, %cast_66, %cast_67) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %expand_shape_68 = memref.expand_shape %87 [[0, 1]] output_shape [16, 1] : memref<16xf32, #hivm.address_space<ub>> into memref<16x1xf32, #hivm.address_space<ub>>
        %89 = hivm.hir.pointer_cast(%c3872_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        %90 = hivm.hir.pointer_cast(%c4384_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_69 = memref.cast %expand_shape_68 : memref<16x1xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_70 = memref.cast %71 : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_71 = memref.cast %89 : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_72 = memref.cast %90 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmul_2d_float(%cast_69, %cast_70, %cast_71, %cast_72) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %91 = hivm.hir.pointer_cast(%c4896_i64) : memref<1x8xf32, #hivm.address_space<ub>>
        %92 = hivm.hir.pointer_cast(%c4928_i64) : memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_73 = memref.cast %89 : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_74 = memref.cast %91 : memref<1x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_75 = memref.cast %92 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @reduce_sum_ra_float(%cast_73, %cast_74, %cast_75, %cst) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32) -> ()
        %collapse_shape_76 = memref.collapse_shape %91 [[0, 1]] : memref<1x8xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
        %subview_77 = memref.subview %87[%11] [8] [1] {to_be_bubbled_slice} : memref<16xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
        %93 = hivm.hir.pointer_cast(%c5440_i64) : memref<8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %94 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_78 = memref.cast %subview_77 : memref<8xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_79 = memref.cast %collapse_shape_76 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_80 = memref.cast %93 : memref<8xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_81 = memref.cast %94 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadd_1d_float(%cast_78, %cast_79, %cast_80, %cast_81) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
        %95 = arith.subi %arg10, %34 : i32
        %expand_shape_82 = memref.expand_shape %93 [[0, 1]] output_shape [1, 8] : memref<8xf32, #hivm.address_space<ub>> into memref<1x8xf32, #hivm.address_space<ub>>
        %96 = hivm.hir.pointer_cast(%c5472_i64) : memref<16x8xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %cast_83 = memref.cast %expand_shape_82 : memref<1x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_84 = memref.cast %96 : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        func.call @broadcast_first_axis_align_2d_float(%cast_83, %cast_84) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %97 = arith.index_cast %95 : i32 to index
        %98 = affine.apply #map40()[%97]
        %99 = arith.cmpi slt, %97, %c0 : index
        %100 = arith.cmpi sge, %97, %c16 : index
        %101 = arith.ori %99, %100 : i1
        scf.if %101 {
        } else {
          %subview_85 = memref.subview %71[0, 0] [%97, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_86 = memref.subview %96[0, 0] [%97, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_87 = memref.collapse_shape %subview_85 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_88 = memref.collapse_shape %subview_86 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_89 = memref.cast %collapse_shape_87 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_90 = memref.cast %collapse_shape_88 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @copy_ubuf_to_ubuf_1d_float(%cast_89, %cast_90, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
          %subview_91 = memref.subview %96[0, 0] [%98, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %subview_92 = memref.subview %71[0, 0] [%98, 8] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>>
          %collapse_shape_93 = memref.collapse_shape %subview_91 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          %collapse_shape_94 = memref.collapse_shape %subview_92 [[0, 1]] : memref<?x8xf32, strided<[8, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
          hivm.hir.pipe_barrier[<PIPE_V>]
          %cast_95 = memref.cast %collapse_shape_93 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          %cast_96 = memref.cast %collapse_shape_94 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
          func.call @copy_ubuf_to_ubuf_1d_float(%cast_95, %cast_96, %c0_i32, %cst) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32) -> ()
        }
      } {pipeline.veconly}
      %78 = hivm.hir.pointer_cast(%c3232_i64) : memref<16x8xf32, #hivm.address_space<ub>>
      %collapse_shape_48 = memref.collapse_shape %78 [[0, 1]] : memref<16x8xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      %79 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
      %cast_49 = memref.cast %collapse_shape_44 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_50 = memref.cast %collapse_shape_25 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_51 = memref.cast %collapse_shape_48 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_52 = memref.cast %79 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @vadd_1d_float(%cast_49, %cast_50, %cast_51, %cast_52) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_53 = memref.reinterpret_cast %arg4 to offset: [%43], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_54 = memref.subview %reinterpret_cast_53[0, %11] [16, 8] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_55 = memref.subview %78[%50, %66] [%51, %65] [1, 1] : memref<16x8xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_56 = memref.subview %subview_54[0, 0] [%51, %61] [1, 1] : memref<16x8xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %cast_57 = memref.cast %subview_55 : memref<?x?xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
      %cast_58 = memref.cast %subview_56 : memref<?x?xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_2d_float(%cast_57, %cast_58, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10 ffts_base_addr = %arg0
      scf.if %33 {
        %80 = arith.addi %35, %c16_i32 : i32
        %81 = arith.maxsi %80, %c0_i32 : i32
        %82 = arith.index_cast %81 : i32 to index
        %83 = affine.apply #map2()[%32, %82]
        %84 = affine.max #map5()[%8, %82]
        %85 = affine.min #map6()[%84]
        %86 = arith.addi %35, %c32_i32 : i32
        %87 = arith.maxsi %86, %c0_i32 : i32
        %88 = arith.index_cast %87 : i32 to index
        %89 = affine.apply #map2()[%32, %88]
        %90 = affine.apply #map3()[%32, %88]
        %91 = affine.max #map5()[%8, %88]
        %92 = affine.min #map6()[%91]
        %93 = arith.addi %35, %c48_i32 : i32
        %94 = arith.maxsi %93, %c0_i32 : i32
        %95 = arith.index_cast %94 : i32 to index
        %96 = affine.apply #map2()[%32, %95]
        %97 = affine.apply #map3()[%32, %95]
        %98 = affine.apply #map4()[%32, %95]
        %99 = affine.max #map5()[%8, %95]
        %100 = affine.min #map6()[%99]
        %101 = arith.subi %c-16_i32, %35 : i32
        %102 = arith.maxsi %101, %c0_i32 : i32
        %103 = arith.index_cast %102 : i32 to index
        %104 = affine.min #map7()[%103, %84]
        %105 = arith.subi %c-32_i32, %35 : i32
        %106 = arith.maxsi %105, %c0_i32 : i32
        %107 = arith.index_cast %106 : i32 to index
        %108 = affine.min #map7()[%107, %91]
        %109 = arith.subi %c-48_i32, %35 : i32
        %110 = arith.maxsi %109, %c0_i32 : i32
        %111 = arith.index_cast %110 : i32 to index
        %112 = affine.min #map7()[%111, %99]
        %113 = hivm.hir.get_block_idx -> i64
        %114 = arith.index_cast %113 : i64 to index
        %115 = affine.apply #map14(%114)
        %view = memref.view %arg2[%115][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
        %subview_59 = memref.subview %view[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        %116 = hivm.hir.pointer_cast(%c5984_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_60 = memref.collapse_shape %subview_59 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_61 = memref.collapse_shape %116 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
        %cast_62 = memref.cast %collapse_shape_60 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_63 = memref.cast %collapse_shape_61 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_62, %cast_63, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 9 ffts_base_addr = %arg0
        %117 = hivm.hir.pointer_cast(%c5984_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_64 = memref.collapse_shape %117 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %118 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_65 = memref.cast %collapse_shape_61 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_66 = memref.cast %collapse_shape_64 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_67 = memref.cast %118 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_65, %cst_0, %cast_66, %cast_67) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %119 = hivm.hir.pointer_cast(%c5984_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_68 = memref.collapse_shape %119 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %120 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_69 = memref.cast %collapse_shape_64 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_70 = memref.cast %collapse_shape_68 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_71 = memref.cast %120 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_69, %cst, %cast_70, %cast_71) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %121 = hivm.hir.get_block_idx -> i64
        %122 = arith.index_cast %121 : i64 to index
        %123 = affine.apply #map15(%122)
        %view_72 = memref.view %arg2[%123][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        %subview_73 = memref.subview %view_72[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 8
        %collapse_shape_74 = memref.collapse_shape %subview_73 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_75 = memref.cast %collapse_shape_68 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_76 = memref.cast %collapse_shape_74 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_75, %cast_76, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10 ffts_base_addr = %arg0
        %124 = hivm.hir.get_block_idx -> i64
        %125 = arith.index_cast %124 : i64 to index
        %126 = affine.apply #map16(%125)
        %view_77 = memref.view %arg2[%126][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        %subview_78 = memref.subview %view_77[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
        %collapse_shape_79 = memref.collapse_shape %subview_78 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %cast_80 = memref.cast %collapse_shape_68 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_81 = memref.cast %collapse_shape_79 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_80, %cast_81, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10 ffts_base_addr = %arg0
        %127 = hivm.hir.get_block_idx -> i64
        %128 = arith.index_cast %127 : i64 to index
        %129 = affine.apply #map18(%128)
        %view_82 = memref.view %arg2[%129][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
        %subview_83 = memref.subview %view_82[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        %130 = hivm.hir.pointer_cast(%c6496_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_84 = memref.collapse_shape %subview_83 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_85 = memref.collapse_shape %130 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
        %cast_86 = memref.cast %collapse_shape_84 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_87 = memref.cast %collapse_shape_85 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_86, %cast_87, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6 ffts_base_addr = %arg0
        %131 = hivm.hir.pointer_cast(%c6496_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_88 = memref.collapse_shape %131 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %132 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_89 = memref.cast %collapse_shape_85 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_90 = memref.cast %collapse_shape_88 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_91 = memref.cast %132 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_89, %cst_0, %cast_90, %cast_91) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %133 = hivm.hir.pointer_cast(%c6496_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_92 = memref.collapse_shape %133 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %134 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_93 = memref.cast %collapse_shape_88 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_94 = memref.cast %collapse_shape_92 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_95 = memref.cast %134 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_93, %cst, %cast_94, %cast_95) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %135 = hivm.hir.get_block_idx -> i64
        %136 = arith.index_cast %135 : i64 to index
        %137 = affine.apply #map19(%136)
        %view_96 = memref.view %arg2[%137][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        %subview_97 = memref.subview %view_96[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
        %collapse_shape_98 = memref.collapse_shape %subview_97 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_99 = memref.cast %collapse_shape_92 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_100 = memref.cast %collapse_shape_98 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_99, %cast_100, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10 ffts_base_addr = %arg0
        %138 = hivm.hir.get_block_idx -> i64
        %139 = arith.index_cast %138 : i64 to index
        %140 = affine.apply #map21(%139)
        %view_101 = memref.view %arg2[%140][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
        %subview_102 = memref.subview %view_101[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        %141 = hivm.hir.pointer_cast(%c7008_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_103 = memref.collapse_shape %subview_102 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_104 = memref.collapse_shape %141 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        %cast_105 = memref.cast %collapse_shape_103 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_106 = memref.cast %collapse_shape_104 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_105, %cast_106, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4 ffts_base_addr = %arg0
        %142 = hivm.hir.pointer_cast(%c7008_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_107 = memref.collapse_shape %142 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %143 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_108 = memref.cast %collapse_shape_104 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_109 = memref.cast %collapse_shape_107 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_110 = memref.cast %143 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_108, %cst_0, %cast_109, %cast_110) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %144 = hivm.hir.pointer_cast(%c7008_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_111 = memref.collapse_shape %144 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %145 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_112 = memref.cast %collapse_shape_107 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_113 = memref.cast %collapse_shape_111 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_114 = memref.cast %145 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_112, %cst, %cast_113, %cast_114) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %146 = hivm.hir.get_block_idx -> i64
        %147 = arith.index_cast %146 : i64 to index
        %148 = affine.apply #map23(%147)
        %view_115 = memref.view %arg2[%148][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
        %subview_116 = memref.subview %view_115[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        %149 = hivm.hir.pointer_cast(%c7520_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_117 = memref.collapse_shape %subview_116 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_118 = memref.collapse_shape %149 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        %cast_119 = memref.cast %collapse_shape_117 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_120 = memref.cast %collapse_shape_118 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_119, %cast_120, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3 ffts_base_addr = %arg0
        %150 = hivm.hir.pointer_cast(%c7520_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_121 = memref.collapse_shape %150 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %151 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_122 = memref.cast %collapse_shape_118 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_123 = memref.cast %collapse_shape_121 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_124 = memref.cast %151 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_122, %cst_0, %cast_123, %cast_124) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %152 = hivm.hir.pointer_cast(%c7520_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_125 = memref.collapse_shape %152 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %153 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_126 = memref.cast %collapse_shape_121 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_127 = memref.cast %collapse_shape_125 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_128 = memref.cast %153 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_126, %cst, %cast_127, %cast_128) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %154 = hivm.hir.get_block_idx -> i64
        %155 = arith.index_cast %154 : i64 to index
        %156 = affine.apply #map24(%155)
        %view_129 = memref.view %arg2[%156][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        %subview_130 = memref.subview %view_129[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
        %collapse_shape_131 = memref.collapse_shape %subview_130 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %cast_132 = memref.cast %collapse_shape_125 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_133 = memref.cast %collapse_shape_131 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_1d_float(%cast_132, %cast_133, %c0_i32) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 10 ffts_base_addr = %arg0
        %157 = hivm.hir.get_block_idx -> i64
        %158 = arith.index_cast %157 : i64 to index
        %159 = affine.apply #map26(%158)
        %view_134 = memref.view %arg2[%159][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
        %subview_135 = memref.subview %view_134[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        %160 = hivm.hir.pointer_cast(%c8032_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_136 = memref.collapse_shape %subview_135 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_137 = memref.collapse_shape %160 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        %cast_138 = memref.cast %collapse_shape_136 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_139 = memref.cast %collapse_shape_137 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_138, %cast_139, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1 ffts_base_addr = %arg0
        %161 = hivm.hir.pointer_cast(%c8032_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_140 = memref.collapse_shape %161 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %162 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_141 = memref.cast %collapse_shape_137 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_142 = memref.cast %collapse_shape_140 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_143 = memref.cast %162 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_141, %cst_0, %cast_142, %cast_143) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %163 = hivm.hir.pointer_cast(%c8032_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_144 = memref.collapse_shape %163 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %164 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_145 = memref.cast %collapse_shape_140 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_146 = memref.cast %collapse_shape_144 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_147 = memref.cast %164 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_145, %cst, %cast_146, %cast_147) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %165 = hivm.hir.get_block_idx -> i64
        %166 = arith.index_cast %165 : i64 to index
        %167 = affine.apply #map28(%166)
        %view_148 = memref.view %arg2[%167][] : memref<?xi8, #hivm.address_space<gm>> to memref<16x16xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 12
        %subview_149 = memref.subview %view_148[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, #hivm.address_space<gm>> to memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>>
        %168 = hivm.hir.pointer_cast(%c8544_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_150 = memref.collapse_shape %subview_149 [[0, 1]] : memref<8x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<gm>> into memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %collapse_shape_151 = memref.collapse_shape %168 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        %cast_152 = memref.cast %collapse_shape_150 : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
        %cast_153 = memref.cast %collapse_shape_151 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @load_gm_to_ubuf_1d_float(%cast_152, %cast_153, %c0_i32, %cst, %c0) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, f32, index) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0 ffts_base_addr = %arg0
        %169 = hivm.hir.pointer_cast(%c8544_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_154 = memref.collapse_shape %169 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %170 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_155 = memref.cast %collapse_shape_151 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_156 = memref.cast %collapse_shape_154 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_157 = memref.cast %170 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vmuls_vs_1d_float(%cast_155, %cst_0, %cast_156, %cast_157) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        %171 = hivm.hir.pointer_cast(%c8544_i64) : memref<8x16xf32, #hivm.address_space<ub>>
        %collapse_shape_158 = memref.collapse_shape %171 [[0, 1]] : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        %172 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>
        %cast_159 = memref.cast %collapse_shape_154 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_160 = memref.cast %collapse_shape_158 : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        %cast_161 = memref.cast %172 : memref<0xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
        func.call @vadds_vs_1d_float(%cast_159, %cst, %cast_160, %cast_161) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, f32, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %reinterpret_cast_162 = memref.reinterpret_cast %arg4 to offset: [%83], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_163 = memref.subview %reinterpret_cast_162[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %reinterpret_cast_164 = memref.reinterpret_cast %arg4 to offset: [%89], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_165 = memref.subview %reinterpret_cast_164[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %reinterpret_cast_166 = memref.reinterpret_cast %arg4 to offset: [%90], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_167 = memref.subview %reinterpret_cast_166[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %reinterpret_cast_168 = memref.reinterpret_cast %arg4 to offset: [%96], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_169 = memref.subview %reinterpret_cast_168[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %reinterpret_cast_170 = memref.reinterpret_cast %arg4 to offset: [%97], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_171 = memref.subview %reinterpret_cast_170[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %reinterpret_cast_172 = memref.reinterpret_cast %arg4 to offset: [%98], sizes: [16, 16], strides: [256, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_173 = memref.subview %reinterpret_cast_172[%11, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : memref<16x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        %173 = affine.max #map35()[%104, %10]
        %174 = affine.min #map36()[%84, %10]
        %175 = affine.max #map37()[%174, %104, %10]
        %176 = affine.apply #map8()[%175, %173]
        %177 = affine.apply #map38()[%173, %10]
        %subview_174 = memref.subview %119[%177, 0] [%176, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
        %178 = affine.min #map33()[%10, %85, %104]
        %179 = affine.min #map34()[%178, %85, %104]
        %subview_175 = memref.subview %subview_163[0, 0] [%179, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 11
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_176 = memref.cast %subview_174 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_177 = memref.cast %subview_175 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_176, %cast_177, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
        %180 = affine.max #map35()[%108, %10]
        %181 = affine.min #map36()[%91, %10]
        %182 = affine.max #map37()[%181, %108, %10]
        %183 = affine.apply #map8()[%182, %180]
        %184 = affine.apply #map38()[%180, %10]
        %subview_178 = memref.subview %152[%184, 0] [%183, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
        %185 = affine.min #map33()[%10, %92, %108]
        %186 = affine.min #map34()[%185, %92, %108]
        %subview_179 = memref.subview %subview_165[0, 0] [%186, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_180 = memref.cast %subview_178 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_181 = memref.cast %subview_179 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_180, %cast_181, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        %subview_182 = memref.subview %133[%184, 0] [%183, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_183 = memref.subview %subview_167[0, 0] [%186, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_184 = memref.cast %subview_182 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_185 = memref.cast %subview_183 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_184, %cast_185, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
        %187 = affine.max #map35()[%112, %10]
        %188 = affine.min #map36()[%99, %10]
        %189 = affine.max #map37()[%188, %112, %10]
        %190 = affine.apply #map8()[%189, %187]
        %191 = affine.apply #map38()[%187, %10]
        %subview_186 = memref.subview %171[%191, 0] [%190, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
        %192 = affine.min #map33()[%10, %100, %112]
        %193 = affine.min #map34()[%192, %100, %112]
        %subview_187 = memref.subview %subview_169[0, 0] [%193, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_188 = memref.cast %subview_186 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_189 = memref.cast %subview_187 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_188, %cast_189, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        %subview_190 = memref.subview %163[%191, 0] [%190, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_191 = memref.subview %subview_171[0, 0] [%193, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_192 = memref.cast %subview_190 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_193 = memref.cast %subview_191 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_192, %cast_193, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        %subview_194 = memref.subview %144[%191, 0] [%190, 16] [1, 1] : memref<8x16xf32, #hivm.address_space<ub>> to memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_195 = memref.subview %subview_173[0, 0] [%193, 16] [1, 1] : memref<8x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_MTE3>]
        %cast_196 = memref.cast %subview_194 : memref<?x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
        %cast_197 = memref.cast %subview_195 : memref<?x16xf32, strided<[256, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
        func.call @store_ubuf_to_gm_2d_float(%cast_196, %cast_197, %c0_i32) : (memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>, memref<?x?xf32, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      }
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 8
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
