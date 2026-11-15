// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @_rms_norm_kernel_full_row(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: f32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, false, false, false, false, false, false, false, false]> : vector<14xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c92928_i64 = arith.constant 92928 : i64
  %c58112_i64 = arith.constant 58112 : i64
  %c58080_i64 = arith.constant 58080 : i64
  %c58048_i64 = arith.constant 58048 : i64
  %c57024_i64 = arith.constant 57024 : i64
  %c56992_i64 = arith.constant 56992 : i64
  %c54944_i64 = arith.constant 54944 : i64
  %c90880_i64 = arith.constant 90880 : i64
  %c52896_i64 = arith.constant 52896 : i64
  %c88832_i64 = arith.constant 88832 : i64
  %c50848_i64 = arith.constant 50848 : i64
  %c50816_i64 = arith.constant 50816 : i64
  %c50784_i64 = arith.constant 50784 : i64
  %c49760_i64 = arith.constant 49760 : i64
  %c49728_i64 = arith.constant 49728 : i64
  %c47680_i64 = arith.constant 47680 : i64
  %c86784_i64 = arith.constant 86784 : i64
  %c45632_i64 = arith.constant 45632 : i64
  %c84736_i64 = arith.constant 84736 : i64
  %c43584_i64 = arith.constant 43584 : i64
  %c43552_i64 = arith.constant 43552 : i64
  %c43520_i64 = arith.constant 43520 : i64
  %c42496_i64 = arith.constant 42496 : i64
  %c42464_i64 = arith.constant 42464 : i64
  %c40416_i64 = arith.constant 40416 : i64
  %c82688_i64 = arith.constant 82688 : i64
  %c38368_i64 = arith.constant 38368 : i64
  %c80640_i64 = arith.constant 80640 : i64
  %c36320_i64 = arith.constant 36320 : i64
  %c36288_i64 = arith.constant 36288 : i64
  %c36256_i64 = arith.constant 36256 : i64
  %c35232_i64 = arith.constant 35232 : i64
  %c35200_i64 = arith.constant 35200 : i64
  %c33152_i64 = arith.constant 33152 : i64
  %c78592_i64 = arith.constant 78592 : i64
  %c31104_i64 = arith.constant 31104 : i64
  %c76544_i64 = arith.constant 76544 : i64
  %c29056_i64 = arith.constant 29056 : i64
  %c29024_i64 = arith.constant 29024 : i64
  %c28992_i64 = arith.constant 28992 : i64
  %c27968_i64 = arith.constant 27968 : i64
  %c27936_i64 = arith.constant 27936 : i64
  %c25888_i64 = arith.constant 25888 : i64
  %c74496_i64 = arith.constant 74496 : i64
  %c23840_i64 = arith.constant 23840 : i64
  %c72448_i64 = arith.constant 72448 : i64
  %c21792_i64 = arith.constant 21792 : i64
  %c21760_i64 = arith.constant 21760 : i64
  %c21728_i64 = arith.constant 21728 : i64
  %c20704_i64 = arith.constant 20704 : i64
  %c20672_i64 = arith.constant 20672 : i64
  %c18624_i64 = arith.constant 18624 : i64
  %c70400_i64 = arith.constant 70400 : i64
  %c16576_i64 = arith.constant 16576 : i64
  %c68352_i64 = arith.constant 68352 : i64
  %c14528_i64 = arith.constant 14528 : i64
  %c14496_i64 = arith.constant 14496 : i64
  %c14464_i64 = arith.constant 14464 : i64
  %c13440_i64 = arith.constant 13440 : i64
  %c13408_i64 = arith.constant 13408 : i64
  %c11360_i64 = arith.constant 11360 : i64
  %c66304_i64 = arith.constant 66304 : i64
  %c9312_i64 = arith.constant 9312 : i64
  %c64256_i64 = arith.constant 64256 : i64
  %c7264_i64 = arith.constant 7264 : i64
  %c7232_i64 = arith.constant 7232 : i64
  %c7200_i64 = arith.constant 7200 : i64
  %c6176_i64 = arith.constant 6176 : i64
  %c6144_i64 = arith.constant 6144 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c62208_i64 = arith.constant 62208 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c60160_i64 = arith.constant 60160 : i64
  %c0_i64 = arith.constant 0 : i64
  %c7_i32 = arith.constant 7 : i32
  %c6_i32 = arith.constant 6 : i32
  %c5_i32 = arith.constant 5 : i32
  %c4_i32 = arith.constant 4 : i32
  %c3_i32 = arith.constant 3 : i32
  %c2_i32 = arith.constant 2 : i32
  %c8_i32 = arith.constant 8 : i32
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c48_i32 = arith.constant 48 : i32
  %0 = arith.muli %arg11, %arg12 : i32
  %1 = arith.muli %0, %arg13 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg13, %arg12 : i32
  %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1]>, #hivm.address_space<gm>>
  %5 = arith.index_cast %arg9 : i32 to index
  %6 = affine.max affine_map<()[s0] -> (0, s0)>()[%5]
  %7 = affine.min affine_map<()[s0] -> (512, s0)>()[%6]
  %8 = arith.cmpi slt, %7, %c512 : index
  %subview = memref.subview %reinterpret_cast[0] [%7] [1] : memref<512xf32, strided<[1]>, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<gm>>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID6>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID7>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  scf.for %arg14 = %3 to %1 step %c48_i32  : i32 {
    %9 = arith.index_cast %arg14 : i32 to index
    %10 = arith.index_cast %3 : i32 to index
    %11 = arith.index_cast %1 : i32 to index
    %12 = arith.index_cast %c48_i32 : i32 to index
    %13 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%9, %10, %12]
    %14 = arith.index_cast %13 : index to i1
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %15 = arith.select %14, %c1_i64, %c2_i64 : i64
    %16 = hivm.hir.pointer_cast(%c58112_i64, %c92928_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c52896_i64, %c90880_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c50848_i64, %c88832_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c45632_i64, %c86784_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c43584_i64, %c84736_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c38368_i64, %c82688_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c36320_i64, %c80640_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c31104_i64, %c78592_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c29056_i64, %c76544_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c23840_i64, %c74496_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c21792_i64, %c72448_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c16576_i64, %c70400_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c14528_i64, %c68352_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c9312_i64, %c66304_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c7264_i64, %c64256_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c2048_i64, %c62208_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %31 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c0_i64, %c60160_i64) : memref<512xf32, #hivm.address_space<ub>>
    annotation.mark %32 {hivm.multi_buffer = 2 : i32} : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.set_mask_norm
    %33 = arith.divsi %arg14, %4 : i32
    %34 = arith.remsi %33, %arg11 : i32
    %35 = arith.muli %34, %c8_i32 : i32
    %subview_1 = memref.subview %32[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %8 {
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%cst : f32) outs(%32 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, %15]
    hivm.hir.pipe_barrier[<PIPE_MTE2>]
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1]>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %36 = arith.cmpi slt, %35, %arg8 : i32
    scf.if %36 {
      %51 = arith.muli %35, %arg6 : i32
      %52 = arith.index_cast %51 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %reinterpret_cast_2[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %31[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %8 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%31 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %53 = hivm.hir.pointer_cast(%c4096_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%31, %31 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%53 : memref<512xf32, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c6144_i64) : memref<1xf32, #hivm.address_space<ub>>
      %55 = hivm.hir.pointer_cast(%c6176_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%53 : memref<512xf32, #hivm.address_space<ub>>) outs(%54 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %56 = memref.load %54[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %57 = arith.sitofp %arg9 : i32 to f32
      %58 = arith.divf %56, %57 : f32
      %59 = arith.addf %58, %arg10 : f32
      %60 = hivm.hir.pointer_cast(%c7200_i64) : memref<1xf32, #hivm.address_space<ub>>
      %61 = math.sqrt %59 : f32
      memref.store %61, %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %62 = hivm.hir.pointer_cast(%c7232_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %cst, %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vdiv ins(%62, %60 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%62 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %63 = memref.load %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%31, %63 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%31 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID7>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%31, %32 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%30 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %64 = arith.muli %35, %arg7 : i32
      %65 = arith.index_cast %64 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %30[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %reinterpret_cast_5[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID7>]
    } {pipeline.veconly}
    %37 = arith.addi %35, %c1_i32 : i32
    %38 = arith.cmpi slt, %37, %arg8 : i32
    scf.if %38 {
      %51 = arith.muli %37, %arg6 : i32
      %52 = arith.index_cast %51 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %reinterpret_cast_2[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %29[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %8 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%29 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %53 = hivm.hir.pointer_cast(%c11360_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%29, %29 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%53 : memref<512xf32, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c13408_i64) : memref<1xf32, #hivm.address_space<ub>>
      %55 = hivm.hir.pointer_cast(%c13440_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%53 : memref<512xf32, #hivm.address_space<ub>>) outs(%54 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %56 = memref.load %54[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %57 = arith.sitofp %arg9 : i32 to f32
      %58 = arith.divf %56, %57 : f32
      %59 = arith.addf %58, %arg10 : f32
      %60 = hivm.hir.pointer_cast(%c14464_i64) : memref<1xf32, #hivm.address_space<ub>>
      %61 = math.sqrt %59 : f32
      memref.store %61, %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %62 = hivm.hir.pointer_cast(%c14496_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %cst, %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vdiv ins(%62, %60 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%62 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %63 = memref.load %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%29, %63 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%29 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID6>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%29, %32 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%28 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %64 = arith.muli %37, %arg7 : i32
      %65 = arith.index_cast %64 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %28[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %reinterpret_cast_5[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID6>]
    } {pipeline.veconly}
    %39 = arith.addi %35, %c2_i32 : i32
    %40 = arith.cmpi slt, %39, %arg8 : i32
    scf.if %40 {
      %51 = arith.muli %39, %arg6 : i32
      %52 = arith.index_cast %51 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %reinterpret_cast_2[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %27[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %8 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%27 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %53 = hivm.hir.pointer_cast(%c18624_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%27, %27 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%53 : memref<512xf32, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c20672_i64) : memref<1xf32, #hivm.address_space<ub>>
      %55 = hivm.hir.pointer_cast(%c20704_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%53 : memref<512xf32, #hivm.address_space<ub>>) outs(%54 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %56 = memref.load %54[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %57 = arith.sitofp %arg9 : i32 to f32
      %58 = arith.divf %56, %57 : f32
      %59 = arith.addf %58, %arg10 : f32
      %60 = hivm.hir.pointer_cast(%c21728_i64) : memref<1xf32, #hivm.address_space<ub>>
      %61 = math.sqrt %59 : f32
      memref.store %61, %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %62 = hivm.hir.pointer_cast(%c21760_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %cst, %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vdiv ins(%62, %60 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%62 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %63 = memref.load %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%27, %63 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%27 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID5>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%27, %32 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%26 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %64 = arith.muli %39, %arg7 : i32
      %65 = arith.index_cast %64 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %26[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %reinterpret_cast_5[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID5>]
    } {pipeline.veconly}
    %41 = arith.addi %35, %c3_i32 : i32
    %42 = arith.cmpi slt, %41, %arg8 : i32
    scf.if %42 {
      %51 = arith.muli %41, %arg6 : i32
      %52 = arith.index_cast %51 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %reinterpret_cast_2[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %25[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %8 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%25 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %53 = hivm.hir.pointer_cast(%c25888_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%25, %25 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%53 : memref<512xf32, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c27936_i64) : memref<1xf32, #hivm.address_space<ub>>
      %55 = hivm.hir.pointer_cast(%c27968_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%53 : memref<512xf32, #hivm.address_space<ub>>) outs(%54 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %56 = memref.load %54[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %57 = arith.sitofp %arg9 : i32 to f32
      %58 = arith.divf %56, %57 : f32
      %59 = arith.addf %58, %arg10 : f32
      %60 = hivm.hir.pointer_cast(%c28992_i64) : memref<1xf32, #hivm.address_space<ub>>
      %61 = math.sqrt %59 : f32
      memref.store %61, %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %62 = hivm.hir.pointer_cast(%c29024_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %cst, %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vdiv ins(%62, %60 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%62 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %63 = memref.load %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%25, %63 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%25 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%25, %32 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%24 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %64 = arith.muli %41, %arg7 : i32
      %65 = arith.index_cast %64 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %24[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %reinterpret_cast_5[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
    } {pipeline.veconly}
    %43 = arith.addi %35, %c4_i32 : i32
    %44 = arith.cmpi slt, %43, %arg8 : i32
    scf.if %44 {
      %51 = arith.muli %43, %arg6 : i32
      %52 = arith.index_cast %51 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %reinterpret_cast_2[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %23[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %8 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%23 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %53 = hivm.hir.pointer_cast(%c33152_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%23, %23 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%53 : memref<512xf32, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c35200_i64) : memref<1xf32, #hivm.address_space<ub>>
      %55 = hivm.hir.pointer_cast(%c35232_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%53 : memref<512xf32, #hivm.address_space<ub>>) outs(%54 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %56 = memref.load %54[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %57 = arith.sitofp %arg9 : i32 to f32
      %58 = arith.divf %56, %57 : f32
      %59 = arith.addf %58, %arg10 : f32
      %60 = hivm.hir.pointer_cast(%c36256_i64) : memref<1xf32, #hivm.address_space<ub>>
      %61 = math.sqrt %59 : f32
      memref.store %61, %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %62 = hivm.hir.pointer_cast(%c36288_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %cst, %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vdiv ins(%62, %60 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%62 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %63 = memref.load %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%23, %63 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%23 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%23, %32 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%22 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %64 = arith.muli %43, %arg7 : i32
      %65 = arith.index_cast %64 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %22[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %reinterpret_cast_5[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    } {pipeline.veconly}
    %45 = arith.addi %35, %c5_i32 : i32
    %46 = arith.cmpi slt, %45, %arg8 : i32
    scf.if %46 {
      %51 = arith.muli %45, %arg6 : i32
      %52 = arith.index_cast %51 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %reinterpret_cast_2[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %21[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %8 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%21 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %53 = hivm.hir.pointer_cast(%c40416_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%21, %21 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%53 : memref<512xf32, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c42464_i64) : memref<1xf32, #hivm.address_space<ub>>
      %55 = hivm.hir.pointer_cast(%c42496_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%53 : memref<512xf32, #hivm.address_space<ub>>) outs(%54 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %56 = memref.load %54[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %57 = arith.sitofp %arg9 : i32 to f32
      %58 = arith.divf %56, %57 : f32
      %59 = arith.addf %58, %arg10 : f32
      %60 = hivm.hir.pointer_cast(%c43520_i64) : memref<1xf32, #hivm.address_space<ub>>
      %61 = math.sqrt %59 : f32
      memref.store %61, %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %62 = hivm.hir.pointer_cast(%c43552_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %cst, %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vdiv ins(%62, %60 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%62 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %63 = memref.load %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%21, %63 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%21 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%21, %32 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%20 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %64 = arith.muli %45, %arg7 : i32
      %65 = arith.index_cast %64 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %20[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %reinterpret_cast_5[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    } {pipeline.veconly}
    %47 = arith.addi %35, %c6_i32 : i32
    %48 = arith.cmpi slt, %47, %arg8 : i32
    scf.if %48 {
      %51 = arith.muli %47, %arg6 : i32
      %52 = arith.index_cast %51 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %reinterpret_cast_2[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %19[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %8 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%19 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %53 = hivm.hir.pointer_cast(%c47680_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%19, %19 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%53 : memref<512xf32, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c49728_i64) : memref<1xf32, #hivm.address_space<ub>>
      %55 = hivm.hir.pointer_cast(%c49760_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%53 : memref<512xf32, #hivm.address_space<ub>>) outs(%54 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %56 = memref.load %54[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %57 = arith.sitofp %arg9 : i32 to f32
      %58 = arith.divf %56, %57 : f32
      %59 = arith.addf %58, %arg10 : f32
      %60 = hivm.hir.pointer_cast(%c50784_i64) : memref<1xf32, #hivm.address_space<ub>>
      %61 = math.sqrt %59 : f32
      memref.store %61, %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %62 = hivm.hir.pointer_cast(%c50816_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %cst, %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vdiv ins(%62, %60 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%62 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %63 = memref.load %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%19, %63 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%19 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%19, %32 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%18 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %64 = arith.muli %47, %arg7 : i32
      %65 = arith.index_cast %64 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %18[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %reinterpret_cast_5[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    } {pipeline.veconly}
    %49 = arith.addi %35, %c7_i32 : i32
    %50 = arith.cmpi slt, %49, %arg8 : i32
    scf.if %50 {
      %51 = arith.muli %49, %arg6 : i32
      %52 = arith.index_cast %51 : i32 to index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%52], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_3 = memref.subview %reinterpret_cast_2[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_4 = memref.subview %17[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      scf.if %8 {
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vbrc ins(%cst_0 : f32) outs(%17 : memref<512xf32, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.load ins(%subview_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_4 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %53 = hivm.hir.pointer_cast(%c54944_i64) : memref<512xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%17, %17 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%53 : memref<512xf32, #hivm.address_space<ub>>)
      %54 = hivm.hir.pointer_cast(%c56992_i64) : memref<1xf32, #hivm.address_space<ub>>
      %55 = hivm.hir.pointer_cast(%c57024_i64) : memref<256xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%53 : memref<512xf32, #hivm.address_space<ub>>) outs(%54 : memref<1xf32, #hivm.address_space<ub>>) temp_buffer(%55 : memref<256xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %56 = memref.load %54[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %57 = arith.sitofp %arg9 : i32 to f32
      %58 = arith.divf %56, %57 : f32
      %59 = arith.addf %58, %arg10 : f32
      %60 = hivm.hir.pointer_cast(%c58048_i64) : memref<1xf32, #hivm.address_space<ub>>
      %61 = math.sqrt %59 : f32
      memref.store %61, %60[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %62 = hivm.hir.pointer_cast(%c58080_i64) : memref<1xf32, #hivm.address_space<ub>>
      memref.store %cst, %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vdiv ins(%62, %60 : memref<1xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>) outs(%62 : memref<1xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %63 = memref.load %62[%c0] : memref<1xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%17, %63 : memref<512xf32, #hivm.address_space<ub>>, f32) outs(%17 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%17, %32 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%16 : memref<512xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %64 = arith.muli %49, %arg7 : i32
      %65 = arith.index_cast %64 : i32 to index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%65], sizes: [512], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %subview_6 = memref.subview %16[0] [%7] [1] : memref<512xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %reinterpret_cast_5[0] [%7] [1] : memref<512xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.store ins(%subview_6 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_7 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    } {pipeline.veconly}
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, %15]
  } {autoblockify.subloop}
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID7>]
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
