"builtin.module"() ({
  "func.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}, {tt.max_divisibility = 8 : i32}, {tt.max_divisibility = 8 : i32}, {tt.max_divisibility = 8 : i32}, {tt.max_divisibility = 8 : i32}, {}, {}, {}, {}, {}, {}], function_type = (memref<*xf32>, memref<*xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (), sym_name = "wrap_side_by_side_masked_loop_0d1d2e3e4e5c6e7c"}> ({
  ^bb0(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32):
    %0 = "arith.constant"() <{value = -9.900000e+01 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 0 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1 : index}> : () -> index
    %3 = "arith.constant"() <{value = 4 : index}> : () -> index
    %4 = "arith.constant"() <{value = 6 : index}> : () -> index
    %5 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %8 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %9 = "arith.constant"() <{value = 2 : index}> : () -> index
    %10 = "arith.index_cast"(%arg4) : (i32) -> index
    %11 = "arith.muli"(%10, %9) : (index, index) -> index
    %12 = "arith.index_cast"(%arg3) : (i32) -> index
    %13 = "arith.addi"(%11, %4) : (index, index) -> index
    %14 = "arith.remsi"(%13, %12) : (index, index) -> index
    %15 = "arith.subi"(%13, %14) : (index, index) -> index
    %16 = "arith.addi"(%14, %3) : (index, index) -> index
    %17 = "arith.minsi"(%16, %12) : (index, index) -> index
    %18 = "arith.subi"(%17, %14) : (index, index) -> index
    %19 = "memref.reinterpret_cast"(%arg0, %13, %3, %18, %10, %2) <{operandSegmentSizes = array<i32: 1, 1, 2, 2>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<*xf32>, index, index, index, index, index) -> memref<4x?xf32, strided<[?, 1], offset: ?>>
    %20 = "arith.subi"(%3, %18) : (index, index) -> index
    %21 = "memref.reinterpret_cast"(%arg0, %15, %3, %20, %10, %2) <{operandSegmentSizes = array<i32: 1, 1, 2, 2>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<*xf32>, index, index, index, index, index) -> memref<4x?xf32, strided<[?, 1], offset: ?>>
    %22 = "arith.index_cast"(%arg5) : (i32) -> index
    %23 = "arith.muli"(%arg4, %8) : (i32, i32) -> i32
    %24 = "arith.index_cast"(%arg4) : (i32) -> index
    %25 = "arith.muli"(%24, %9) : (index, index) -> index
    %26 = "arith.index_cast"(%arg3) : (i32) -> index
    %27 = "memref.reinterpret_cast"(%arg1, %1, %22, %2) <{operandSegmentSizes = array<i32: 1, 1, 0, 2>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 4, 4>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<*xf32>, index, index, index) -> memref<4x4xf32, strided<[?, ?], offset: ?>>
    %28:6 = "scf.for"(%5, %7, %6, %19, %27, %25, %1, %1, %21) ({
    ^bb0(%arg12: i32, %arg13: memref<4x?xf32, strided<[?, 1], offset: ?>>, %arg14: memref<4x4xf32, strided<[?, ?], offset: ?>>, %arg15: index, %arg16: index, %arg17: index, %arg18: memref<4x?xf32, strided<[?, 1], offset: ?>>):
      %29 = "memref.reinterpret_cast"(%arg13, %1, %3, %18, %10, %2) <{operandSegmentSizes = array<i32: 1, 1, 2, 2>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<4x?xf32, strided<[?, 1], offset: ?>>, index, index, index, index, index) -> memref<4x?xf32, strided<[?, 1], offset: ?>>
      %30 = "memref.reinterpret_cast"(%arg18, %1, %3, %20, %10, %2) <{operandSegmentSizes = array<i32: 1, 1, 2, 2>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<4x?xf32, strided<[?, 1], offset: ?>>, index, index, index, index, index) -> memref<4x?xf32, strided<[?, 1], offset: ?>>
      %31 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x4xf32>
      "linalg.fill"(%0, %31) <{operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%arg19: f32, %arg20: f32):
        "linalg.yield"(%arg19) : (f32) -> ()
      }) : (f32, memref<4x4xf32>) -> ()
      %32 = "arith.minsi"(%18, %3) : (index, index) -> index
      %33 = "arith.subi"(%3, %32) : (index, index) -> index
      %34 = "memref.subview"(%29, %32, %10) <{operandSegmentSizes = array<i32: 1, 0, 1, 1>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 2, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, 1>}> : (memref<4x?xf32, strided<[?, 1], offset: ?>>, index, index) -> memref<2x?xf32, strided<[?, 1], offset: ?>>
      %35 = "memref.subview"(%30, %33, %10) <{operandSegmentSizes = array<i32: 1, 0, 1, 1>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 2, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, 1>}> : (memref<4x?xf32, strided<[?, 1], offset: ?>>, index, index) -> memref<2x?xf32, strided<[?, 1], offset: ?>>
      %36 = "memref.subview"(%31, %32, %10) <{operandSegmentSizes = array<i32: 1, 0, 1, 1>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 2, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, 1>}> : (memref<4x4xf32>, index, index) -> memref<2x?xf32, strided<[?, 1]>>
      %37 = "memref.subview"(%31, %32, %33, %10) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>, static_offsets = array<i64: 0, -9223372036854775808>, static_sizes = array<i64: 2, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, 1>}> : (memref<4x4xf32>, index, index, index) -> memref<2x?xf32, strided<[?, 1], offset: ?>>
      "memref.copy"(%34, %36) : (memref<2x?xf32, strided<[?, 1], offset: ?>>, memref<2x?xf32, strided<[?, 1]>>) -> ()
      "memref.copy"(%35, %37) : (memref<2x?xf32, strided<[?, 1], offset: ?>>, memref<2x?xf32, strided<[?, 1], offset: ?>>) -> ()
      %38 = "bufferization.to_tensor"(%31) <{restrict, writable}> : (memref<4x4xf32>) -> tensor<4x4xf32>
      "memref.tensor_store"(%38, %arg14) : (tensor<4x4xf32>, memref<4x4xf32, strided<[?, ?], offset: ?>>) -> ()
      %39 = "arith.index_cast"(%23) : (i32) -> index
      %40 = "arith.addi"(%arg15, %39) : (index, index) -> index
      %41 = "arith.addi"(%40, %4) : (index, index) -> index
      %42 = "arith.remsi"(%41, %26) : (index, index) -> index
      %43 = "arith.subi"(%41, %42) : (index, index) -> index
      %44 = "arith.addi"(%42, %3) : (index, index) -> index
      %45 = "arith.minsi"(%44, %26) : (index, index) -> index
      %46 = "arith.subi"(%45, %42) : (index, index) -> index
      %47 = "memref.reinterpret_cast"(%arg0, %41, %3, %46, %24, %2) <{operandSegmentSizes = array<i32: 1, 1, 2, 2>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<*xf32>, index, index, index, index, index) -> memref<4x?xf32, strided<[?, ?], offset: ?>>
      %48 = "arith.subi"(%3, %46) : (index, index) -> index
      %49 = "memref.reinterpret_cast"(%arg0, %43, %3, %48, %24, %2) <{operandSegmentSizes = array<i32: 1, 1, 2, 2>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<*xf32>, index, index, index, index, index) -> memref<4x?xf32, strided<[?, ?], offset: ?>>
      %50 = "arith.addi"(%arg16, %3) : (index, index) -> index
      %51 = "arith.addi"(%50, %arg17) : (index, index) -> index
      %52 = "memref.reinterpret_cast"(%arg1, %51, %22, %2) <{operandSegmentSizes = array<i32: 1, 1, 0, 2>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 4, 4>, static_strides = array<i64: -9223372036854775808, -9223372036854775808>}> : (memref<*xf32>, index, index, index) -> memref<4x4xf32, strided<[?, ?], offset: ?>>
      "scf.yield"(%47, %52, %40, %51, %1, %49) : (memref<4x?xf32, strided<[?, ?], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<4x?xf32, strided<[?, ?], offset: ?>>) -> ()
    }) : (i32, i32, i32, memref<4x?xf32, strided<[?, 1], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<4x?xf32, strided<[?, 1], offset: ?>>) -> (memref<4x?xf32, strided<[?, 1], offset: ?>>, memref<4x4xf32, strided<[?, ?], offset: ?>>, index, index, index, memref<4x?xf32, strided<[?, 1], offset: ?>>)
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
