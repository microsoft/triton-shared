"builtin.module"() ({
  "tt.func"() <{function_type = (!tt.ptr<f32>, !tt.ptr<f32>, i32, i32) -> (), sym_name = "nested2_complex_body", sym_visibility = "public"}> ({
  ^bb0(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = dense<3> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
    %5 = "arith.constant"() <{value = dense<1> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
    %6 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %7 = "tt.make_range"() <{end = 2 : i32, start = 0 : i32}> : () -> tensor<2xi32>
    %8 = "tt.expand_dims"(%7) <{axis = 1 : i32}> : (tensor<2xi32>) -> tensor<2x1xi32>
    %9 = "tt.splat"(%arg2) : (i32) -> tensor<2x1xi32>
    %10 = "arith.muli"(%8, %9) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x1xi32>
    %11 = "tt.expand_dims"(%7) <{axis = 0 : i32}> : (tensor<2xi32>) -> tensor<1x2xi32>
    %12 = "tt.splat"(%arg3) : (i32) -> tensor<1x2xi32>
    %13 = "arith.muli"(%11, %12) <{overflowFlags = #arith.overflow<none>}> : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
    %14 = "tt.broadcast"(%10) : (tensor<2x1xi32>) -> tensor<2x2xi32>
    %15 = "tt.broadcast"(%13) : (tensor<1x2xi32>) -> tensor<2x2xi32>
    %16 = "arith.addi"(%14, %15) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %17 = "tt.splat"(%1) : (i32) -> tensor<2x2xi32>
    %18 = "arith.addi"(%17, %16) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %19 = "tt.addptr"(%17, %16) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
    %20 = "tt.splat"(%0) : (i32) -> tensor<2x1xi32>
    %21 = "arith.addi"(%20, %10) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x1xi32>
    %22 = "tt.addptr"(%20, %10) : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x1x!tt.ptr<f32>>
    %23 = "tt.broadcast"(%22) : (tensor<2x1x!tt.ptr<f32>>) -> tensor<2x2xi32>
    %24 = "arith.addi"(%23, %15) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %25 = "tt.addptr"(%23, %15) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
    %26 = "arith.muli"(%arg2, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %27 = "tt.splat"(%26) : (i32) -> tensor<2x2xi32>
    %28:2 = "scf.for"(%3, %6, %2, %19, %25) ({
    ^bb0(%arg4: i32, %arg5: tensor<2x2xi32>, %arg6: tensor<2x2xi32>):
      %29 = "arith.addi"(%arg5, %5) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
      %30 = "tt.addptr"(%arg5, %5) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
      %31 = "arith.addi"(%arg6, %5) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
      %32 = "tt.addptr"(%arg6, %5) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
      %33:2 = "scf.for"(%3, %6, %2, %30, %32) ({
      ^bb0(%arg7: i32, %arg8: tensor<2x2xi32>, %arg9: tensor<2x2xi32>):
        %42 = "tts.make_unstructured_tptr"(%arg0, %arg8) : (!tt.ptr<f32>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
        %43 = "tt.load"(%42) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
        %44 = "tts.make_unstructured_tptr"(%arg1, %arg9) : (!tt.ptr<f32>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
        "tt.store"(%44, %43) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
        %45 = "arith.addi"(%arg8, %4) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
        %46 = "tt.addptr"(%arg8, %4) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
        %47 = "arith.addi"(%arg9, %4) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
        %48 = "tt.addptr"(%arg9, %4) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
        "scf.yield"(%46, %48) : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>) -> ()
      }) : (i32, i32, i32, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2xi32>, tensor<2x2xi32>)
      %34 = "arith.addi"(%arg5, %27) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
      %35 = "tt.addptr"(%arg5, %27) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
      %36 = "arith.addi"(%35, %5) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
      %37 = "tt.addptr"(%35, %5) : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
      %38 = "arith.addi"(%arg6, %27) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
      %39 = "tt.addptr"(%arg6, %27) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
      %40 = "arith.addi"(%39, %5) <{overflowFlags = #arith.overflow<none>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
      %41 = "tt.addptr"(%39, %5) : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
      "scf.yield"(%37, %41) : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>) -> ()
    }) : (i32, i32, i32, tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>) -> (tensor<2x2xi32>, tensor<2x2xi32>)
    "tt.return"() : () -> ()
  }) {noinline = false} : () -> ()
}) : () -> ()