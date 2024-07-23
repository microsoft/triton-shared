// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @nested_who_knows_how_many_levels(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %13 = tt.addptr %12, %3 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %14 = tt.broadcast %13 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %15 = tt.addptr %14, %8 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %16 = arith.muli %arg3, %c2_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18 = arith.muli %arg3, %c2_i32 : i32
    %19 = tt.splat %18 : i32 -> tensor<2x2xi32>
    %20 = arith.muli %arg3, %c2_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<2x2xi32>
    %22 = arith.muli %arg3, %c2_i32 : i32
    %23 = tt.splat %22 : i32 -> tensor<2x2xi32>
    %24:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %25 = tt.load %arg5 : tensor<2x2x!tt.ptr<f32>>
      %26:3 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5, %arg9 = %arg6, %arg10 = %25) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>)  : i32 {
        %29 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %30 = tt.load %29 : tensor<2x2x!tt.ptr<f32>>
        %31:4 = scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg12 = %29, %arg13 = %arg9, %arg14 = %arg10, %arg15 = %30) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
          %33 = tt.addptr %arg12, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %34 = tt.load %33 : tensor<2x2x!tt.ptr<f32>>
          tt.store %arg13, %arg14 : tensor<2x2x!tt.ptr<f32>>
          %35 = tt.addptr %arg13, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %35, %arg15 : tensor<2x2x!tt.ptr<f32>>
          %36 = tt.addptr %35, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %36, %34 : tensor<2x2x!tt.ptr<f32>>
          %37 = tt.addptr %36, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %38:5 = scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg17 = %arg14, %arg18 = %33, %arg19 = %arg15, %arg20 = %34, %arg21 = %37) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
            %40 = tt.load %arg18 : tensor<2x2x!tt.ptr<f32>>
            %41:5 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg23 = %arg18, %arg24 = %arg19, %arg25 = %arg20, %arg26 = %arg21, %arg27 = %40) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>)  : i32 {
              %42 = tt.addptr %arg23, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
              %43 = tt.load %42 : tensor<2x2x!tt.ptr<f32>>
              %44:5 = scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg29 = %42, %arg30 = %arg25, %arg31 = %arg26, %arg32 = %arg27, %arg33 = %43) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
                %45 = tt.addptr %arg29, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                %46 = tt.load %45 : tensor<2x2x!tt.ptr<f32>>
                tt.store %arg31, %arg32 : tensor<2x2x!tt.ptr<f32>>
                %47 = tt.addptr %arg31, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %47, %arg33 : tensor<2x2x!tt.ptr<f32>>
                %48 = tt.addptr %47, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %48, %46 : tensor<2x2x!tt.ptr<f32>>
                %49 = tt.addptr %48, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                %50:5 = scf.for %arg34 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg35 = %arg32, %arg36 = %45, %arg37 = %arg33, %arg38 = %46, %arg39 = %49) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                  %51 = tt.load %arg36 : tensor<2x2x!tt.ptr<f32>>
                  %52:5 = scf.for %arg40 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg41 = %arg36, %arg42 = %arg37, %arg43 = %arg38, %arg44 = %arg39, %arg45 = %51) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>)  : i32 {
                    %53 = tt.addptr %arg41, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                    %54 = tt.load %53 : tensor<2x2x!tt.ptr<f32>>
                    %55:5 = scf.for %arg46 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg47 = %53, %arg48 = %arg43, %arg49 = %arg44, %arg50 = %arg45, %arg51 = %54) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
                      %56 = tt.addptr %arg47, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      %57 = tt.load %56 : tensor<2x2x!tt.ptr<f32>>
                      tt.store %arg49, %arg50 : tensor<2x2x!tt.ptr<f32>>
                      %58 = tt.addptr %arg49, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      tt.store %58, %arg51 : tensor<2x2x!tt.ptr<f32>>
                      %59 = tt.addptr %58, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      tt.store %59, %57 : tensor<2x2x!tt.ptr<f32>>
                      %60 = tt.addptr %59, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      %61:5 = scf.for %arg52 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg53 = %arg50, %arg54 = %56, %arg55 = %arg51, %arg56 = %57, %arg57 = %60) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                        %62 = tt.load %arg54 : tensor<2x2x!tt.ptr<f32>>
                        %63:4 = scf.for %arg58 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg59 = %arg54, %arg60 = %arg55, %arg61 = %arg56, %arg62 = %arg57) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                          %64 = tt.addptr %arg59, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                          %65 = tt.load %64 : tensor<2x2x!tt.ptr<f32>>
                          %66:3 = scf.for %arg63 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg64 = %64, %arg65 = %arg61, %arg66 = %arg62) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                            %67 = tt.addptr %arg64, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            %68 = tt.load %67 : tensor<2x2x!tt.ptr<f32>>
                            tt.store %arg66, %62 : tensor<2x2x!tt.ptr<f32>>
                            %69 = tt.addptr %arg66, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            tt.store %69, %65 : tensor<2x2x!tt.ptr<f32>>
                            %70 = tt.addptr %69, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            tt.store %70, %68 : tensor<2x2x!tt.ptr<f32>>
                            %71 = tt.addptr %70, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            scf.yield %67, %68, %71 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                          }
                          scf.yield %66#0, %65, %66#1, %66#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                        }
                        scf.yield %62, %63#0, %63#1, %63#2, %63#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                      }
                      scf.yield %61#1, %61#3, %61#4, %61#0, %61#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>
                    }
                    scf.yield %55#0, %55#4, %55#1, %55#2, %55#3 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>
                  }
                  scf.yield %52#4, %52#0, %52#1, %52#2, %52#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                }
                scf.yield %50#1, %50#3, %50#4, %50#0, %50#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>
              }
              scf.yield %44#0, %44#4, %44#1, %44#2, %44#3 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>
            }
            scf.yield %41#4, %41#0, %41#1, %41#2, %41#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
          }
          %39:5 = scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg17 = %38#0, %arg18 = %38#1, %arg19 = %38#2, %arg20 = %38#3, %arg21 = %38#4) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
            %40 = tt.load %arg18 : tensor<2x2x!tt.ptr<f32>>
            %41:4 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg23 = %arg18, %arg24 = %arg19, %arg25 = %arg20, %arg26 = %arg21) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
              %42 = tt.addptr %arg23, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
              %43 = tt.load %42 : tensor<2x2x!tt.ptr<f32>>
              %44:3 = scf.for %arg27 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg28 = %42, %arg29 = %arg25, %arg30 = %arg26) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                %45 = tt.addptr %arg28, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                %46 = tt.load %45 : tensor<2x2x!tt.ptr<f32>>
                tt.store %arg30, %40 : tensor<2x2x!tt.ptr<f32>>
                %47 = tt.addptr %arg30, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %47, %43 : tensor<2x2x!tt.ptr<f32>>
                %48 = tt.addptr %47, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %48, %46 : tensor<2x2x!tt.ptr<f32>>
                %49 = tt.addptr %48, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                scf.yield %45, %46, %49 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
              }
              scf.yield %44#0, %43, %44#1, %44#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
            }
            scf.yield %40, %41#0, %41#1, %41#2, %41#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
          }
          scf.yield %39#1, %39#4, %39#0, %39#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>
        }
        %32 = tt.addptr %31#0, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %32, %31#1, %31#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>
      }
      %27:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %26#0, %arg9 = %26#1) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %29 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        %30:2 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
          %32 = tt.addptr %arg11, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %33 = tt.load %32 : tensor<2x2x!tt.ptr<f32>>
          %34:2 = scf.for %arg13 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg14 = %32, %arg15 = %arg12) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
            %35 = tt.addptr %arg14, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            %36 = tt.load %35 : tensor<2x2x!tt.ptr<f32>>
            tt.store %arg15, %29 : tensor<2x2x!tt.ptr<f32>>
            %37 = tt.addptr %arg15, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            tt.store %37, %33 : tensor<2x2x!tt.ptr<f32>>
            %38 = tt.addptr %37, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            tt.store %38, %36 : tensor<2x2x!tt.ptr<f32>>
            %39 = tt.addptr %38, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            scf.yield %35, %39 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
          }
          scf.yield %34#0, %34#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
        }
        %31 = tt.addptr %30#0, %21 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %31, %30#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %28 = tt.addptr %27#0, %23 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %28, %27#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-LABEL:  func.func @nested_who_knows_how_many_levels
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[PARAM_3_]], [[CST_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]]:3 = scf.for [[VAR_arg10_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg11_:%.+]] = [[CST_0_1_]], [[VAR_arg12_:%.+]] = [[VAR_reinterpret_cast_]], [[VAR_arg13_:%.+]] = [[CST_0_1_]]) -> (index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:         [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg11_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:             memref.copy [[VAR_reinterpret_cast_0_]], [[RES_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:             [[VAR_5_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:         [[VAR_6_:%.+]]:4 = scf.for [[VAR_arg14_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg15_:%.+]] = [[VAR_arg11_]], [[VAR_arg16_:%.+]] = [[VAR_arg12_]], [[VAR_arg17_:%.+]] = [[VAR_arg13_]], [[VAR_arg18_:%.+]] = [[VAR_5_]]) -> (index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:           [[VAR_9_:%.+]] = arith.addi [[VAR_arg15_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_9_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:           [[RES_1_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:               memref.copy [[VAR_reinterpret_cast_1_]], [[RES_1_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:               [[VAR_10_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:           [[VAR_11_:%.+]]:5 = scf.for [[VAR_arg19_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg20_:%.+]] = [[VAR_9_]], [[VAR_arg21_:%.+]] = [[VAR_arg16_]], [[VAR_arg22_:%.+]] = [[VAR_arg17_]], [[VAR_arg23_:%.+]] = [[VAR_arg18_]], [[VAR_arg24_:%.+]] = [[VAR_10_]]) -> (index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:             [[VAR_reinterpret_cast_3_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg22_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:             [[VAR_13_:%.+]] = arith.addi [[VAR_arg20_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_reinterpret_cast_4_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_13_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:             [[RES_2_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                 memref.copy [[VAR_reinterpret_cast_4_]], [[RES_2_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:                 [[VAR_14_:%.+]] = bufferization.to_tensor [[RES_2_]] restrict writable : memref<2x2xf32>
// CHECK:                 bufferization.materialize_in_destination [[VAR_arg23_]] in writable [[VAR_reinterpret_cast_3_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                 [[VAR_15_:%.+]] = arith.addi [[VAR_arg22_]], [[VAR_3_]] : index
// CHECK:                 [[VAR_reinterpret_cast_6_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_15_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                 bufferization.materialize_in_destination [[VAR_arg24_]] in writable [[VAR_reinterpret_cast_6_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                 [[VAR_16_:%.+]] = arith.addi [[VAR_15_]], [[VAR_3_]] : index
// CHECK:                 [[VAR_reinterpret_cast_7_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_16_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                 bufferization.materialize_in_destination [[VAR_14_]] in writable [[VAR_reinterpret_cast_7_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                 [[VAR_17_:%.+]] = arith.addi [[VAR_16_]], [[VAR_3_]] : index
// CHECK:                 [[VAR_reinterpret_cast_8_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_17_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:             [[VAR_18_:%.+]]:7 = scf.for [[VAR_arg25_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg26_:%.+]] = [[VAR_arg23_]], [[VAR_arg27_:%.+]] = [[VAR_reinterpret_cast_4_]], [[VAR_arg28_:%.+]] = [[VAR_13_]], [[VAR_arg29_:%.+]] = [[VAR_arg24_]], [[VAR_arg30_:%.+]] = [[VAR_14_]], [[VAR_arg31_:%.+]] = [[VAR_reinterpret_cast_8_]], [[VAR_arg32_:%.+]] = [[VAR_17_]]) -> (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:               [[VAR_reinterpret_cast_9_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg28_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:               [[RES_3_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                   memref.copy [[VAR_reinterpret_cast_9_]], [[RES_3_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:                   [[VAR_20_:%.+]] = bufferization.to_tensor [[RES_3_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:               [[VAR_21_:%.+]]:7 = scf.for [[VAR_arg33_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg34_:%.+]] = [[VAR_arg27_]], [[VAR_arg35_:%.+]] = [[VAR_arg28_]], [[VAR_arg36_:%.+]] = [[VAR_arg29_]], [[VAR_arg37_:%.+]] = [[VAR_arg30_]], [[VAR_arg38_:%.+]] = [[VAR_arg31_]], [[VAR_arg39_:%.+]] = [[VAR_arg32_]], [[VAR_arg40_:%.+]] = [[VAR_20_]]) -> (memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:                 [[VAR_22_:%.+]] = arith.addi [[VAR_arg35_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_reinterpret_cast_11_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_22_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                 [[RES_4_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                     memref.copy [[VAR_reinterpret_cast_11_]], [[RES_4_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:                     [[VAR_23_:%.+]] = bufferization.to_tensor [[RES_4_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:                 [[VAR_24_:%.+]]:7 = scf.for [[VAR_arg41_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg42_:%.+]] = [[VAR_reinterpret_cast_11_]], [[VAR_arg43_:%.+]] = [[VAR_22_]], [[VAR_arg44_:%.+]] = [[VAR_arg37_]], [[VAR_arg45_:%.+]] = [[VAR_arg38_]], [[VAR_arg46_:%.+]] = [[VAR_arg39_]], [[VAR_arg47_:%.+]] = [[VAR_arg40_]], [[VAR_arg48_:%.+]] = [[VAR_23_]]) -> (memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:                   [[VAR_reinterpret_cast_13_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg46_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                   [[VAR_25_:%.+]] = arith.addi [[VAR_arg43_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                   [[VAR_reinterpret_cast_14_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_25_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                   [[RES_5_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                       memref.copy [[VAR_reinterpret_cast_14_]], [[RES_5_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:                       [[VAR_26_:%.+]] = bufferization.to_tensor [[RES_5_]] restrict writable : memref<2x2xf32>
// CHECK:                       bufferization.materialize_in_destination [[VAR_arg47_]] in writable [[VAR_reinterpret_cast_13_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                       [[VAR_27_:%.+]] = arith.addi [[VAR_arg46_]], [[VAR_3_]] : index
// CHECK:                       [[VAR_reinterpret_cast_16_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_27_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                       bufferization.materialize_in_destination [[VAR_arg48_]] in writable [[VAR_reinterpret_cast_16_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                       [[VAR_28_:%.+]] = arith.addi [[VAR_27_]], [[VAR_3_]] : index
// CHECK:                       [[VAR_reinterpret_cast_17_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_28_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                       bufferization.materialize_in_destination [[VAR_26_]] in writable [[VAR_reinterpret_cast_17_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                       [[VAR_29_:%.+]] = arith.addi [[VAR_28_]], [[VAR_3_]] : index
// CHECK:                       [[VAR_reinterpret_cast_18_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_29_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                   [[VAR_30_:%.+]]:7 = scf.for [[VAR_arg49_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg50_:%.+]] = [[VAR_arg47_]], [[VAR_arg51_:%.+]] = [[VAR_reinterpret_cast_14_]], [[VAR_arg52_:%.+]] = [[VAR_25_]], [[VAR_arg53_:%.+]] = [[VAR_arg48_]], [[VAR_arg54_:%.+]] = [[VAR_26_]], [[VAR_arg55_:%.+]] = [[VAR_reinterpret_cast_18_]], [[VAR_arg56_:%.+]] = [[VAR_29_]]) -> (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:                     [[VAR_reinterpret_cast_19_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg52_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                     [[RES_6_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                         memref.copy [[VAR_reinterpret_cast_19_]], [[RES_6_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:                         [[VAR_31_:%.+]] = bufferization.to_tensor [[RES_6_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:                     [[VAR_32_:%.+]]:7 = scf.for [[VAR_arg57_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg58_:%.+]] = [[VAR_arg51_]], [[VAR_arg59_:%.+]] = [[VAR_arg52_]], [[VAR_arg60_:%.+]] = [[VAR_arg53_]], [[VAR_arg61_:%.+]] = [[VAR_arg54_]], [[VAR_arg62_:%.+]] = [[VAR_arg55_]], [[VAR_arg63_:%.+]] = [[VAR_arg56_]], [[VAR_arg64_:%.+]] = [[VAR_31_]]) -> (memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:                       [[VAR_33_:%.+]] = arith.addi [[VAR_arg59_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                       [[VAR_reinterpret_cast_21_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_33_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                       [[RES_7_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                           memref.copy [[VAR_reinterpret_cast_21_]], [[RES_7_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:                           [[VAR_34_:%.+]] = bufferization.to_tensor [[RES_7_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:                       [[VAR_35_:%.+]]:7 = scf.for [[VAR_arg65_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg66_:%.+]] = [[VAR_reinterpret_cast_21_]], [[VAR_arg67_:%.+]] = [[VAR_33_]], [[VAR_arg68_:%.+]] = [[VAR_arg61_]], [[VAR_arg69_:%.+]] = [[VAR_arg62_]], [[VAR_arg70_:%.+]] = [[VAR_arg63_]], [[VAR_arg71_:%.+]] = [[VAR_arg64_]], [[VAR_arg72_:%.+]] = [[VAR_34_]]) -> (memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:                         [[VAR_reinterpret_cast_23_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg70_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                         [[VAR_36_:%.+]] = arith.addi [[VAR_arg67_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                         [[VAR_reinterpret_cast_24_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_36_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                         [[RES_8_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                             memref.copy [[VAR_reinterpret_cast_24_]], [[RES_8_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:                             [[VAR_37_:%.+]] = bufferization.to_tensor [[RES_8_]] restrict writable : memref<2x2xf32>
// CHECK:                             bufferization.materialize_in_destination [[VAR_arg71_]] in writable [[VAR_reinterpret_cast_23_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                             [[VAR_38_:%.+]] = arith.addi [[VAR_arg70_]], [[VAR_3_]] : index
// CHECK:                             [[VAR_reinterpret_cast_26_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_38_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                             bufferization.materialize_in_destination [[VAR_arg72_]] in writable [[VAR_reinterpret_cast_26_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                             [[VAR_39_:%.+]] = arith.addi [[VAR_38_]], [[VAR_3_]] : index
// CHECK:                             [[VAR_reinterpret_cast_27_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_39_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                             bufferization.materialize_in_destination [[VAR_37_]] in writable [[VAR_reinterpret_cast_27_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                             [[VAR_40_:%.+]] = arith.addi [[VAR_39_]], [[VAR_3_]] : index
// CHECK:                             [[VAR_reinterpret_cast_28_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_40_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                         [[VAR_41_:%.+]]:7 = scf.for [[VAR_arg73_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg74_:%.+]] = [[VAR_arg71_]], [[VAR_arg75_:%.+]] = [[VAR_reinterpret_cast_24_]], [[VAR_arg76_:%.+]] = [[VAR_36_]], [[VAR_arg77_:%.+]] = [[VAR_arg72_]], [[VAR_arg78_:%.+]] = [[VAR_37_]], [[VAR_arg79_:%.+]] = [[VAR_reinterpret_cast_28_]], [[VAR_arg80_:%.+]] = [[VAR_40_]]) -> (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:                           [[VAR_reinterpret_cast_29_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg76_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                           [[RES_9_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                               memref.copy [[VAR_reinterpret_cast_29_]], [[RES_9_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK-DAG:                           [[VAR_42_:%.+]] = bufferization.to_tensor [[RES_9_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:                           [[VAR_43_:%.+]]:6 = scf.for [[VAR_arg81_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg82_:%.+]] = [[VAR_arg75_]], [[VAR_arg83_:%.+]] = [[VAR_arg76_]], [[VAR_arg84_:%.+]] = [[VAR_arg77_]], [[VAR_arg85_:%.+]] = [[VAR_arg78_]], [[VAR_arg86_:%.+]] = [[VAR_arg79_]], [[VAR_arg87_:%.+]] = [[VAR_arg80_]]) -> (memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:                             [[VAR_44_:%.+]] = arith.addi [[VAR_arg83_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                             [[VAR_reinterpret_cast_31_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_44_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                             [[RES_10_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                                 memref.copy [[VAR_reinterpret_cast_31_]], [[RES_10_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK-DAG:                             [[VAR_45_:%.+]] = bufferization.to_tensor [[RES_10_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:                             [[VAR_46_:%.+]]:5 = scf.for [[VAR_arg88_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg89_:%.+]] = [[VAR_reinterpret_cast_31_]], [[VAR_arg90_:%.+]] = [[VAR_44_]], [[VAR_arg91_:%.+]] = [[VAR_arg85_]], [[VAR_arg92_:%.+]] = [[VAR_arg86_]], [[VAR_arg93_:%.+]] = [[VAR_arg87_]]) -> (memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:                               [[VAR_reinterpret_cast_33_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg93_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                               [[VAR_47_:%.+]] = arith.addi [[VAR_arg90_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                               [[VAR_reinterpret_cast_34_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_47_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                               [[RES_11_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                                   memref.copy [[VAR_reinterpret_cast_34_]], [[RES_11_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:                                   [[VAR_48_:%.+]] = bufferization.to_tensor [[RES_11_]] restrict writable : memref<2x2xf32>
// CHECK:                                   bufferization.materialize_in_destination [[VAR_42_]] in writable [[VAR_reinterpret_cast_33_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                                   [[VAR_49_:%.+]] = arith.addi [[VAR_arg93_]], [[VAR_3_]] : index
// CHECK:                                   [[VAR_reinterpret_cast_36_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_49_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                                   bufferization.materialize_in_destination [[VAR_45_]] in writable [[VAR_reinterpret_cast_36_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                                   [[VAR_50_:%.+]] = arith.addi [[VAR_49_]], [[VAR_3_]] : index
// CHECK:                                   [[VAR_reinterpret_cast_37_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_50_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                                   bufferization.materialize_in_destination [[VAR_48_]] in writable [[VAR_reinterpret_cast_37_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                                   [[VAR_51_:%.+]] = arith.addi [[VAR_50_]], [[VAR_3_]] : index
// CHECK:                                   [[VAR_reinterpret_cast_38_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_51_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                                   scf.yield [[VAR_reinterpret_cast_34_]], [[VAR_47_]], [[VAR_48_]], [[VAR_reinterpret_cast_38_]], [[VAR_51_]] : memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:                                 }
// CHECK:                                 scf.yield [[VAR_46_]]#0, [[VAR_46_]]#1, [[VAR_45_]], [[VAR_46_]]#2, [[VAR_46_]]#3, [[VAR_46_]]#4 : memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:                               }
// CHECK:                               scf.yield [[VAR_42_]], [[VAR_43_]]#0, [[VAR_43_]]#1, [[VAR_43_]]#2, [[VAR_43_]]#3, [[VAR_43_]]#4, [[VAR_43_]]#5 : tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:                             }
// CHECK:                             scf.yield [[VAR_41_]]#1, [[VAR_41_]]#2, [[VAR_41_]]#4, [[VAR_41_]]#5, [[VAR_41_]]#6, [[VAR_41_]]#0, [[VAR_41_]]#3 : memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>
// CHECK:                           }
// CHECK:                           scf.yield [[VAR_35_]]#0, [[VAR_35_]]#1, [[VAR_35_]]#6, [[VAR_35_]]#2, [[VAR_35_]]#3, [[VAR_35_]]#4, [[VAR_35_]]#5 : memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>
// CHECK:                         }
// CHECK:                         scf.yield [[VAR_32_]]#6, [[VAR_32_]]#0, [[VAR_32_]]#1, [[VAR_32_]]#2, [[VAR_32_]]#3, [[VAR_32_]]#4, [[VAR_32_]]#5 : tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:                       }
// CHECK:                       scf.yield [[VAR_30_]]#1, [[VAR_30_]]#2, [[VAR_30_]]#4, [[VAR_30_]]#5, [[VAR_30_]]#6, [[VAR_30_]]#0, [[VAR_30_]]#3 : memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>
// CHECK:                     }
// CHECK:                     scf.yield [[VAR_24_]]#0, [[VAR_24_]]#1, [[VAR_24_]]#6, [[VAR_24_]]#2, [[VAR_24_]]#3, [[VAR_24_]]#4, [[VAR_24_]]#5 : memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>
// CHECK:                   }
// CHECK:                   scf.yield [[VAR_21_]]#6, [[VAR_21_]]#0, [[VAR_21_]]#1, [[VAR_21_]]#2, [[VAR_21_]]#3, [[VAR_21_]]#4, [[VAR_21_]]#5 : tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:                 }
// CHECK-DAG:             [[VAR_19_:%.+]]:7 = scf.for [[VAR_arg25_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg26_1_:%.+]] = [[VAR_18_]]#0, [[VAR_arg27_1_:%.+]] = [[VAR_18_]]#1, [[VAR_arg28_1_:%.+]] = [[VAR_18_]]#2, [[VAR_arg29_1_:%.+]] = [[VAR_18_]]#3, [[VAR_arg30_1_:%.+]] = [[VAR_18_]]#4, [[VAR_arg31_1_:%.+]] = [[VAR_18_]]#5, [[VAR_arg32_1_:%.+]] = [[VAR_18_]]#6) -> (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:               [[VAR_reinterpret_cast_9_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg28_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:               [[RES_12_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                   memref.copy [[VAR_reinterpret_cast_9_1_]], [[RES_12_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK-DAG:               [[VAR_20_1_:%.+]] = bufferization.to_tensor [[RES_12_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:               [[VAR_21_1_:%.+]]:6 = scf.for [[VAR_arg33_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg34_1_:%.+]] = [[VAR_arg27_1_]], [[VAR_arg35_1_:%.+]] = [[VAR_arg28_1_]], [[VAR_arg36_1_:%.+]] = [[VAR_arg29_1_]], [[VAR_arg37_1_:%.+]] = [[VAR_arg30_1_]], [[VAR_arg38_1_:%.+]] = [[VAR_arg31_1_]], [[VAR_arg39_1_:%.+]] = [[VAR_arg32_1_]]) -> (memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:                 [[VAR_22_1_:%.+]] = arith.addi [[VAR_arg35_1_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[VAR_reinterpret_cast_11_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_22_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                 [[RES_13_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                     memref.copy [[VAR_reinterpret_cast_11_1_]], [[RES_13_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK-DAG:                 [[VAR_23_1_:%.+]] = bufferization.to_tensor [[RES_13_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:                 [[VAR_24_1_:%.+]]:5 = scf.for [[VAR_arg40_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg41_1_:%.+]] = [[VAR_reinterpret_cast_11_1_]], [[VAR_arg42_1_:%.+]] = [[VAR_22_1_]], [[VAR_arg43_1_:%.+]] = [[VAR_arg37_1_]], [[VAR_arg44_1_:%.+]] = [[VAR_arg38_1_]], [[VAR_arg45_1_:%.+]] = [[VAR_arg39_1_]]) -> (memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:                   [[VAR_reinterpret_cast_13_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg45_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                   [[VAR_25_1_:%.+]] = arith.addi [[VAR_arg42_1_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                   [[VAR_reinterpret_cast_14_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_25_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:                   [[RES_14_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                       memref.copy [[VAR_reinterpret_cast_14_1_]], [[RES_14_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:                       [[VAR_26_1_:%.+]] = bufferization.to_tensor [[RES_14_]] restrict writable : memref<2x2xf32>
// CHECK:                       bufferization.materialize_in_destination [[VAR_20_1_]] in writable [[VAR_reinterpret_cast_13_1_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                       [[VAR_27_1_:%.+]] = arith.addi [[VAR_arg45_1_]], [[VAR_3_]] : index
// CHECK:                       [[VAR_reinterpret_cast_16_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_27_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                       bufferization.materialize_in_destination [[VAR_23_1_]] in writable [[VAR_reinterpret_cast_16_1_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                       [[VAR_28_1_:%.+]] = arith.addi [[VAR_27_1_]], [[VAR_3_]] : index
// CHECK:                       [[VAR_reinterpret_cast_17_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_28_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                       bufferization.materialize_in_destination [[VAR_26_1_]] in writable [[VAR_reinterpret_cast_17_1_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                       [[VAR_29_1_:%.+]] = arith.addi [[VAR_28_1_]], [[VAR_3_]] : index
// CHECK:                       [[VAR_reinterpret_cast_18_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_29_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                       scf.yield [[VAR_reinterpret_cast_14_1_]], [[VAR_25_1_]], [[VAR_26_1_]], [[VAR_reinterpret_cast_18_1_]], [[VAR_29_1_]] : memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:                     }
// CHECK:                     scf.yield [[VAR_24_1_]]#0, [[VAR_24_1_]]#1, [[VAR_23_1_]], [[VAR_24_1_]]#2, [[VAR_24_1_]]#3, [[VAR_24_1_]]#4 : memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:                   }
// CHECK:                   scf.yield [[VAR_20_1_]], [[VAR_21_1_]]#0, [[VAR_21_1_]]#1, [[VAR_21_1_]]#2, [[VAR_21_1_]]#3, [[VAR_21_1_]]#4, [[VAR_21_1_]]#5 : tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:                 }
// CHECK:                 scf.yield [[VAR_19_]]#2, [[VAR_19_]]#5, [[VAR_19_]]#6, [[VAR_19_]]#0, [[VAR_19_]]#3 : index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>, tensor<2x2xf32>
// CHECK:               }
// CHECK:               [[VAR_12_:%.+]] = arith.addi [[VAR_11_]]#0, [[VAR_3_]] : index
// CHECK:               scf.yield [[VAR_12_]], [[VAR_11_]]#1, [[VAR_11_]]#2, [[VAR_11_]]#3 : index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index, tensor<2x2xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_7_:%.+]]:3 = scf.for [[VAR_arg14_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg15_1_:%.+]] = [[VAR_6_]]#0, [[VAR_arg16_1_:%.+]] = [[VAR_6_]]#1, [[VAR_arg17_1_:%.+]] = [[VAR_6_]]#2) -> (index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:           [[VAR_reinterpret_cast_1_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg15_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:           [[RES_15_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:               memref.copy [[VAR_reinterpret_cast_1_1_]], [[RES_15_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK-DAG:           [[VAR_9_1_:%.+]] = bufferization.to_tensor [[RES_15_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:           [[VAR_10_1_:%.+]]:3 = scf.for [[VAR_arg18_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg19_1_:%.+]] = [[VAR_arg15_1_]], [[VAR_arg20_1_:%.+]] = [[VAR_arg16_1_]], [[VAR_arg21_1_:%.+]] = [[VAR_arg17_1_]]) -> (index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:             [[VAR_12_1_:%.+]] = arith.addi [[VAR_arg19_1_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_reinterpret_cast_3_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_12_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:             [[RES_16_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                 memref.copy [[VAR_reinterpret_cast_3_1_]], [[RES_16_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK-DAG:             [[VAR_13_1_:%.+]] = bufferization.to_tensor [[RES_16_]] restrict writable : memref<2x2xf32>
// CHECK-DAG:             [[VAR_14_1_:%.+]]:3 = scf.for [[VAR_arg22_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg23_1_:%.+]] = [[VAR_12_1_]], [[VAR_arg24_1_:%.+]] = [[VAR_arg20_1_]], [[VAR_arg25_2_:%.+]] = [[VAR_arg21_1_]]) -> (index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index)  : i32 {
// CHECK-DAG:               [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg25_2_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:               [[VAR_15_1_:%.+]] = arith.addi [[VAR_arg23_1_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_reinterpret_cast_6_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_15_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK-DAG:               [[RES_17_:%.+]] = memref.alloc() : memref<2x2xf32>
// CHECK:                   memref.copy [[VAR_reinterpret_cast_6_1_]], [[RES_17_]] : memref<2x2xf32, strided<[?, ?], offset: ?>> to memref<2x2xf32>
// CHECK:                   [[VAR_16_1_:%.+]] = bufferization.to_tensor [[RES_17_]] restrict writable : memref<2x2xf32>
// CHECK:                   bufferization.materialize_in_destination [[VAR_9_1_]] in writable [[VAR_reinterpret_cast_5_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                   [[VAR_17_1_:%.+]] = arith.addi [[VAR_arg25_2_]], [[VAR_3_]] : index
// CHECK:                   [[VAR_reinterpret_cast_8_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_17_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                   bufferization.materialize_in_destination [[VAR_13_1_]] in writable [[VAR_reinterpret_cast_8_1_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                   [[VAR_18_1_:%.+]] = arith.addi [[VAR_17_1_]], [[VAR_3_]] : index
// CHECK:                   [[VAR_reinterpret_cast_9_2_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_18_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                   bufferization.materialize_in_destination [[VAR_16_1_]] in writable [[VAR_reinterpret_cast_9_2_]] : (tensor<2x2xf32>, memref<2x2xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:                   [[VAR_19_1_:%.+]] = arith.addi [[VAR_18_1_]], [[VAR_3_]] : index
// CHECK:                   [[VAR_reinterpret_cast_10_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_19_1_]]{{.}}, sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}} : memref<*xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
// CHECK:                   scf.yield [[VAR_15_1_]], [[VAR_reinterpret_cast_10_]], [[VAR_19_1_]] : index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:                 }
// CHECK:                 scf.yield [[VAR_14_1_]]#0, [[VAR_14_1_]]#1, [[VAR_14_1_]]#2 : index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:               }
// CHECK:               [[VAR_11_1_:%.+]] = arith.addi [[VAR_10_1_]]#0, [[VAR_3_]] : index
// CHECK:               scf.yield [[VAR_11_1_]], [[VAR_10_1_]]#1, [[VAR_10_1_]]#2 : index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:             }
// CHECK:             [[VAR_8_:%.+]] = arith.addi [[VAR_7_]]#0, [[VAR_3_]] : index
// CHECK:             scf.yield [[VAR_8_]], [[VAR_7_]]#1, [[VAR_7_]]#2 : index, memref<2x2xf32, strided<[?, ?], offset: ?>>, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
