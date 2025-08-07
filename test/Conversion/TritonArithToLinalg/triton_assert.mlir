// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

tt.func public @assert_tensor_1d(%arg0: tensor<4xi1>) {
  tt.assert %arg0, "message" : tensor<4xi1>
  tt.return
}

// CHECK-LABEL:   func.func @assert_tensor_1d
// CHECK:           %c1 = arith.constant 1 : index
// CHECK:           %c0 = arith.constant 0 : index
// CHECK:           %c4 = arith.constant 4 : index
// CHECK:           scf.for %{{.*}} = %c0 to %c1 step %c1 {
// CHECK:             scf.for [[INDEX:%.+]] = %c0 to %c4 step %c1 {
// CHECK:               scf.for %{{.*}} = %c0 to %c1 step %c1 {
// CHECK:                 %extracted = tensor.extract %arg0[[[INDEX]]] : tensor<4xi1>
// CHECK:                 cf.assert %extracted, "Assertion `message` failed"
// CHECK:               }
// CHECK:             }
// CHECK:           }

tt.func public @assert_tensor_2d(%arg0: tensor<4x4xi1>) {
  tt.assert %arg0, "message" : tensor<4x4xi1>
  tt.return
}

// CHECK-LABEL:   func.func @assert_tensor_2d
// CHECK:           %c1 = arith.constant 1 : index
// CHECK:           %c0 = arith.constant 0 : index
// CHECK:           %c4 = arith.constant 4 : index
// CHECK:           scf.for [[INDEX0:%.+]] = %c0 to %c4 step %c1 {
// CHECK:             scf.for [[INDEX1:%.+]] = %c0 to %c4 step %c1 {
// CHECK:               scf.for %{{.*}} = %c0 to %c1 step %c1 {
// CHECK:                 %extracted = tensor.extract %arg0[[[INDEX0]], [[INDEX1]]] : tensor<4x4xi1>
// CHECK:                 cf.assert %extracted, "Assertion `message` failed"
// CHECK:               }
// CHECK:             }
// CHECK:           }

tt.func public @assert_tensor_3d(%arg0: tensor<4x4x4xi1>) {
  tt.assert %arg0, "message" : tensor<4x4x4xi1>
  tt.return
}

// CHECK-LABEL:   func.func @assert_tensor_3d
// CHECK:           %c1 = arith.constant 1 : index
// CHECK:           %c0 = arith.constant 0 : index
// CHECK:           %c4 = arith.constant 4 : index
// CHECK:           scf.for [[INDEX0:%.+]] = %c0 to %c4 step %c1 {
// CHECK:             scf.for [[INDEX1:%.+]] = %c0 to %c4 step %c1 {
// CHECK:               scf.for [[INDEX2:%.+]] = %c0 to %c4 step %c1 {
// CHECK:                 %extracted = tensor.extract %arg0[[[INDEX0]], [[INDEX1]], [[INDEX2]]] : tensor<4x4x4xi1>
// CHECK:                 cf.assert %extracted, "Assertion `message` failed"
// CHECK:               }
// CHECK:             }
// CHECK:           }