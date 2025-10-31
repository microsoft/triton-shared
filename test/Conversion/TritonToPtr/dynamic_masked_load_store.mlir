// Test that triton load and store with mask are lowered correctly
// (scf.if guarding the load and store)
// RUN: triton-shared-opt --triton-arith-to-linalg="tensor-ptr-to-linalg" --triton-to-ptr --cse --canonicalize %s | FileCheck %s

module {
  tt.func public @tensor_ptr(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16xi32>
    %cst_0 = arith.constant dense<0> : tensor<16xi32>
    %cst_1 = arith.constant dense<16> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
    %4 = arith.extsi %3 : tensor<16xi32> to tensor<16xi64>
    %5 = tt.int_to_ptr %4 : tensor<16xi64> -> tensor<16x!tt.ptr<i32>>
    %6 = tt.addptr %5, %cst_1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %7 = tt.load %6 : tensor<16x!tt.ptr<i32>>
    %8 = arith.cmpi ne, %7, %cst_0 : tensor<16xi32>
    %9 = tt.load %5, %8 : tensor<16x!tt.ptr<i32>>
    %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %11 = tt.addptr %10, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %12 = tt.ptr_to_int %11 : tensor<16x!tt.ptr<i32>> -> tensor<16xi64>
    %13 = arith.extsi %9 : tensor<16xi32> to tensor<16xi64>
    %14 = arith.addi %12, %13 : tensor<16xi64>
    %15 = tt.bitcast %11 : tensor<16x!tt.ptr<i32>> -> tensor<16x!tt.ptr<i64>>
    %16 = tt.addptr %15, %cst : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    %17 = tt.bitcast %16 : tensor<16x!tt.ptr<i64>> -> tensor<16x!tt.ptr<i32>>
    %18 = arith.trunci %14 : tensor<16xi64> to tensor<16xi32>
    tt.store %17, %18, %8 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}


// CHECK-COUNT-7: linalg.generic

// CHECK: [[MASK:%.+]] = linalg.generic
// CHECK: ^bb0(%in: i32, %in_0: i32, %out: i1):
// CHECK:     [[YIELD:%.+]] = arith.cmpi ne, %in, %in_0 : i32
// CHECK:     linalg.yield [[YIELD]] : i1
// CHECK: } -> tensor<16xi1>

// CHECK:  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%{{.+}}, [[MASK]] : tensor<16x!ptr.ptr<#ptr.generic_space>>, tensor<16xi1>)
// CHECK:  ^bb0(%in: !ptr.ptr<#ptr.generic_space>, %in_0: i1, %out: i32):
// CHECK:    [[MEMREF:%.+]] = tptr.to_memref %in : <#ptr.generic_space> to memref<1xi32>
// CHECK:    [[SCF_IF:%.+]] = scf.if %in_0 -> (i32) {
// CHECK:      [[LOAD:%.+]] = memref.load [[MEMREF]][%c0] : memref<1xi32>
// CHECK:      scf.yield [[LOAD]] : i32
// CHECK:    } else {
// CHECK:      scf.yield %c0_i32 : i32
// CHECK:    }
// CHECK:    linalg.yield [[SCF_IF]] : i32
// CHECK:  } -> tensor<16xi32>

// CHECK:  linalg.generic
// CHECK:  ^bb0(%in: !ptr.ptr<#ptr.generic_space>, %in_0: i32, %in_1: i1):
// CHECK:    scf.if %in_1 {
// CHECK:      [[MEMREF:%.+]] = tptr.to_memref %in : <#ptr.generic_space> to memref<1xi32>
// CHECK:      memref.store %in_0, [[MEMREF]][%c0] : memref<1xi32>
// CHECK:    }
// CHECK:    linalg.yield
// CHECK:  }
