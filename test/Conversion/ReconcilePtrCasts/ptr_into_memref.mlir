// RUN: triton-shared-opt --reconcile-ptr-casts %s | FileCheck %s

module {
  func.func @bitcast_ptr_as_src(%arg0: memref<*xi32>, %arg1: memref<*xi32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %0 = tptr.type_offset i32  : i32
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %1 = builtin.unrealized_conversion_cast %arg1 : memref<*xi32> to !tt.ptr<i32>
    %2 = builtin.unrealized_conversion_cast %1 : !tt.ptr<i32> to !ptr.ptr<#tptr.default_memory_space>
    %3 = builtin.unrealized_conversion_cast %arg0 : memref<*xi32> to !tt.ptr<i32>
    %4 = builtin.unrealized_conversion_cast %3 : !tt.ptr<i32> to !ptr.ptr<#tptr.default_memory_space>
    %5 = arith.muli %c1_i32, %0 : i32
    %6 = tptr.ptradd %4 %5 : !ptr.ptr<#tptr.default_memory_space>, i32 to !ptr.ptr<#tptr.default_memory_space>
    %7 = builtin.unrealized_conversion_cast %6 : !ptr.ptr<#tptr.default_memory_space> to !tt.ptr<i64>
    %8 = builtin.unrealized_conversion_cast %7 : !tt.ptr<i64> to memref<*xi64>
    %reinterpret_cast = memref.reinterpret_cast %8 to offset: [%c2], sizes: [16], strides: [1] : memref<*xi64> to memref<16xi64, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<16xi64>
    memref.copy %reinterpret_cast, %alloc : memref<16xi64, strided<[1], offset: ?>> to memref<16xi64>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<16xi64> to tensor<16xi64>
    %10 = tptr.ptradd %2 %5 : !ptr.ptr<#tptr.default_memory_space>, i32 to !ptr.ptr<#tptr.default_memory_space>
    %11 = builtin.unrealized_conversion_cast %10 : !ptr.ptr<#tptr.default_memory_space> to !tt.ptr<i64>
    %12 = builtin.unrealized_conversion_cast %11 : !tt.ptr<i64> to memref<*xi64>
    %reinterpret_cast_0 = memref.reinterpret_cast %12 to offset: [%c2], sizes: [16], strides: [1] : memref<*xi64> to memref<16xi64, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %9 in writable %reinterpret_cast_0 : (tensor<16xi64>, memref<16xi64, strided<[1], offset: ?>>) -> ()
    return
  }
}

// CHECK-NOT: unrealized_conversion_cast
