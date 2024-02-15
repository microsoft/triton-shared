module {
  func.func @assert_lol(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sgt, %arg0, %c0_i32 : i32
    cf.assert %0, ".py:0:  Assertion `lol` failed"
    return
  }
}

