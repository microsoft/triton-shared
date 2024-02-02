module {
  tt.func public @assert_lol(%arg0: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sgt, %arg0, %c0_i32 : i32
    %1 = tt.splat %0 : (i1) -> tensor<1xi1>
    tt.assert %1, "lol", "", "", 0 : tensor<1xi1>
    tt.return
  }
}

