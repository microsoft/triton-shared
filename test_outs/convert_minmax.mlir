module {
  module {
    tt.func public @minmax_olt(%arg0: !tt.ptr<f32, 1>, %arg1: f32, %arg2: f32) {
      %0 = arith.cmpf olt, %arg1, %arg2 : f32
      %1 = arith.select %0, %arg1, %arg2 : f32
      tt.store %arg0, %1 {cache = 1 : i32, evict = 1 : i32} : f32
      tt.return
    }
  }
  module {
    tt.func public @minmax_ole(%arg0: !tt.ptr<f32, 1>, %arg1: f32, %arg2: f32) {
      %0 = arith.cmpf ole, %arg1, %arg2 : f32
      %1 = arith.select %0, %arg1, %arg2 : f32
      tt.store %arg0, %1 {cache = 1 : i32, evict = 1 : i32} : f32
      tt.return
    }
  }
  module {
    tt.func public @minmax_ogt(%arg0: !tt.ptr<f32, 1>, %arg1: f32, %arg2: f32) {
      %0 = arith.cmpf ogt, %arg1, %arg2 : f32
      %1 = arith.select %0, %arg1, %arg2 : f32
      tt.store %arg0, %1 {cache = 1 : i32, evict = 1 : i32} : f32
      tt.return
    }
  }
  module {
    tt.func public @minmax_oge(%arg0: !tt.ptr<f32, 1>, %arg1: f32, %arg2: f32) {
      %0 = arith.cmpf oge, %arg1, %arg2 : f32
      %1 = arith.select %0, %arg1, %arg2 : f32
      tt.store %arg0, %1 {cache = 1 : i32, evict = 1 : i32} : f32
      tt.return
    }
  }
}

