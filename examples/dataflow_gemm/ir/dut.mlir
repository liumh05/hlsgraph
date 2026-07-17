module {
  handshake.func @dut(%arg0: !handshake.channel<i32>) -> !handshake.channel<i32> {
    %0 = handshake.buffer %arg0 {numSlots = 8 : ui32} : <i32>
    %1 = handshake.mul %0, %0 : i32 loc("kernel.cpp":18:5)
    %2 = handshake.buffer %1 {numSlots = 16 : ui32} : <i32>
    handshake.return %2 : !handshake.channel<i32>
  }
}
