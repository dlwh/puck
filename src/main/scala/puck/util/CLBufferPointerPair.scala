package puck.util

import com.nativelibs4java.opencl._
import org.bridj.Pointer

case class CLBufferPointerPair[V](val buffer: CLBuffer[V])(implicit queue: CLQueue) {
  val ptr = Pointer.allocateArray(buffer.getIO, buffer.getElementCount)
  
  def writeInts(dstOffset: Int, src: Array[Int], srcOffset: Int, len: Int, ev: CLEvent*)(implicit wit: Integer =:= V): CLEvent = {
    CLEvent.waitFor(ev:_*)
    ptr.setIntsAtOffset(dstOffset * 4, src, srcOffset, len)
    buffer.write(queue, dstOffset, len, ptr.offset(dstOffset * 4), false)
  }
}


