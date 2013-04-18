package puck.util

import com.nativelibs4java.opencl._
import java.util

class ZeroMemoryKernel(zero: Float)(implicit val context: CLContext) {
  val program = context.createProgram{
"""
__kernel void mem_zero(__global float* data, float x, int len) {
  int trg = get_global_id(0);
  if(trg < len)
    data[trg] = x;
}
"""
  }

  val kernel = program.createKernel("mem_zero")


  def zeroMemory(data: CLBuffer[java.lang.Float], events: CLEvent*)(implicit queue: CLQueue): CLEvent = synchronized {
    fillMemory(data, zero, events:_*)
  }

  def fillMemory(data: CLBuffer[java.lang.Float], f: Float, events: CLEvent*)(implicit queue: CLQueue): CLEvent = synchronized {
    kernel.setArgs(data, java.lang.Float.valueOf(f), java.lang.Integer.valueOf(data.getElementCount.toInt))
    kernel.enqueueNDRange(queue, Array(data.getElementCount.toInt), events:_*)
  }


  def fillMemory(data: MemBufPair[Float], f: Float, events: CLEvent*)(implicit queue: CLQueue): CLEvent = synchronized {
    kernel.setArgs(data.dev, java.lang.Float.valueOf(f), java.lang.Integer.valueOf(data.dev.getElementCount.toInt))
    kernel.enqueueNDRange(queue, Array(data.dev.getElementCount.toInt), events:_*)
  }
}


object ZeroMemoryKernel {
  // TODO: leak here.
  def apply(zero: Float)(implicit context: CLContext) = map.synchronized {
    import scala.collection.JavaConverters._
    map.asScala.getOrElseUpdate(context, new ZeroMemoryKernel(zero))
  }

  private val map = new util.IdentityHashMap[CLContext, ZeroMemoryKernel]

}
