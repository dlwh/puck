package puck.util

import com.nativelibs4java.opencl._
import java.util

class ZeroMemoryKernel(implicit val context: CLContext) {
  val program = context.createProgram{
"""
__kernel void mem_zero(__global float* data, int beginOffset, float x, int len) {
  int trg = get_global_id(0);
  if(trg >= beginOffset && trg < len + beginOffset)
    data[trg] = x;
}
"""
  }

  val kernel = program.createKernel("mem_zero")


  def fillMemory(data: CLBuffer[java.lang.Float], f: Float, events: CLEvent*)(implicit queue: CLQueue): CLEvent = synchronized {
    fillMemory(data, f, 0, -1, events:_*)
  }
  def fillMemory(data: CLBuffer[java.lang.Float],  f: Float, offset: Int, len: Int, events: CLEvent*)(implicit queue: CLQueue): CLEvent = synchronized {
    val ll = if(len < 0) data.getElementCount - offset else len
    kernel.setArgs(data, java.lang.Integer.valueOf(offset), java.lang.Float.valueOf(f), java.lang.Integer.valueOf(ll.toInt))
    // TODO: we possibly waste a lot of time if the offset is >> 0
    // but, we want to ensure that we do coalesced reads and rights, which
    // means aligned reads and writes.
    kernel.enqueueNDRange(queue, Array(data.getElementCount.toInt), events:_*)
  }

}


object ZeroMemoryKernel {
  def apply(implicit context: CLContext) = map.synchronized {
    import scala.collection.JavaConverters._
    map.asScala.getOrElseUpdate(context, new ZeroMemoryKernel)
  }

  private val map = new util.WeakHashMap[CLContext, ZeroMemoryKernel]

}
