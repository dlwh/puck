package puck.util

import com.nativelibs4java.opencl._
import org.bridj._
import library._
import OpenCLLibrary._
import java.util
import JavaCL.CL



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
  def fillMemory(data: CLBuffer[java.lang.Float],  f: Float, offset: Int, len: Int, eventsToWaitFor: CLEvent*)(implicit queue: CLQueue): CLEvent = synchronized {
    /*
    val eventsInCount = new Array[Int](1)
    val method =  Class.forName("com.nativelibs4java.opencl.CLAbstractEntity").getDeclaredMethod("getEntity")
    method.setAccessible(true)
    val filtered = if(eventsToWaitFor eq null) Array.empty else eventsToWaitFor.filter(_ ne null).map(method.invoke(_).asInstanceOf[java.lang.Long].longValue).filter(_ != 0).toArray
    val eventsIn = if(filtered.isEmpty) null else Pointer.pointerToArray(filtered).asInstanceOf[Pointer[cl_event]]
    val eventOut: Pointer[cl_event] = if(eventsToWaitFor == null) null else Pointer.allocateTypedPointer(classOf[cl_event]).withoutValidityInformation()
    val fptr = Pointer.pointerToFloat(f)
    val cl = {
      val field = classOf[JavaCL].getDeclaredField("CL")
      field.setAccessible(true)
      field.get(null).asInstanceOf[OpenCLLibrary]
    }
    CLException.error(cl.clEnqueueFillBuffer(
                    new OpenCLLibrary.cl_command_queue(method.invoke(queue).asInstanceOf[java.lang.Long]),
                    new OpenCLLibrary.cl_mem(method.invoke(data).asInstanceOf[java.lang.Long]),
                    fptr,
                    4,
                    offset * 4,
                    len * 4,
                    eventsInCount(0), eventsIn,  eventOut))
    if(eventsIn ne null)
      eventsIn.release()
    classOf[CLEvent].getConstructors.head.newInstance(queue, eventOut).asInstanceOf[CLEvent]
    */
    val ll = if(len < 0) data.getElementCount - offset else len
    kernel.setArgs(data, java.lang.Integer.valueOf(offset), java.lang.Float.valueOf(f), java.lang.Integer.valueOf(ll.toInt))
    // TODO: we possibly waste a lot of time if the offset is >> 0
    // but, we want to ensure that we do coalesced reads and rights, which
    // means aligned reads and writes.
    kernel.enqueueNDRange(queue, Array(data.getElementCount.toInt), eventsToWaitFor:_*)
  }

}


object ZeroMemoryKernel {
  def apply()(implicit context: CLContext) = map.synchronized {
    import scala.collection.JavaConverters._
    map.asScala.getOrElseUpdate(context, new ZeroMemoryKernel)
  }

  private val map = new util.WeakHashMap[CLContext, ZeroMemoryKernel]

}
