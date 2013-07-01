package puck.util

import com.nativelibs4java.opencl._
import org.bridj._
import library._
import OpenCLLibrary._
import java.util
import JavaCL.CL
import puck.linalg._



class ZeroMemoryKernel(implicit val context: CLContext) {
  val program = context.createProgram{
"""
__kernel void mem_zero(__global float* data, int beginOffset, float x, int len) {
  int trg = get_global_id(0);
  if(trg >= beginOffset && trg < len + beginOffset)
    data[trg] = x;
}

__kernel void shaped_fill(__global float* data, int beginOffset, int rows, int cols, int stride, float x) {
  int col = get_global_id(1);
  int row = get_global_id(0);
  if(col < cols && row < rows) {
    data[beginOffset + stride * col + row] = x;
  }
}
  """
  }

  val groupSize = if( context.getDevices.head.toString.contains("Apple") && context.getDevices.head.toString.contains("Intel")) {
    1
  } else {
    val x = context.getDevices.head.getMaxWorkItemSizes
    val size0 = x(0)
    math.min(size0, 32).toInt
  }


  val kernel = program.createKernel("mem_zero")
  val shapedKernel = program.createKernel("shaped_fill")

  def shapedFill(data: CLMatrix[Float], f: Float, eventsToWaitFor: CLEvent*)(implicit queue: CLQueue): CLEvent = synchronized {
    val r = if(data.isTranspose) data.cols else data.rows
    val c = if(data.isTranspose) data.rows else data.cols
    shapedKernel.setArgs(data.data.safeBuffer, Integer.valueOf(data.offset), Integer.valueOf(r), Integer.valueOf(c), Integer.valueOf(data.majorStride), java.lang.Float.valueOf(f))

    shapedKernel.enqueueNDRange(queue, Array((r+groupSize-1)/groupSize * groupSize, c), Array(groupSize, 1), eventsToWaitFor:_*)
  }

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
    kernel.enqueueNDRange(queue, Array((data.getElementCount.toInt + groupSize - 1)/groupSize * groupSize), Array(groupSize), eventsToWaitFor:_*)
  }

}


object ZeroMemoryKernel {
  def apply()(implicit context: CLContext) = map.synchronized {
    import scala.collection.JavaConverters._
    map.asScala.getOrElseUpdate(context, new ZeroMemoryKernel)
  }

  private val map = new util.WeakHashMap[CLContext, ZeroMemoryKernel]

}
