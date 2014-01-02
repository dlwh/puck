package puck.util

import com.nativelibs4java.opencl._
import org.bridj.Pointer

case class CLBufferPointerPair[V](val buffer: CLBuffer[V])(implicit queue: CLQueue) {
  private var _mappedPointer:Pointer[V] = null
  
  def mappedPointer:Pointer[V] = mappedPointer()

  def mappedPointer(events: CLEvent*): Pointer[V] = synchronized {
    if(_mappedPointer eq null) {
      _mappedPointer = buffer.map(queue, CLMem.MapFlags.ReadWrite, events:_*)
    }
    _mappedPointer
  }

  def unmap(evs: CLEvent*) = synchronized {
    if(_mappedPointer ne null) { 
      val ev = buffer.unmap(queue, _mappedPointer, evs:_*)
      val mm = _mappedPointer
      ev.invokeUponCompletion(new Runnable() {
        def run() {  mm.release() }
        })
      _mappedPointer = null
      ev
    } else {
      queue.enqueueWaitForEvents(evs:_*)
      null
    }
  }

  def writeInts(dstOffset: Int, src: Array[Int], srcOffset: Int, len: Int, ev: CLEvent*)(implicit wit: Integer =:= V): CLEvent = {
    val ptr = mappedPointer(ev:_*)
    ptr.setIntsAtOffset(dstOffset * 4, src, srcOffset, len)
    unmap()
  }

  def waitUnmap() {
    Option(unmap()).foreach(_.waitFor)
  }

  def release() {
    waitUnmap()
    buffer.release()
  }

  def safeBuffer = { waitUnmap(); buffer }
}


