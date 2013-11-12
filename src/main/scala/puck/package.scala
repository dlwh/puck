import org.bridj.{PointerIO, Pointer}
import com.nativelibs4java.opencl._
import java.{lang=>jl}
import puck.util.CLBufferMappedPointerPair
import scala.reflect._

/**
 * TODO
 *
 * @author dlwh
 **/
package object puck {
  implicit class RichPointer[T](val pointer: Pointer[T])(implicit tag: scala.reflect.ClassTag[T]) {
    def update(v: T) {pointer.set(v)}
    def update(off: Long, v: T) {pointer.set(off, v)}
    def apply(off: Long = 0) {pointer.get(off)}


    def +(off: Long) = pointer.next(off)
    
  }

  implicit class RichFloatPointer(val pointer: Pointer[Float]) extends AnyVal {
    def update(v: Float) {pointer.setFloat(v)}
    def update(off: Long, v: Float) {pointer.setFloatAtIndex(off, v)}
    def apply(off: Long = 0) {pointer.getFloatAtIndex(off)}
    def copyToArray(array: Array[Float]) {
      pointer.getFloats(array)
    }

    def +(off: Long) = pointer.next(off)
  }

  implicit class RichIntPointer(val pointer: Pointer[Int]) extends AnyVal {
    def update(v: Int) {pointer.setInt(v)}
    def update(off: Long, v: Int) {pointer.setIntAtIndex(off, v)}
    def apply(off: Long = 0) {pointer.getIntAtIndex(off)}
    def copyToArray(array: Array[Int]) {
      pointer.getInts(array)
    }

    def +(off: Long) = pointer.next(off)
  }

  implicit class RichDoublePointer(val pointer: Pointer[Double]) extends AnyVal {
    def update(v: Double) {pointer.setDouble(v)}
    def update(off: Long, v: Double) {pointer.setDoubleAtIndex(off, v)}
    def apply(off: Long = 0) {pointer.getDoubleAtIndex(off)}
    def copyToArray(array: Array[Double]) {
      pointer.getDoubles(array)
    }


    def +(off: Long) = pointer.next(off)
  }

  implicit class RichCharPointer(val pointer: Pointer[Char]) extends AnyVal {
    def update(v: Char) {pointer.setChar(v)}
    def update(off: Long, v: Char) {pointer.setCharAtIndex(off, v)}
    def apply(off: Long = 0) {pointer.getCharAtIndex(off)}
    def copyToArray(array: Array[Char]) {
      pointer.getChars(array.length).copyToArray(array)
    }

    def +(off: Long) = pointer.next(off)
  }

  implicit class RichLongPointer(val pointer: Pointer[Long]) extends AnyVal {
    def update(v: Long) {pointer.setLong(v)}
    def update(off: Long, v: Long) {pointer.setLongAtIndex(off, v)}
    def apply(off: Long = 0) {pointer.getLongAtIndex(off)}
    def copyToArray(array: Array[Long]) {
      pointer.getLongs(array.length).copyToArray(array)
    }

    def +(off: Long) = pointer.next(off)
  }

  implicit def bufFloatTojlFloatBuffer(buffer: CLBuffer[Float]) = buffer.asInstanceOf[CLBuffer[jl.Float]]
  implicit def bufDoubleTojlDoubleBuffer(buffer: CLBuffer[Double]) = buffer.asInstanceOf[CLBuffer[jl.Double]]
  implicit def bufIntTojlIntBuffer(buffer: CLBuffer[Int]) = buffer.asInstanceOf[CLBuffer[jl.Integer]]

  implicit def bufFloatTojlFloatBuffer(buffer: CLBufferMappedPointerPair[Float]) = CLBufferMappedPointerPair.toBuffer(buffer).asInstanceOf[CLBuffer[jl.Float]]
  implicit def bufDoubleTojlDoubleBuffer(buffer: CLBufferMappedPointerPair[Double]) = CLBufferMappedPointerPair.toBuffer(buffer).asInstanceOf[CLBuffer[jl.Double]]
  implicit def bufIntTojlIntBuffer(buffer: CLBufferMappedPointerPair[Int]) = CLBufferMappedPointerPair.toBuffer(buffer).asInstanceOf[CLBuffer[jl.Integer]]

  implicit def bufjlFloatToFloatBuffer(buffer: CLBuffer[jl.Float]) = buffer.asInstanceOf[CLBuffer[Float]]
  implicit def bufjlDoubleToDoubleBuffer(buffer: CLBuffer[jl.Double]) = buffer.asInstanceOf[CLBuffer[Double]]
  implicit def bufjlIntToIntBuffer(buffer: CLBuffer[jl.Integer]) = buffer.asInstanceOf[CLBuffer[Int]]


  implicit class RichCLIntBuffer(val buffer: CLBuffer[Integer])(implicit tag: scala.reflect.ClassTag[Integer]) {

    def writeArray(queue: CLQueue, arr: Array[Int], lengthOfArray: Int, events: CLEvent*):CLEvent = {
      require(lengthOfArray <= arr.length)
      val ptr = if(lengthOfArray == arr.length) {
        Pointer.pointerToArray[Integer](arr)
      } else {
        Pointer.allocateInts(lengthOfArray).setIntsAtOffset(0, arr, 0, lengthOfArray)
      }
      val ev = buffer.write(queue, ptr, false, events:_*)
      ev.invokeUponCompletion(new Runnable {
        def run() = {ptr.release()}
      })

      ev
    }
  }
}
