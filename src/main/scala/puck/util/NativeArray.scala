package puck.util

import org.bridj.Pointer
import scala.reflect.ClassTag

/**
 * TODO
 *
 * @author dlwh
 **/
class NativeArray[@specialized(Int, Float, Double) T](val pointer: Pointer[T], val length: Long, var autorelease: Boolean = true) {
  def apply(i: Long) = {require(i < length); pointer.get(i)}
  def toArray = {require(length <= Int.MaxValue); pointer.toArray.take(length.toInt)}
  def update(i: Long, v: T) {
    pointer(i) = v
  }



  def release() {
    if(!released) {
      _released = true
      pointer.release()
    }
  }

  def released = _released
  private var _released = false

  override def finalize() {
    if(autorelease) release()
  }
}

object NativeArray {
  def apply[T:ClassTag](length: Long) = new NativeArray[T](PointerUtil.allocate[T](length), length)
  def apply[T](data: Array[T]):NativeArray[T] = {
    new NativeArray[T](Pointer.pointerToArray[T](data), data.length)
  }
}
