package puck.util

import scala.reflect.ClassTag
import org.bridj.Pointer

/**
 * TODO
 *
 * @author dlwh
 **/
object PointerUtil {
  def allocate[T: ClassTag](length: Long = 1) = implicitly[ClassTag[T]] match {
    case ClassTag.Float => Pointer.allocateFloats(length).asInstanceOf[Pointer[T]]
    case ClassTag.Double => Pointer.allocateDoubles(length).asInstanceOf[Pointer[T]]
    case ClassTag.Int => Pointer.allocateInts(length).asInstanceOf[Pointer[T]]
    case ClassTag.Long => Pointer.allocateLongs(length).asInstanceOf[Pointer[T]]
    case ClassTag.Short => Pointer.allocateShorts(length).asInstanceOf[Pointer[T]]
    case ClassTag.Char => Pointer.allocateChars(length).asInstanceOf[Pointer[T]]
    case x => Pointer.allocateArray(x.runtimeClass, length).asInstanceOf[Pointer[T]]
  }


}
