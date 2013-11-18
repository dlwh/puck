package puck.linalg

/**
 * TODO
 *
 * @author dlwh
 **/
trait CanRepresentAsFloatBytes[V] {
  def asFloat(v: V):Float
  def fromFloat(v: Float):V
}

object CanRepresentAsFloatBytes {
  implicit object FloatsIsFloats extends CanRepresentAsFloatBytes[Float] {
    def asFloat(v: Float): Float = v
    def fromFloat(v: Float): Float = v
  }


  implicit object IntsIsFloats extends CanRepresentAsFloatBytes[Int] {
    def asFloat(v: Int): Float = java.lang.Float.intBitsToFloat(v)

    def fromFloat(v: Float): Int = java.lang.Float.floatToRawIntBits(v)
  }
}
