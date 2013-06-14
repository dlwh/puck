package puck.util

/**
 * stuff from http://graphics.stanford.edu/~seander/bithacks.html
 *
 * @author dlwh
 **/
object BitHacks {

  private val logTable256 = new Array[Int](256)
  for(i <- 2 until 256) {
    logTable256(i) = 1 + logTable256(i/2)
  }

  def log2(v: Int) = {
    var tt = 0
    if ({tt = v >> 24; tt != 0}) {
      24 + logTable256(tt)
    } else if ({tt = v >> 16; tt != 0}) {
      16 + logTable256(tt)
    } else if ({tt = v >> 8; tt != 0}) {
      8 + logTable256(tt)
    } else {
      logTable256(v)
    }

  }


  def roundToNextPowerOfTwo(vv: Int) = {
    var v = vv - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    v
  }
}

