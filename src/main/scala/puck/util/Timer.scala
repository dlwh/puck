package puck.util

/**
 *
 *
 * @author dlwh
 */
class Timer {
  private var time = 0L
  private var accum = 0L

  def tic() = {
    time = System.currentTimeMillis()
  }

  def toc() = {
    val out = System.currentTimeMillis()
    accum += (out - time)
    time = out
  }

  def clear() = {
    val out = accumulated
    accum = 0
    out
  }

  def accumulated = accum/1000.0

}
