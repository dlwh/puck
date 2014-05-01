package puck.parser

import java.util.concurrent.ConcurrentLinkedQueue

/**
 *
 *
 * @author dlwh
 */
class ObjectPool[T](gen: =>T) {

  private val queue = new ConcurrentLinkedQueue[T]()

  def get():T = {
    val t = queue.poll()
    if (t == null) {
      gen
    }  else {
      t
    }
  }

  def put(t: T):Unit = {
    if(queue.size() < 10)
      queue.offer(t)
  }

}
