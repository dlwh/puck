package puck.parser

import scala.collection.mutable.ArrayBuffer

/**
 *
 *
 * @author dlwh
 */
final class ParseItemQueue(maxSize: Int) {

  private var pArray, lArray, rArray = new ArrayBuffer[Array[Int]]
  private var offsets = new Array[Int](10)
  ensure(1)

  def parentQueue(block: Int) = if(block >= pArray.size) null else pArray(block)
  def leftQueue(block: Int) = if(block >= pArray.size) null else lArray(block)
  def rightQueue(block: Int) = if(block >= pArray.size) null else rArray(block)
  def queueSize(block: Int) = if(block >= offsets.length) 0 else offsets(block)


  private def ensure(numBlocks: Int) {
    while(pArray.length < numBlocks) {
      pArray += new Array[Int](maxSize)
      lArray += new Array[Int](maxSize)
      rArray += new Array[Int](maxSize)
    }
    if(offsets.length < numBlocks)  {
      offsets = java.util.Arrays.copyOf(offsets, offsets.length * 2)
    }
  }

  def clear(block: Int): Unit = {
    if(block < offsets.length) {
      offsets(block) = 0
    }

  }

  def clear() {
    java.util.Arrays.fill(offsets, 0)
  }

  def enqueue(block: Int, p: Int, l: Int, r: Int): Unit = {
    ensure(block + 1)
    val size = offsets(block)
    pArray(block)(size) = p
    lArray(block)(size) = l
    rArray(block)(size) = r
    offsets(block) += 1
  }

  def enqueue(block: Int, p: Int, l: Int): Unit = {
    enqueue(block, p, l, -1)
  }

  def nonEmpty: Boolean = {
    var i = 0
    while(i < pArray.length) {
      if (offsets(i) != 0) return true
      i += 1
    }
    false
  }

  def nonEmpty(block: Int): Boolean = block < offsets.length  && offsets(block) != 0

}
