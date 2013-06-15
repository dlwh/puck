package puck.newparser

import puck.linalg.NativeMatrix
import breeze.collection.mutable.TriangularArray

class ParseChart(val length: Int, numLabels: Int, numTermLabels: Int, zero: Float) {
  val top, bot = new ChartHalf(length, numLabels, zero)
  val terms = new TermChart(length, numTermLabels, zero)
}

class ChartHalf(val length: Int, val numLabels: Int, zero: Float) {
  val array = new NativeMatrix[Float](TriangularArray.arraySize(length+1), numLabels)
  array := zero

  def apply(begin: Int, end: Int, label: Int) = array(ChartHalf.chartIndex(begin, end, length), label)

  def spanSlice(spanLength: Int, offset: Int = 0, end: Int = length): NativeMatrix[Float] = {
    val firstIndex: Int = ChartHalf.chartIndex(offset, spanLength, length)
    val lastIndex = math.min(ChartHalf.chartIndex(0,spanLength+1,length), end + firstIndex)
    array(firstIndex until lastIndex, ::)
  }
}

object ChartHalf {
  @inline
  def chartIndex(begin: Int, end: Int, length: Int) = {
    val span = end-begin-1
    begin + span * length - span * (span - 1) / 2

  }
}

class TermChart(length: Int, numTermLabels: Int, zero: Float) {
  val array = new NativeMatrix[Float](length, numTermLabels)
  array := zero
  def rowSlice(begin: Int, end: Int) = array(begin until end, ::)
  def apply(begin: Int, label: Int) = array(begin, label)
}
