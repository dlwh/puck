package puck.newparser

import puck.linalg.CLMatrix
import breeze.collection.mutable.TriangularArray

class ParseChart(val length: Int, botMat: CLMatrix[Float], topMat: CLMatrix[Float]) {
  val top = new ChartHalf(length, botMat)
  val bot = new ChartHalf(length, topMat)
}

class ChartHalf(val length: Int, val matrix: CLMatrix[Float]) {

  def apply(begin: Int, end: Int, label: Int) = matrix(ChartHalf.chartIndex(begin, end, length), label)

  def spanSlice(spanLength: Int, offset: Int = 0, end: Int = length): CLMatrix[Float] = {
    val firstIndex: Int = ChartHalf.chartIndex(offset, spanLength, length)
    val lastIndex = math.min(ChartHalf.chartIndex(0,spanLength+1,length), end + firstIndex)
    matrix(firstIndex until lastIndex, ::)
  }
}

object ChartHalf {
  @inline
  def chartIndex(begin: Int, end: Int, length: Int) = {
    val span = end-begin-1
    begin + span * length - span * (span - 1) / 2

  }
}

