package puck.newparser

import generator.RuleStructure
import puck.linalg.CLMatrix
import breeze.collection.mutable.TriangularArray

class ParseChart(val length: Int, botMat: CLMatrix[Float], topMat: CLMatrix[Float]) {
  val top = new ChartHalf(length, topMat, false)
  val bot = new ChartHalf(length, botMat, true)
}

class ChartHalf(val length: Int, val matrix: CLMatrix[Float], isBot: Boolean) {
  val globalRowOffset = matrix.offset / matrix.rows

  def apply(begin: Int, end: Int, label: Int) = matrix(label, ChartHalf.chartIndex(begin, end, length))

  def spanRangeSlice(spanLength: Int, firstPos: Int = 0, end: Int = length): Range = {
    assert(spanLength > 0)
    val firstIndex: Int = ChartHalf.chartIndex(firstPos, firstPos + spanLength, length)
    assert(firstIndex >= 0, (firstPos, spanLength, length, end))
    val lastIndex = math.min(ChartHalf.chartIndex(0,spanLength+1,length), end + firstIndex)
    assert(lastIndex >= 0)
    (firstIndex + globalRowOffset until lastIndex + globalRowOffset)
  }

  def treeIndex(begin: Int, end: Int) = globalRowOffset + ChartHalf.chartIndex(begin, end, length)

  def rootIndex = treeIndex(0, length)

  def toString(structure: RuleStructure[_, _], zero: Float) = {
    (for(span <- 1 to length; begin <- 0 to length-span) yield {
      val r = ChartHalf.chartIndex(begin,begin+span,length)
      matrix(::, r).iterator.collect { case ((k, _),v)  if(v != zero) => 
        if(isBot && k == structure.root)
          throw new RuntimeException("What is the root doing in the bot chart?" + k + " " + v)
        if(span == 1 && isBot) 
          structure.termIndex.get(k) -> v
        else 
          structure.nontermIndex.get(k) -> v
      }.mkString(s"($begin,${begin+span}) ${if(isBot) "bot" else "top"} {",", ", "}")
    }).mkString("\n")
  }
}

object ChartHalf {
  @inline
  def chartIndex(begin: Int, end: Int, length: Int) = {
    val span = end-begin-1
    begin + span * length - span * (span - 1) / 2

  }
}

