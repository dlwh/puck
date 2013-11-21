package puck.parser

import puck.linalg.CLMatrix
import breeze.collection.mutable.TriangularArray

class ParseChart(val length: Int, botMat: CLMatrix[Float], topMat: CLMatrix[Float]) {
  val top = new ChartHalf(length, topMat, false)
  val bot = new ChartHalf(length, botMat, true)
}

class ChartHalf(val length: Int, val matrix: CLMatrix[Float], isBot: Boolean) {
  val globalRowOffset = matrix.offset / matrix.rows

  def apply(begin: Int, end: Int, label: Int) = {
    matrix(label, ChartHalf.chartIndex(begin, end, length))
  }


  def apply(begin: Int, end: Int) = {
    matrix(::, ChartHalf.chartIndex(begin, end, length)).toDense
  }

  def spanRangeSlice(spanLength: Int, firstPos: Int = 0, end: Int = length): Array[Int] = {
    assert(spanLength > 0)
    val firstIndex: Int = ChartHalf.chartIndex(firstPos, firstPos + spanLength, length)
    assert(firstIndex >= 0, (firstPos, spanLength, length, end))
    val lastIndex = math.min(ChartHalf.chartIndex(0,spanLength+1,length), end + firstIndex)
    assert(lastIndex >= 0)
    Array.range(firstIndex + globalRowOffset, lastIndex + globalRowOffset)
  }

  def cellOffset(begin: Int, end: Int) = globalRowOffset + ChartHalf.chartIndex(begin, end, length)

  def rootIndex = cellOffset(0, length)

  def cellString(begin: Int, end: Int, structure: RuleStructure[_, _], zero: Float) = {
    val span = end-begin
    val r = ChartHalf.chartIndex(begin,end,length)
    matrix(::, r).iterator.collect { case ((k, _),v)  if(v != zero) => 
      if(isBot && k == structure.root && span > 1)
        println("What is the root doing in the bot chart?" + k + " " + v + " " + (begin, end))
      if(span == 1 && isBot) 
        structure.termIndex.get(k) -> v
      else 
        structure.nontermIndex.get(k) -> v
    }.mkString(s"$r ${cellOffset(begin, end)} ($begin,$end) ${if(isBot) "bot" else "top"} {",", ", "}")

  }

  def toString(structure: RuleStructure[_, _], zero: Float) = {
    (for(span <- 1 to length; begin <- 0 to length-span) yield cellString(begin,begin+span,structure,zero)).mkString("\n")
  }
}

object ChartHalf {
  @inline
  def chartIndex(begin: Int, end: Int, length: Int) = {
    val span = end-begin-1
    begin + span * length - span * (span - 1) / 2

  }
}

