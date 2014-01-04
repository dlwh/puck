package puck.parser

import breeze.linalg.DenseMatrix
import breeze.collection.mutable.TriangularArray
import puck.linalg.CLMatrix
import puck.util.BitHacks

/**
 * TODO
 *
 * @author dlwh
 **/
private[parser] case class Batch[W](lengthTotals: Array[Int],
                                    cellOffsets: Array[Int],
                                    sentences: IndexedSeq[IndexedSeq[W]],
                                    devInside: CLMatrix[Float],
                                    devOutside: CLMatrix[Float],
                                    masks: Option[DenseMatrix[Int]]) {

  def totalLength = lengthTotals.last
  def numSentences = sentences.length
  def numCellsUsed: Int = cellOffsets.last
  val maxLength = sentences.map(_.length).max
  assert(numCellsUsed <= devInside.cols)
  assert(masks.forall(m => m.cols == numCellsUsed))

  def isAllowedSpan(sent: Int, begin: Int, end: Int) = botMaskFor(sent, begin, end).forall(BitHacks.any)

  def botMaskFor(sent: Int, begin: Int, end: Int) = masks.map(m =>  m(::, insideCharts(sent).bot.cellOffset(begin, end)))
  def topMaskFor(sent: Int, begin: Int, end: Int) = masks.map(m =>  m(::, insideCharts(sent).top.cellOffset(begin, end)))

  def insideBotCell(sent: Int, begin: Int, end: Int) = insideCharts(sent).bot.cellOffset(begin, end)
  def insideTopCell(sent: Int, begin: Int, end: Int) = insideCharts(sent).top.cellOffset(begin, end)

  def outsideBotCell(sent: Int, begin: Int, end: Int) = outsideCharts(sent).bot.cellOffset(begin, end)
  def outsideTopCell(sent: Int, begin: Int, end: Int) = outsideCharts(sent).top.cellOffset(begin, end)

  def hasMasks = masks.nonEmpty

  lazy val insideCharts = for (i <- 0 until numSentences) yield {
    val numCells = (cellOffsets(i+1)-cellOffsets(i))/2
    assert(numCells == TriangularArray.arraySize(sentences(i).length))
    val chart = new ParseChart(sentences(i).length, devInside(::, cellOffsets(i) until (cellOffsets(i) + numCells)), devInside(::, cellOffsets(i) + numCells until cellOffsets(i+1)))
    chart
  }

  lazy val outsideCharts = {
    for (i <- 0 until numSentences) yield {

      val numCells = (cellOffsets(i+1)-cellOffsets(i))/2
      assert(numCells == TriangularArray.arraySize(sentences(i).length))
      val botBegin = cellOffsets(i)
      val botEnd = botBegin + numCells
      val topBegin = botEnd
      val topEnd = topBegin + numCells
      assert(topEnd <= devOutside.cols)
      val chart = new ParseChart(sentences(i).length, devOutside(::, botBegin until botEnd), devOutside(::, topBegin until topEnd))
      chart
    }
  }

}
