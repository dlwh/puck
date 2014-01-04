package puck.parser

import breeze.linalg._
import puck.util.BitHacks

/**
 * TODO
 *
 * @author dlwh
 **/
trait PruningMask {
  def hasMasks: Boolean

  def isAllowedSpan(sent: Int, begin: Int, end: Int) = !hasMasks || maskForBotCell(sent, begin, end).forall(BitHacks.any)
  def isAllowedTopSpan(sent: Int, begin: Int, end: Int) = !hasMasks || maskForTopCell(sent, begin, end).forall(BitHacks.any)

  def maskForBotCell(sent: Int, begin: Int, end: Int):Option[DenseVector[Int]]
  def maskForTopCell(sent: Int, begin: Int, end: Int):Option[DenseVector[Int]]

  def ++(mask: PruningMask) = (this, mask) match {
    case (NoPruningMask, NoPruningMask) => NoPruningMask
    case (NoPruningMask, _) => throw new RuntimeException("Can't concat empty mask with nonempty masks")
    case (_, NoPruningMask) => throw new RuntimeException("Can't concat empty mask with nonempty masks")
    case (x: DenseMatrixMask, y: DenseMatrixMask) =>
      new DenseMatrixMask(DenseMatrix.horzcat(x.matrix, y.matrix), x.lengths ++ y.lengths, x.cellOffsets ++ y.cellOffsets.map(_ + x.cellOffsets.last))
  }

  def slice(fromSentence: Int, toSentence: Int):PruningMask

}

object NoPruningMask extends PruningMask {
  def hasMasks = false

  def maskForBotCell(sent: Int, begin: Int, end: Int): Option[DenseVector[Int]] = None

  def maskForTopCell(sent: Int, begin: Int, end: Int): Option[DenseVector[Int]] = None

  def slice(fromSentence: Int, toSentence: Int): PruningMask = NoPruningMask
}

case class DenseMatrixMask(matrix: DenseMatrix[Int], lengths: Array[Int], cellOffsets: Array[Int]) extends PruningMask {
  def hasMasks: Boolean = true
  assert(lengths.length == cellOffsets.length - 1)

  def maskForBotCell(sent: Int, begin: Int, end: Int): Option[DenseVector[Int]] = {
    Some(matrix(::, cellOffsets(sent) + ChartHalf.chartIndex(begin, end, lengths(sent))))
  }

  def maskForTopCell(sent: Int, begin: Int, end: Int): Option[DenseVector[Int]] = {
    Some(matrix(::, cellOffsets(sent) + (cellOffsets(sent+1) - cellOffsets(sent))/2 + ChartHalf.chartIndex(begin, end, lengths(sent))))
  }

  def slice(fromSentence: Int, toSentence: Int): PruningMask = {
    val mslice = matrix(::, cellOffsets(fromSentence) until cellOffsets(toSentence))
    new DenseMatrixMask(mslice, lengths.slice(fromSentence, toSentence), cellOffsets.slice(fromSentence, toSentence+1))
  }
}