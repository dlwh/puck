package puck.parser

import breeze.linalg._
import puck.util.BitHacks
import org.bridj.Pointer

/**
 * TODO
 *
 * @author dlwh
 **/
trait PruningMask {
  def getIScales: Pointer[java.lang.Float]
  def getOScales: Pointer[java.lang.Float]

  def hasMasks: Boolean

  def isAllowedSpan(sent: Int, begin: Int, end: Int) = !hasMasks || maskForBotCell(sent, begin, end).forall(BitHacks.any)
  def isAllowedTopSpan(sent: Int, begin: Int, end: Int) = !hasMasks || maskForTopCell(sent, begin, end).forall(BitHacks.any)

  def insideScaleFor(sent: Int, begin: Int, end: Int): Float
  def insideTopScaleFor(sent: Int, begin: Int, end: Int): Float
  def outsideScaleFor(sent: Int, begin: Int, end: Int): Float
  def outsideTopScaleFor(sent: Int, begin: Int, end: Int): Float

  def maskForBotCell(sent: Int, begin: Int, end: Int):Option[DenseVector[Int]]
  def maskForTopCell(sent: Int, begin: Int, end: Int):Option[DenseVector[Int]]

  def ++(mask: PruningMask) = (this, mask) match {
    case (NoPruningMask, NoPruningMask) => NoPruningMask
    case (NoPruningMask, _) => throw new RuntimeException("Can't concat empty mask with nonempty masks")
    case (_, NoPruningMask) => throw new RuntimeException("Can't concat empty mask with nonempty masks")
    case (x: DenseMatrixMask, y: DenseMatrixMask) =>
      new DenseMatrixMask(DenseMatrix.horzcat(x.matrix, y.matrix),
        DenseVector.vertcat(x.insideScale, y.insideScale),
        DenseVector.vertcat(x.outsideScale, y.outsideScale),
        x.lengths ++ y.lengths,
        x.cellOffsets.dropRight(1) ++ y.cellOffsets.map(_ + x.cellOffsets.last))
  }

  def slice(fromSentence: Int, toSentence: Int):PruningMask

}

object NoPruningMask extends PruningMask {
  def hasMasks = false


  def insideScaleFor(sent: Int, begin: Int, end: Int): Float = 0.0f
  def outsideScaleFor(sent: Int, begin: Int, end: Int): Float = 0.0f


  def insideTopScaleFor(sent: Int, begin: Int, end: Int): Float = 0.0f


  def outsideTopScaleFor(sent: Int, begin: Int, end: Int): Float = 0.0f

  def maskForBotCell(sent: Int, begin: Int, end: Int): Option[DenseVector[Int]] = None

  def maskForTopCell(sent: Int, begin: Int, end: Int): Option[DenseVector[Int]] = None

  def slice(fromSentence: Int, toSentence: Int): PruningMask = NoPruningMask


  def getIScales: Pointer[java.lang.Float] = Pointer.pointerToFloat(0)
  def getOScales: Pointer[java.lang.Float] = Pointer.pointerToFloat(0)
}

case class DenseMatrixMask(matrix: DenseMatrix[Int],
                           insideScale: DenseVector[Float],
                           outsideScale: DenseVector[Float],
                           lengths: Array[Int], cellOffsets: Array[Int]) extends PruningMask {
  lazy val getIScales: Pointer[java.lang.Float] = Pointer.pointerToFloats(insideScale.toArray:_*)
  lazy val getOScales: Pointer[java.lang.Float] = Pointer.pointerToFloats(outsideScale.toArray:_*)

  def insideScaleFor(sent: Int, begin: Int, end: Int) = insideScale(cellOffsets(sent) + ChartHalf.chartIndex(begin, end, lengths(sent)))
  def outsideScaleFor(sent: Int, begin: Int, end: Int) = outsideScale(cellOffsets(sent) + ChartHalf.chartIndex(begin, end, lengths(sent)))

  def insideTopScaleFor(sent: Int, begin: Int, end: Int) = insideScale((cellOffsets(sent + 1) + cellOffsets(sent))/2 + ChartHalf.chartIndex(begin, end, lengths(sent)))
  def outsideTopScaleFor(sent: Int, begin: Int, end: Int) = outsideScale((cellOffsets(sent + 1) + cellOffsets(sent))/2 + ChartHalf.chartIndex(begin, end, lengths(sent)))


  def hasMasks: Boolean = true
  assert(lengths.length == cellOffsets.length - 1)
  assert(matrix.cols == cellOffsets.last, matrix.cols + " " + cellOffsets.last)
  assert(insideScale.length == cellOffsets.last)
  assert(outsideScale.length == cellOffsets.last)

  def maskForBotCell(sent: Int, begin: Int, end: Int): Option[DenseVector[Int]] = {
    val index: Int = ChartHalf.chartIndex(begin, end, lengths(sent))
    assert(cellOffsets(sent) + index < cellOffsets(sent+1))
    Some(matrix(::, cellOffsets(sent) + index))
  }

  def maskForTopCell(sent: Int, begin: Int, end: Int): Option[DenseVector[Int]] = {
    Some(matrix(::, cellOffsets(sent) + (cellOffsets(sent+1) - cellOffsets(sent))/2 + ChartHalf.chartIndex(begin, end, lengths(sent))))
  }

  def slice(fromSentence: Int, toSentence: Int): PruningMask = {
    val mslice = matrix(::, cellOffsets(fromSentence) until cellOffsets(toSentence))
    val islice = insideScale.slice(cellOffsets(fromSentence), cellOffsets(toSentence))
    val oslice = outsideScale.slice(cellOffsets(fromSentence), cellOffsets(toSentence))
    new DenseMatrixMask(mslice, islice, oslice,
      lengths.slice(fromSentence, toSentence), cellOffsets.slice(fromSentence, toSentence+1).map(_ - cellOffsets(fromSentence)))
  }
}