package puck.parser

import com.nativelibs4java.opencl._
import puck.linalg.CLMatrix

/**
 *
 *
 * @author dlwh
 */
class WorkSpace(numWorkCells: Int, numChartCells: Int, cellSize: Int)(implicit context: CLContext, queue: CLQueue) extends AutoCloseable {
  println(numWorkCells,numChartCells, cellSize)

  val pArray, lArray, rArray = new Array[Int](numWorkCells)
  val splitPointOffsets = new Array[Int](numWorkCells+1)
  val devParentPtrs = context.createIntBuffer(CLMem.Usage.Input, numWorkCells)


  val devParent = new CLMatrix[Float]( numWorkCells, cellSize)
  val devLeft = new CLMatrix[Float]( numWorkCells, cellSize)
  val devRight = new CLMatrix[Float]( numWorkCells, cellSize)

  val devInside = new CLMatrix[Float](cellSize, numChartCells/2)
  val devOutside = new CLMatrix[Float](cellSize, numChartCells/2)

  def close() = release()


  def release() {
    devParent.release()
    devLeft.release()
    devRight.release()
    devInside.release()
    devOutside.release()

    devParentPtrs.release()
  }


}

