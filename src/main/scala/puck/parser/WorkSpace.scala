package puck.parser

import com.nativelibs4java.opencl._
import puck.linalg.CLMatrix
import java.io.Closeable
import breeze.collection.mutable.TriangularArray
import scala.collection.mutable.ArrayBuffer

/**
 *
 *
 * @author dlwh
 */
class WorkSpace(val numWorkCells: Int,
                val numChartCells: Int,
                val cellSize: Int,
                val maskSize: Int)(implicit context: CLContext, queue: CLQueue) extends Closeable {

  def getBatches[W](sentences: IndexedSeq[IndexedSeq[W]], masks: PruningMask = NoPruningMask): IndexedSeq[Batch[W]] = {
    val result = ArrayBuffer[Batch[W]]()
    var current = ArrayBuffer[IndexedSeq[W]]()
    var currentCellTotal = 0
    for ( (s, i) <- sentences.zipWithIndex) {
      currentCellTotal += TriangularArray.arraySize(s.length) * 2
      if (currentCellTotal > devInside.cols) {
        currentCellTotal -= TriangularArray.arraySize(s.length) * 2
        assert(current.nonEmpty)
        result += createBatch(current, masks.slice(i - current.length, i))
        currentCellTotal = TriangularArray.arraySize(s.length) * 2
        current = ArrayBuffer()
      }
      current += s
    }


    if (current.nonEmpty) {
      result += createBatch(current, masks.slice(sentences.length - current.length, sentences.length))
    }
    result
  }


  private def createBatch[W](sentences: IndexedSeq[IndexedSeq[W]], masks: PruningMask): Batch[W] = {
    val batch = Batch[W](sentences, devInside, devOutside, masks)
    println(f"Batch size of ${sentences.length}, ${batch.numCellsUsed} cells used, total inside ${batch.numCellsUsed * cellSize * 4.0/1024/1024}%.2fM  ")
    batch
  }


  // On the Device side we have 5 data matrices:
  // One is where we calculate P = L * R * rules, for fixed spans and split points (the "bot")
  val devParent = new CLMatrix[Float]( numWorkCells, cellSize)
  // One is the L part of the above
  val devLeft = new CLMatrix[Float]( numWorkCells, cellSize)
  // Another is the R part.
  val devRight = new CLMatrix[Float]( numWorkCells, cellSize)

  // finally, we have the array of parse charts
  val devInside = new CLMatrix[Float](cellSize, numChartCells/2)
  val devOutside = new CLMatrix[Float](cellSize, numChartCells/2)
  val maskCharts = new CLMatrix[Int](maskSize, numChartCells/2)
  val devInsideScale, devOutsideScale = context.createFloatBuffer(CLMem.Usage.InputOutput, numChartCells / 2)

  // work queue stuff
  val pArray, lArray, rArray = new Array[Int](numWorkCells)
//  val parentQueue, leftQueue, rightQueue = context.createIntBuffer(CLMem.Usage.Input, numWorkCells)
  val offsetBuffer = context.createIntBuffer(CLMem.Usage.Input, numWorkCells)
  val splitPointOffsets = new Array[Int](numWorkCells+1)
  val devParentPtrs = context.createIntBuffer(CLMem.Usage.Input, numWorkCells)
  val devSplitPointOffsets = context.createIntBuffer(CLMem.Usage.Input, numWorkCells + 1)


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

object WorkSpace {
  def allocate( cellSize: Int, maskSize: Int, maxAllocSize: Long = -1, ratioOfChartsToP: Int = 7)(implicit context: CLContext, queue: CLQueue): WorkSpace = {
    var maxMemToUse =  context.getDevices.head.getGlobalMemSize
    if(maxAllocSize >= 0) maxMemToUse = math.min(maxAllocSize, maxMemToUse)

    val sizeOfFloat = 4
    val fractionOfMemoryToUse = 0.8 // slack!
    val maxSentencesPerBatch: Long = 400 // just for calculation's sake
    val sizeToAllocate = (maxMemToUse * fractionOfMemoryToUse).toInt  - maxSentencesPerBatch * 3 * 4;
    val maxPossibleNumberOfCells = ((sizeToAllocate / sizeOfFloat) / (cellSize + 4 + maskSize)).toInt // + 4 for each kind of offset
    // We want numGPUCells and numGPUChartCells to be divisible by 16, so that we get aligned strided access:
    //       On devices of compute capability 1.0 or 1.1, the k-th thread in a half warp must access the
    //       k-th word in a segment aligned to 16 times the size of the elements being accessed; however,
    //       not all threads need to participate... If sequential threads in a half warp access memory that is
    //       sequential but not aligned with the segments, then a separate transaction results for each element
    //       requested on a device with compute capability 1.1 or lower.
    val numberOfUnitsOf32 = maxPossibleNumberOfCells / 32
    // average sentence length of sentence, let's say n.
    // for the gpu charts, we'll need (n choose 2) * 2 * 2 =
    // for the "P/L/R" parts, the maximum number of relaxations (P = L * R * rules) for a fixed span
    // in a fixed sentence is (n/2)^2= n^2/4.
    // Take n = 32, then we want our P/L/R arrays to be of the ratio (3 * 256):992 \approx 3/4 (3/4 exaclty if we exclude the - n term)
    // doesn't quite work the way we want (outside), so we'll bump the number to 4/5
    val baseSize = numberOfUnitsOf32 / (3 + ratioOfChartsToP)
    val extra = numberOfUnitsOf32 % (3 + ratioOfChartsToP)
    val plrSize = baseSize
    // TODO, can probably do a better job of these calculations?
    val (workCells, chartCells) = (plrSize * 32, (baseSize * ratioOfChartsToP + extra) * 32)

    new WorkSpace(workCells, chartCells, cellSize, maskSize)

  }
}
