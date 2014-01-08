package puck.parser.gen

import puck.parser.{SymId, RuleSemiring, RuleStructure}
import scala.collection.JavaConverters._
import puck.linalg.CLMatrix
import com.nativelibs4java.opencl._
import org.bridj.Pointer
import java.util.zip.{ZipOutputStream, ZipFile}
import puck.util.ZipUtil
import scala.Array
import puck.PointerFreer

/**
 * TODO
 *
 * @author dlwh
 **/
case class CLScalingKernels(maskSize: Int, getMasksKernel: CLKernel) {


  def write(out: ZipOutputStream) {
    ZipUtil.addKernel(out, "computeMasksKernel", getMasksKernel)
    ZipUtil.serializedEntry(out, "MasksInts", Array(maskSize))
  }

  def getMasks(masks: CLMatrix[Int],
               inside: CLMatrix[Float],
               outside: CLMatrix[Float],
               firstOutside: Int,
               chartIndices: Array[Int],
               lengths: Array[Int],
               root: Int, threshold: Float,
               events: CLEvent*)(implicit queue: CLQueue):CLEvent = {
    require(masks.rows == maskSize, masks.rows + " " + maskSize)
    require(masks.cols == inside.cols)
    require(masks.cols == outside.cols)
    queue.finish()



    val ptrCI = Pointer.pointerToArray[java.lang.Integer](chartIndices)
    val intBufferCI = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, chartIndices.length)
    val evCI = intBufferCI.write(queue, 0, chartIndices.length, ptrCI, false, events:_*)

    val ptrL = Pointer.pointerToArray[java.lang.Integer](lengths)
    val intBufferL = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, lengths.length)
    val evL = intBufferL.write(queue, 0, lengths.length, ptrL, false, events:_*)

    getMasksKernel.setArgs(masks.data.safeBuffer,
      inside.data.safeBuffer, outside.data.safeBuffer, intBufferCI, intBufferL,
      Integer.valueOf(chartIndices(chartIndices.length-1)), Integer.valueOf(inside.rows),
      Integer.valueOf(root), java.lang.Float.valueOf(threshold))
    //, LocalSize.ofIntArray(fieldSize * groupSize * 5))

    val ev = getMasksKernel.enqueueNDRange(queue, Array(chartIndices.length-1, 1), Array(1, 1), evCI, evL)
    queue.finish()
    PointerFreer.enqueue(ptrCI.release(), ev)
    PointerFreer.enqueue(intBufferCI.release(), ev)

    PointerFreer.enqueue(ptrL.release(), ev)
    PointerFreer.enqueue(intBufferL.release(), ev)
    ev
  }

}


