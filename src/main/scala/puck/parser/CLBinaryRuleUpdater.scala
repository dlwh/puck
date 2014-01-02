package puck.parser

import com.nativelibs4java.opencl._
import puck._
import puck.linalg.CLMatrix
import java.util.zip.{ZipFile, ZipInputStream, ZipOutputStream}
import puck.util.{CLProfiler, ZipUtil}
import scala.collection.JavaConverters._
import org.bridj.Pointer
import java.util
import breeze.linalg.DenseVector

/**
 *
 *
 * @author dlwh
 */
case class CLBinaryRuleUpdater(kernels: IndexedSeq[RuleKernel],
                               globalSize: Array[Int],
                               wgSize: Array[Int], extra: Option[Array[Int]] = None) {
  def this(kernels: java.util.List[RuleKernel], globalSize: Array[Int], wgSize: Array[Int]) = this(kernels.asScala.toIndexedSeq, globalSize, wgSize)

  private val buffer = extra.map(arr => kernels.head.kernels.head.getProgram.getContext.createIntBuffer(CLMem.Usage.Input, Pointer.pointerToInts(arr:_*), true))

  def numKernelBlocks = kernels.length

  def update(block: IndexedSeq[Int], profiler: CLProfiler, parent: CLMatrix[Float], parentPointers: CLBuffer[Int],
             left: CLMatrix[Float],  leftPointers: CLBuffer[Int],
             right: CLMatrix[Float],  rightPointers: CLBuffer[Int],
             masks: CLMatrix[Int], events: CLEvent*)(implicit queue: CLQueue) = synchronized {
//    require(parent.rows <= parentPointers.getElementCount)
    require(left.rows <= leftPointers.getElementCount)
    require(right.rows <= rightPointers.getElementCount)
    require(parent.rows == left.cols)
    require(parent.cols > parentPointers.read(queue).toArray.take(left.rows).map(_.toInt).max)
//    require(parent.rows == left.rows)
//    require(parent.cols == left.cols)
//    require(parent.majorStride == left.majorStride)
    require(left.rows == right.rows)
    require(left.cols == right.cols)
    require(left.majorStride == right.majorStride)
    block.flatMap(kernels(_).kernels).foldLeft(events) { (ev, k) =>
      k.setArgs(parent.data.safeBuffer, parentPointers,
        left.data.safeBuffer, leftPointers,
        right.data.safeBuffer,  rightPointers,
        masks.data.safeBuffer,
        Integer.valueOf(left.majorStride), Integer.valueOf(left.rows) )
      buffer.foreach(buf => k.setArg(7, buf))

      val evv = k.enqueueNDRange(queue, globalSize, wgSize, ev:_*) profileIn profiler
      IndexedSeq(evv)
    }

  }

  def write(name: String, out: ZipOutputStream) {
    ZipUtil.serializedEntry(out, s"$name/numKernels", Integer.valueOf(kernels.length))
    for(i <- 0 until kernels.length) {
      kernels(i).write(s"$name/$i", out)
    }
    ZipUtil.serializedEntry(out, s"$name/globalSize", globalSize)
    ZipUtil.serializedEntry(out, s"$name/wgSize", wgSize)
    ZipUtil.serializedEntry(out, s"$name/extra", extra)
  }
}


object CLBinaryRuleUpdater {
  def read(in: ZipFile, name: String)(implicit ctxt: CLContext) = {
    val x = ZipUtil.deserializeEntry[Integer](in.getInputStream(in.getEntry(s"$name/numKernels")))
    val kernels = for(i <- 0 until x.intValue()) yield RuleKernel.read(in, s"$name/$i")

    val globalSize = ZipUtil.deserializeEntry[Array[Int]](in.getInputStream(in.getEntry(s"$name/globalSize")))
    val wgSize = ZipUtil.deserializeEntry[Array[Int]](in.getInputStream(in.getEntry(s"$name/wgSize")))
    val extra = ZipUtil.deserializeEntry[Option[Array[Int]]](in.getInputStream(in.getEntry(s"$name/extra")))
    CLBinaryRuleUpdater(kernels, globalSize, wgSize, extra)
  }
}

case class RuleKernel(kernels: IndexedSeq[CLKernel], parents: DenseVector[Int]) {
  def write(name: String, out: ZipOutputStream) {
    ZipUtil.addKernelSet(out, s"$name", kernels)
    ZipUtil.serializedEntry(out, s"$name/bits", parents)
  }
}

object RuleKernel {

  def read(in: ZipFile, name: String)(implicit context: CLContext) = {
    val kernels = ZipUtil.readKernelSet(in, name)
    val parents = ZipUtil.deserializeEntry[DenseVector[Int]](in.getInputStream(in.getEntry(s"$name/bits")))
    RuleKernel(kernels, parents)
  }
}