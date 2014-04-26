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
import puck.parser.gen.{CLWorkQueueKernels, HasParent, IndexedBinaryRule}

/**
 *
 *
 * @author dlwh
 */
case class CLBinaryRuleUpdater(kernels: IndexedSeq[RuleKernel],
                               enqueuer: CLWorkQueueKernels,
                               directWriteToChart: Boolean,
                               extra: Option[Array[Int]] = None) {
  def this(kernels: java.util.List[RuleKernel], enqueuer: CLWorkQueueKernels, directWrite: Boolean) = this(kernels.asScala.toIndexedSeq, enqueuer, directWrite)

  private val buffer = extra.map(arr => kernels.head.kernels.head.getProgram.getContext.createIntBuffer(CLMem.Usage.Input, Pointer.pointerToInts(arr:_*), true))

  def numKernelBlocks = kernels.length

  def update(block: IndexedSeq[Int], profiler: CLProfiler#EventTimer,
             parent: CLMatrix[Float], parentScale: CLBuffer[Float], parentPointers: CLBuffer[Int], parentOff: Integer,
             left: CLMatrix[Float], leftScale: CLBuffer[Float], leftPointers: CLBuffer[Int], leftOff: Integer,
             right: CLMatrix[Float], rightScale: CLBuffer[Float], rightPointers: CLBuffer[Int], rightOff: Integer,
             masks: CLMatrix[Int], events: CLEvent*)(implicit queue: CLQueue) = synchronized {
    require(!directWriteToChart || parent.rows == left.cols)
    require(directWriteToChart || parent.cols == left.cols)
    require(directWriteToChart || parent.rows == left.rows)
    require(directWriteToChart || parent.majorStride == left.majorStride)
//    require(parent.rows == left.rows)
//    require(parent.cols == left.cols)
    require(left.rows == right.rows)
    require(left.cols == right.cols)
    require(left.majorStride == right.majorStride)

    block.foldLeft(events) { (ev, i) =>
      kernels(i).kernels.foldLeft(ev) { (ev, k) =>
        k.setArgs(parent.data.safeBuffer, parentScale, parentPointers, parentOff,
          left.data.safeBuffer, leftScale, leftPointers, leftOff,
          right.data.safeBuffer, rightScale, rightPointers, rightOff,
          masks.data.safeBuffer,
          Integer.valueOf(left.majorStride), Integer.valueOf(left.rows) )
        buffer.foreach(buf => k.setArg(7, buf))

        val evv = k.enqueueNDRange(queue, kernels(i).globalSize, kernels(i).wgSize, ev:_*) profileIn profiler
        IndexedSeq(evv)
      }
    }

  }

  def write(name: String, out: ZipOutputStream) {
    ZipUtil.serializedEntry(out, s"$name/numKernels", Integer.valueOf(kernels.length))
    for(i <- 0 until kernels.length) {
      kernels(i).write(s"$name/$i", out)
    }
    enqueuer.write(s"$name/enqueuer", out)
    ZipUtil.serializedEntry(out, s"$name/extra", extra)
    ZipUtil.serializedEntry(out, s"$name/directWrite", java.lang.Boolean.valueOf(directWriteToChart))
  }
}


object CLBinaryRuleUpdater {
  def read(in: ZipFile, name: String)(implicit ctxt: CLContext) = {
    val x = ZipUtil.deserializeEntry[Integer](in.getInputStream(in.getEntry(s"$name/numKernels")))
    val kernels = for(i <- 0 until x.intValue()) yield RuleKernel.read(in, s"$name/$i")
    val queue = CLWorkQueueKernels.read(s"$name/enqueuer", in)

    val extra = ZipUtil.deserializeEntry[Option[Array[Int]]](in.getInputStream(in.getEntry(s"$name/extra")))
    val directWrite = ZipUtil.deserializeEntry[java.lang.Boolean](in.getInputStream(in.getEntry(s"$name/directWrite")))
    CLBinaryRuleUpdater(kernels, queue, directWrite.booleanValue(), extra)
  }
}

case class RuleKernel(kernels: IndexedSeq[CLKernel],
                      rules: IndexedSeq[HasParent[_, _]],
                      globalSize: Array[Int],
                      wgSize: Array[Int],
                      parents: DenseVector[Int]) {
  def write(name: String, out: ZipOutputStream) {
    ZipUtil.addKernelSet(out, s"$name", kernels)
    ZipUtil.serializedEntry(out, s"$name/globalSize", globalSize)
    ZipUtil.serializedEntry(out, s"$name/wgSize", wgSize)
    ZipUtil.serializedEntry(out, s"$name/bits", parents)
    ZipUtil.serializedEntry(out, s"$name/rules", rules)
  }
}

object RuleKernel {

  def read(in: ZipFile, name: String)(implicit context: CLContext) = {
    val kernels = ZipUtil.readKernelSet(in, name)
    val globalSize = ZipUtil.deserializeEntry[Array[Int]](in.getInputStream(in.getEntry(s"$name/globalSize")))
    val wgSize = ZipUtil.deserializeEntry[Array[Int]](in.getInputStream(in.getEntry(s"$name/wgSize")))
    val parents = ZipUtil.deserializeEntry[DenseVector[Int]](in.getInputStream(in.getEntry(s"$name/bits")))
    val rules = ZipUtil.deserializeEntry[IndexedSeq[HasParent[_, _]]](in.getInputStream(in.getEntry(s"$name/rules")))
    RuleKernel(kernels, rules, globalSize, wgSize, parents)
  }
}