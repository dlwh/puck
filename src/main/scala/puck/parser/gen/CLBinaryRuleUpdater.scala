package puck.parser.gen

import com.nativelibs4java.opencl._
import puck.linalg.CLMatrix
import java.util.zip.{ZipFile, ZipInputStream, ZipOutputStream}
import puck.util.ZipUtil

/**
 *
 *
 * @author dlwh
 */
case class CLBinaryRuleUpdater(kernels: IndexedSeq[CLKernel], globalSize: Array[Int], wgSize: Array[Int]) {
  def update(parent: CLMatrix[Float], parentPointers: CLBuffer[Int],
             left: CLMatrix[Float], right: CLMatrix[Float],
             masks: CLMatrix[Int], events: CLEvent*)(implicit queue: CLQueue) = synchronized {
    require(parent.rows <= parentPointers.getElementCount)
    require(parent.rows == left.rows)
    require(parent.cols == left.cols)
    require(parent.majorStride == left.majorStride)
    require(parent.rows == right.rows)
    require(parent.cols == right.cols)
    require(parent.majorStride == right.majorStride)
    kernels.map { k =>
      k.setArgs(parent.data.safeBuffer, parentPointers,
        left.data.safeBuffer, right.data.safeBuffer,
        masks.data.safeBuffer,
        Integer.valueOf(parent.majorStride), Integer.valueOf(parent.rows) )
      k.enqueueNDRange(queue, globalSize, wgSize, events: _*)
    }

  }

  def write(name: String, out: ZipOutputStream) {
    ZipUtil.addKernelSet(out, name, kernels)
    ZipUtil.serializedEntry(out, s"$name/globalSize", globalSize)
    ZipUtil.serializedEntry(out, s"$name/wgSize", wgSize)
  }
}


object CLBinaryRuleUpdater {
  def read(in: ZipFile, name: String)(implicit ctxt: CLContext) = {
    val globalSize = ZipUtil.deserializeEntry[Array[Int]](in.getInputStream(in.getEntry(s"$name/globalSize")))
    val wgSize = ZipUtil.deserializeEntry[Array[Int]](in.getInputStream(in.getEntry(s"$name/wgSize")))
    CLBinaryRuleUpdater(ZipUtil.readKernelSet(in, name), globalSize, wgSize)
  }
}