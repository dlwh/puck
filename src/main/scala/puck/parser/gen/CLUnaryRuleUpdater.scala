package puck.parser.gen

import com.nativelibs4java.opencl._
import puck.linalg.CLMatrix
import java.util.zip.{ZipFile, ZipOutputStream}
import puck.util.ZipUtil

/**
 *
 *
 * @author dlwh
 */
case class CLUnaryRuleUpdater(kernels: IndexedSeq[CLKernel]) {
  def update(parent: CLMatrix[Float],
             child: CLMatrix[Float],  events: CLEvent*)(implicit queue: CLQueue) = synchronized {
    require(parent.rows == child.rows)
    require(parent.cols == child.cols)
    require(parent.majorStride == child.majorStride)
    kernels.map { k =>
      k.setArgs(parent.data.safeBuffer,  child.data.safeBuffer,
        Integer.valueOf(parent.majorStride), Integer.valueOf(parent.rows) )
      k.enqueueNDRange(queue, Array(parent.rows), events: _*)
    }

  }

  def write(name: String, out: ZipOutputStream) {
    ZipUtil.addKernelSet(out, name, kernels)
  }
}


object CLUnaryRuleUpdater {
  def read(in: ZipFile, name: String)(implicit ctxt: CLContext) = {
    CLUnaryRuleUpdater(ZipUtil.readKernelSet(in, name))
  }
}