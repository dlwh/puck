package puck.parser

import puck._
import com.nativelibs4java.opencl.{CLContext, CLQueue, CLEvent, CLKernel}
import puck.linalg.CLMatrix
import scala.Array
import java.util.zip._
import puck.util._

/**
 *
 *
 * @author dlwh
 */
case class CLUnaryRuleUpdater(kernels: IndexedSeq[CLKernel]) {
  def update(profiler: CLProfiler,
             parent: CLMatrix[Float],
             child: CLMatrix[Float],  events: CLEvent*)(implicit queue: CLQueue) = synchronized {
    require(parent.rows == child.rows)
    require(parent.cols == child.cols)
    require(parent.majorStride == child.majorStride)
    kernels.map { k =>
      k.setArgs(parent.data.safeBuffer,  child.data.safeBuffer,
        Integer.valueOf(parent.majorStride), Integer.valueOf(parent.rows) )
      k.enqueueNDRange(queue, Array(parent.rows), events: _*) profileIn profiler
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