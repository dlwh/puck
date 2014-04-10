package puck.util

import org.scalatest.FunSuite
import com.nativelibs4java.opencl.{CLPlatform, JavaCL}
import scala.collection.immutable.IndexedSeq

/**
 *
 *
 * @author dlwh
 */

class CLScanTest extends FunSuite {

  test("scan") {
    implicit val context = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU)
    implicit val queue = context.createDefaultOutOfOrderQueueIfPossible()
    val kernel = CLScan.make(context)

    val n = 1000

    val gpuScan: IndexedSeq[Int] = kernel.scan(Array.range(0, n)).toIndexedSeq
    val cpuScan = Array.range(0, n).scan(0)(_ + _).drop(1).toIndexedSeq
    assert(gpuScan === cpuScan, (0 until (n-1)).map(i => (gpuScan(i), cpuScan(i), i)))
  }
}
