package puck.util

import scala.virtualization.lms.common.{NumericOps, ArrayOps}
import trochee.kernels.{Local, KernelOps}

/**
 *
 *
 * @author dlwh
trait ReductionOperations extends KernelOps with ArrayOps with NumericOps {
  def workgroupSum(arr: Rep[Array[Int] with Local], offset: Rep[Int],
                   length: Rep[Int],
                   targetLength: Rep[Int],
                   tid: Rep[Int],
                   wgSize: Rep[Int]) = {
    if(length % targetLength != 0)
      printf("...\n")
    var len = length
    while(len >= targetLength * 2) {
      memFence(BarrierType.Local)
      sumArrays(arr, offset, arr, offset + len / 2, len / 2, tid, wgSize)
      len /= 2
    }
    memFence(BarrierType.Local)
  }

  def sumArrays(dest: Rep[Array[Int]], destOff: Rep[Int],
                src: Rep[Array[Int]], srcOff: Rep[Int],
                len: Rep[Int],
                tid: Rep[Int],
                wgSize: Rep[Int]):Rep[Unit] = {
    val div = len / wgSize
    val mod = len % wgSize
    for(i <- 0 until div) {
      dest(destOff + wgSize * i + tid) += src(srcOff + wgSize * i + tid)
    }
    if(tid < mod) {
      dest(destOff + div + tid) += src(srcOff + div + tid)
    }
  }

}
 */
