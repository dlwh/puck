package puck.linalg.kernels

import org.scalatest.FunSuite
import com.nativelibs4java.opencl.{CLPlatform, JavaCL}
import puck.linalg.CLMatrix

/**
 * TODO
 *
 * @author dlwh
 **/
class CLMatrixSliceCopyTest extends FunSuite {


  test("sliceCopy") {
    implicit val context = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU)
    implicit val queue = context.createDefaultOutOfOrderQueueIfPossible()
    val sliceCopy = CLMatrixSliceCopy()
    val mat = CLMatrix.zeros[Float](10,10)
    val mat2 = CLMatrix.zeros[Float](10,3)
    mat2 := -2.0f
    mat2(::, 1) := -3.0f
    val ev = sliceCopy.sliceCopy(mat(::, 0 until 3), mat2, Array(1, 1, 2), 3)
    ev.waitFor
    val dense = mat.toDense
    assert(dense(::, 0).valuesIterator.forall(_ == -3.0f), dense)
    assert(dense(::, 1).valuesIterator.forall(_ == -3.0f), dense)
    assert(dense(::, 2).valuesIterator.forall(_ == -2.0f), dense)
  }

  test("sliceCopyOut") {
    implicit val context = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU)
    implicit val queue = context.createDefaultOutOfOrderQueueIfPossible()
    val sliceCopy = CLMatrixSliceCopy()
    val mat = CLMatrix.zeros[Float](10,10)
    val mat2 = CLMatrix.zeros[Float](10,3)
    mat2 := -2.0f
    mat2(::, 1) := -3.0f
    mat2(::, 2) := -0.0f
    val ev = sliceCopy.sliceCopyOut(mat(::, 0 until 3), Array(1, 2, 0), 3, mat2)
    ev.waitFor
    val dense = mat.toDense
    assert(dense(::, 0).valuesIterator.forall(_ == -0.0f), dense)
    assert(dense(::, 1).valuesIterator.forall(_ == -2.0f), dense)
    assert(dense(::, 2).valuesIterator.forall(_ == -3.0f), dense)
  }


}
