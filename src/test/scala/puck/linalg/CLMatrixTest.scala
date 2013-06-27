package puck.linalg

import breeze.linalg._
import org.scalatest._
import com.nativelibs4java.opencl._

/**
 * TODO
 *
 * @author dlwh
 **/
class CLMatrixTest extends FunSuite {

  test("Matrix Assignment and such") {
    implicit val context = JavaCL.createBestContext()
    implicit val queue = context.createDefaultOutOfOrderQueueIfPossible()
    val mat = CLMatrix.zeros[Float](10,10)
    val mat2 = CLMatrix.zeros[Float](10,3)
    mat2 := 1.0f
    assert(mat2(0,1) === 1.0f)
    mat(::, 0 until 3) := mat2
    assert(mat(0,1) === 1.0f)
    assert(mat(5,1) === 1.0f)
    assert(mat(0,8) === 0.0f)
    assert(mat(5,9) === 0.0f)
    mat2 := 3.0f
    mat(1 until 6, 0 until 3) := mat2(0 until 5, ::)
    assert(mat(0,1) === 1.0f)
    assert(mat(1,1) === 3.0f)
    assert(mat(5,1) === 3.0f)
    assert(mat(6,1) === 1.0f)
    assert(mat(0,8) === 0.0f)
    assert(mat(5,9) === 0.0f)
    mat2 := 4.0f
    mat(2 until 7, 3 to 4) := mat2(0 until 5, 1 to 2)
    assert(mat(0,1) === 1.0f)
    assert(mat(1,1) === 3.0f)
    assert(mat(3,4) === 4.0f)
    mat.release()
    mat2.release()
    queue.release()
    context.release()
  }

  test("Matrix Transpose") {
    implicit val context = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU)
    implicit val queue = context.createDefaultOutOfOrderQueueIfPossible()
    val mat = CLMatrix.zeros[Float](10,10)
    val mat2 = CLMatrix.zeros[Float](10, 3)
    mat2 := DenseMatrix.rand(10, 3).values.map(_.toFloat)
    mat(0 until 3, ::).t := mat2
    assert(mat(0 until 3, ::).t.toString === mat2.toString)
    mat.release()
    mat2.release()
    queue.release()
    context.release()
  }





}
