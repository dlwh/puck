package puck.linalg

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
    mat(0 until 5, 0 until 3) := mat2(0 until 5, ::)
    assert(mat(0,1) === 3.0f)
    assert(mat(5,1) === 1.0f)
    assert(mat(0,8) === 0.0f)
    assert(mat(5,9) === 0.0f)
  }

}
