package puck.util

import org.scalatest.FunSuite
import breeze.linalg.DenseVector

/**
 * TODO
 *
 * @author dlwh
 **/
class BitHacksTest extends FunSuite {

  test("order bit vectors: DV(0) < DV(1)") {
    val a = DenseVector(0)
    val b = DenseVector(1)
    assert(BitHacks.bitVectorLT(a, b))
    assert(!BitHacks.bitVectorLT(b, a))
  }

  test("order bit vectors: !(DV(1) < DV(1))") {
    val a = DenseVector(1)
    val b = DenseVector(1)
    assert(!BitHacks.bitVectorLT(a, b))
    assert(!BitHacks.bitVectorLT(b, a))
  }
  test("order bit vectors: (DV(1, 0) < DV(1, 1))") {
    val a = DenseVector(1, 0)
    val b = DenseVector(1, 1)
    assert(BitHacks.bitVectorLT(a, b))
    assert(!BitHacks.bitVectorLT(b, a))
  }

  test("order bit vectors: (DV(1, 0) < DV(0, 1))") {
    val a = DenseVector(1, 0)
    val b = DenseVector(0, 1)
    assert(BitHacks.bitVectorLT(a, b))
    assert(!BitHacks.bitVectorLT(b, a))
  }

  test("order bit vectors: (DV(0, 0) < DV(1, 0))") {
    val a = DenseVector(0, 0)
    val b = DenseVector(1, 1)
    assert(BitHacks.bitVectorLT(a, b))
    assert(!BitHacks.bitVectorLT(b, a))
  }
  test("order bit vectors: (DV(1, 0) < DV(2, 0))") {
    val a = DenseVector(1, 0)
    val b = DenseVector(2, 0)
    assert(BitHacks.bitVectorLT(a, b))
    assert(!BitHacks.bitVectorLT(b, a))
  }

  test("order bit vectors: (DV(1<<31, 0) < DV(-1, 0))") {
    val a = DenseVector(1<<31, 0)
    val b = DenseVector(-1, 0)
    assert(BitHacks.bitVectorLT(a, b), s"$a $b")
    assert(!BitHacks.bitVectorLT(b, a))
  }

  test("order bit vectors: !(DV(1<<31, -1) < DV(-1, 0))") {
    val a = DenseVector(1<<31, -1)
    val b = DenseVector(-1, 0)
    assert(!BitHacks.bitVectorLT(a, b), s"$a $b")
    assert(BitHacks.bitVectorLT(b, a))
  }


  test("order bit vectors: DV(0, 31) < DV(0, 32)") {
    val a = DenseVector(0, 31)
    val b = DenseVector(0, 32)
    assert(BitHacks.bitVectorLT(a, b), s"$a $b")
    assert(!BitHacks.bitVectorLT(b, a))
  }

  test("order bit vectors: DV(0, 1<<30) < DV(0, -1)") {
    val a = DenseVector(0, 1<<30)
    val b = DenseVector(0, -1)
    assert(BitHacks.bitVectorLT(a, b), s"$a $b")
    assert(!BitHacks.bitVectorLT(b, a))
  }

}
