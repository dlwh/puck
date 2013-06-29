package puck.linalg

import breeze.linalg._
import java.{lang=>jl}
import java.nio._
import scala.reflect.ClassTag
import scala.Vector
import breeze.util.ArrayUtil
import breeze.generic.CanTransformValues
import breeze.storage.DefaultArrayValue
import breeze.math.Ring
import breeze.linalg.operators._
import breeze.linalg.support._
import puck.util._
import com.nativelibs4java.opencl._
import org.bridj.Pointer
import com.nativelibs4java.opencl.util.Primitive
import scala.concurrent._
import ExecutionContext.Implicits.global
import kernels._

/**
 * A CLMatrix is a matrix with all elements found in an NativeArray. It is column major unless isTranspose is true,
 * It is designed to be fast: Double- (and potentially Float-)valued NativeMatrices
 * can be used with blas, and support operations to that effect.
 *
 * @author dlwh
 * @param rows number of rows
 * @param cols number of cols
 * @param data The underlying data.
 *             Column-major unless isTranspose is true.
 *             Mutate at your own risk.
 *             Note that this matrix may be a view of the data.
 *             Use linearIndex(r,c) to calculate indices.
 * @param offset starting point into array
 * @param majorStride distance separating columns (or rows, for isTranspose). should be >= rows (or cols, for isTranspose)
 * @param isTranspose if true, then the matrix is considered to be "transposed" (that is, row major)
 */
final class CLMatrix[@specialized(Int, Float, Double) V](val rows: Int,
                                                         val cols: Int,
                                                         val data: CLBufferMappedPointerPair[V],
                                                         val offset: Int,
                                                         val majorStride: Int,
                                                         val isTranspose: Boolean = false)(implicit val queue: CLQueue)
  extends Matrix[V] with MatrixLike[V, CLMatrix[V]] {
  /** Creates a matrix with the specified data array, rows, and columns. */
  def this(rows: Int, cols: Int)(implicit man: ClassTag[V], context: CLContext, queue: CLQueue) = this(rows, cols, context.createBuffer[V](CLMem.Usage.InputOutput, man.runtimeClass.asInstanceOf[Class[V]], rows * cols), 0, rows)
  /** Creates a matrix with the specified data array, rows, and columns. Data must be column major */
  def this(rows: Int, cols: Int, data: CLBuffer[V], offset: Int = 0)(implicit queue: CLQueue) = this(rows, cols, data, offset, rows)



  def apply(row: Int, col: Int) = {
    if(row < 0 || row >= rows) throw new IndexOutOfBoundsException((row,col) + " not in [0,"+rows+") x [0," + cols+")")
    if(col < 0 || col >= cols) throw new IndexOutOfBoundsException((row,col) + " not in [0,"+rows+") x [0," + cols+")")
    data.mappedPointer.get(linearIndex(row, col))
  }

  /** Calculates the index into the data array for row and column */
  final def linearIndex(row: Int, col: Int): Int = {
    if(isTranspose)
      offset + col + row * majorStride
    else
      offset + row + col * majorStride
  }

  def update(row: Int, col: Int, v: V) {
    if(row < 0 || row > rows) throw new IndexOutOfBoundsException((row,col) + " not in [0,"+rows+") x [0," + cols+")")
    if(col < 0 || col > cols) throw new IndexOutOfBoundsException((row,col) + " not in [0,"+rows+") x [0," + cols+")")
    data.mappedPointer.set(linearIndex(row, col), v)
  }

  def repr = this

  def activeIterator = iterator

  def activeValuesIterator = valuesIterator

  def activeKeysIterator = keysIterator

  override def equals(p1: Any) = p1 match {
    case x: CLMatrix[_] =>
      // todo: make this faster in obvious cases
      rows == x.rows && cols == x.cols && (valuesIterator sameElements x.valuesIterator )

    case _ => false
  }

  def majorSize = if(isTranspose) rows else cols

  def activeSize = size

  def valueAt(i: Int) = data.mappedPointer.get(offset + i)

  def indexAt(i: Int) = i

  def isActive(i: Int) = true
  def allVisitableIndicesActive = true

  def writeFrom(b: DenseMatrix[V], blocking: Boolean, events: CLEvent*):Seq[CLEvent] = {
    require(b.rows == this.rows, "Matrices must have same number of rows")
    require(b.cols == this.cols, "Matrices must have same number of columns")
    val evv = data.unmap(events:_*)
    val floats =  Pointer.pointerToArray[V](b.data)
    val ev = if(isGapless(b) && this.isGapless && b.isTranspose == this.isTranspose) {
       IndexedSeq(data.buffer.write(queue, offset, size, floats.next(b.offset), blocking, evv))
    } else if(b.isTranspose == this.isTranspose) {
      // copy one "column" at b time
      val rr = if(b.isTranspose) b.cols else b.rows
      val cc = if(b.isTranspose) b.rows else b.cols
      val ev = for(column <- 0 until cc) yield {
       data.buffer.write(queue, offset + majorStride * column, 
         rr, floats.next(b.offset + b.majorStride * column), blocking, evv)
      }
      ev
    } else {
      ???
    }
    if(blocking) {
      ev.filter(_ ne null).foreach(_.waitFor())
      floats.release()
    } 

    ev
  }

  private def isGapless = (!this.isTranspose && this.majorStride == this.rows) || (this.isTranspose && this.majorStride == this.cols)
  private def isGapless(dm: DenseMatrix[V]) = (!dm.isTranspose && dm.majorStride == dm.rows) || (dm.isTranspose && dm.majorStride == dm.cols)


  def writeFrom(b: CLMatrix[V], blocking: Boolean, events: CLEvent*):CLEvent = {
    require(b.queue eq this.queue)
    require(b.rows == this.rows, "Matrices must have same number of rows")
    require(b.cols == this.cols, "Matrices must have same number of columns")
    val evv = data.unmap(events:_*)
    val evv2 = b.data.unmap(events:_*)
    (Option(evv).iterator ++ Option(evv2).iterator).foreach(_.waitFor())
    val ev = if(b.isGapless && this.isGapless && b.isTranspose == this.isTranspose) {
       b.data.buffer.copyTo(queue, b.offset, rows * cols, this.data, this.offset, evv, evv2)
    } else if(b.isTranspose == this.isTranspose) {
      // copy one "column" at b time
      val rr = if(b.isTranspose) b.cols else b.rows
      val cc = if(b.isTranspose) b.rows else b.cols
      val ev = for(column <- 0 until cc) yield {
        b.data.buffer.copyTo(queue, b.offset + b.majorStride * column, rr,
                    this.data.buffer, this.offset + this.majorStride * column, evv, evv2)
      }
      this.queue.enqueueMarker()
    } else {
      // TODO: currently assumes elements are 4 bytes long!!!!
      val tc = CLMatrixTransposeCopy()(queue.getContext)
      //tc.permuteTransposeCopy(this.t.asInstanceOf[CLMatrix[Float]], b.asInstanceOf[CLMatrix[Float]], Array.range(0, b.cols), evv, evv2)
      tc.permuteTransposeCopyOut(this.t.asInstanceOf[CLMatrix[Float]], Array.range(0, rows), b.asInstanceOf[CLMatrix[Float]], events:_*)
    }
    if(blocking)
      ev.waitFor()



    ev

  }

  /** Forcibly releases the buffer. Note that other slices will be invalidated! */
  def release() = {
    data.release()
  }

}

object CLMatrix extends LowPriorityNativeMatrix {
  /**
   * The standard way to create an empty matrix, size is rows * cols
   */
  def zeros[@specialized(Int, Float, Double) V](rows: Int, cols: Int)(implicit ct: ClassTag[V],
                                                                      dav: DefaultArrayValue[V],
                                                                      context: CLContext, queue: CLQueue): CLMatrix[V] = {
    val data = new Array[V](rows * cols)
    if(rows * cols != 0 && data(0) != implicitly[DefaultArrayValue[V]].value)
      ArrayUtil.fill(data, 0, data.length, implicitly[DefaultArrayValue[V]].value)
    create(rows, cols, data)
  }

  def create[@specialized(Int, Float, Double) V:DefaultArrayValue](rows: Int, cols: Int, data: Array[V])(implicit  ct: ClassTag[V],
                                                                                                         dav: DefaultArrayValue[V],
                                                                                                         context: CLContext, queue: CLQueue): CLMatrix[V] = {
    val ptr = Pointer.pointerToArray[V](data)
    new CLMatrix[V](rows, cols, context.createBuffer[V](CLMem.Usage.InputOutput, ptr))
  }



  // slices
  implicit def canSliceRow[V]: CanSlice2[CLMatrix[V], Int, ::.type, CLMatrix[V]] = {
    new CanSlice2[CLMatrix[V], Int, ::.type, CLMatrix[V]] {
      def apply(m: CLMatrix[V], row: Int, ignored: ::.type) = {
        import m.queue
        if(row < 0 || row >= m.rows) throw new ArrayIndexOutOfBoundsException("Row must be in bounds for slice!")
        if(!m.isTranspose)
          new CLMatrix(1, m.cols, m.data, m.offset + row, m.majorStride)
        else
          new CLMatrix(1, m.cols, m.data, m.offset + row * m.cols, 1)
      }
    }
  }

  implicit def canSliceCol[V]: CanSlice2[CLMatrix[V], ::.type, Int, CLMatrix[V]] = {
    new CanSlice2[CLMatrix[V], ::.type, Int, CLMatrix[V]] {
      def apply(m: CLMatrix[V], ignored: ::.type, col: Int) = {
        import m.queue
        if(col < 0 || col >= m.cols) throw new ArrayIndexOutOfBoundsException("Column must be in bounds for slice!")
        if(!m.isTranspose)
          new CLMatrix(m.rows, 1, m.data, col * m.majorStride + m.offset)
        else
          new CLMatrix(1, m.cols, m.data, offset = m.offset + col, majorStride = m.majorStride)
      }
    }
  }

  implicit def canSliceRows[V]: CanSlice2[CLMatrix[V], Range, ::.type, CLMatrix[V]] = {
    new CanSlice2[CLMatrix[V], Range, ::.type, CLMatrix[V]] {
      def apply(m: CLMatrix[V], rows: Range, ignored: ::.type) = {
        import m.queue
        if(rows.isEmpty) new CLMatrix(0, 0, m.data, 0, 0)
        else if(!m.isTranspose) {
          require(rows.step == 1, "Sorry, we can't support row ranges with step sizes other than 1")
          val first = rows.head
          new CLMatrix(rows.length, m.cols, m.data, m.offset + first, m.majorStride)
        } else {
          canSliceCols(m.t, ::, rows).t
        }
      }
    }
  }

  implicit def canSliceCols[V]: CanSlice2[CLMatrix[V], ::.type, Range, CLMatrix[V]] = {
    new CanSlice2[CLMatrix[V], ::.type, Range, CLMatrix[V]] {
      def apply(m: CLMatrix[V], ignored: ::.type, cols: Range) = {
        import m.queue
        if(cols.isEmpty) new CLMatrix(m.rows, 0, m.data, 0, 1)
        else if(!m.isTranspose) {
          val first = cols.head
          new CLMatrix(m.rows, cols.length, m.data, m.offset + first * m.majorStride, m.majorStride * cols.step)
        } else {
          canSliceRows(m.t, cols, ::).t
        }
      }
    }
  }

  implicit def canSliceColsAndRows[V]: CanSlice2[CLMatrix[V], Range, Range, CLMatrix[V]] = {
    new CanSlice2[CLMatrix[V], Range, Range, CLMatrix[V]] {
      def apply(m: CLMatrix[V], rows: Range, cols: Range) = {
        import m.queue
        if(rows.isEmpty || cols.isEmpty) new CLMatrix(0, 0, m.data, 0, 1)
        else if(!m.isTranspose) {
          require(rows.step == 1, "Sorry, we can't support row ranges with step sizes other than 1 for non transposed matrices")
          val first = cols.head
          new CLMatrix(rows.length, cols.length, m.data, m.offset + first * m.rows + rows.head, m.majorStride * cols.step)(m.queue)
        } else {
          require(cols.step == 1, "Sorry, we can't support col ranges with step sizes other than 1 for transposed matrices")
          canSliceColsAndRows(m.t, cols, rows).t
        }
      }
    }
  }



  implicit def negFromScale[V](implicit scale: BinaryOp[CLMatrix[V], V, OpMulScalar, CLMatrix[V]], field: Ring[V]) = {
    new UnaryOp[CLMatrix[V], OpNeg, CLMatrix[V]] {
      override def apply(a : CLMatrix[V]) = {
        scale(a, field.negate(field.one))
      }
    }
  }

  implicit def canSlicePartOfRow[V]: CanSlice2[CLMatrix[V], Int, Range, CLMatrix[V]] = {
    new CanSlice2[CLMatrix[V], Int, Range, CLMatrix[V]] {
      def apply(m: CLMatrix[V], row: Int, cols: Range) = {
        import m.queue
        if(row < 0  || row > m.rows) throw new IndexOutOfBoundsException("Slice with out of bounds row! " + row)
        if(cols.isEmpty) new CLMatrix(0, 0, m.data, 0, 1)
        else if(!m.isTranspose) {
          val first = cols.head
          new CLMatrix(1, cols.length, m.data, m.offset + first * m.rows + row, m.majorStride * cols.step)
        } else {
          require(cols.step == 1, "Sorry, we can't support col ranges with step sizes other than 1 for transposed matrices")
          canSlicePartOfCol(m.t, cols, row).t
        }
      }
    }
  }

  implicit def canSlicePartOfCol[V]: CanSlice2[CLMatrix[V], Range, Int, CLMatrix[V]] = {
    new CanSlice2[CLMatrix[V], Range, Int, CLMatrix[V]] {
      def apply(m: CLMatrix[V], rows: Range, col: Int) = {
        import m.queue
        if(rows.isEmpty) new CLMatrix(0, 0, m.data, 0)
        else if(!m.isTranspose) {
          new CLMatrix(col * m.rows + m.offset + rows.head, 1, m.data, rows.step, rows.length)
        } else {
          val m2 = canSlicePartOfRow(m.t, col, rows).t
          m2(::, 0)
        }
      }
    }
  }

  /*
  implicit def canMapValues[V, R:ClassTag] = {
    new CanMapValues[CLMatrix[V],V,R,CLMatrix[R]] {
      override def map(from : CLMatrix[V], fn : (V=>R)) = {
        val data = new Array[R](from.size)
        var j = 0
        var off = 0
        while (j < from.cols) {
          var i = 0
          while(i < from.rows) {
            data(off) = fn(from(i, j))
            off += 1
            i += 1
          }
          j += 1
        }
        new CLMatrix[R](from.rows, from.cols, data)
      }

      override def mapActive(from : CLMatrix[V], fn : (V=>R)) =
        map(from, fn)
    }
  }


  implicit def canTransformValues[V]:CanTransformValues[CLMatrix[V], V, V] = {
    new CanTransformValues[CLMatrix[V], V, V] {
      def transform(from: CLMatrix[V], fn: (V) => V) {
        var j = 0
        while (j < from.cols) {
          var i = 0
          while(i < from.rows) {
            from(i, j) = fn(from(i, j))
            i += 1
          }
          j += 1
        }
      }

      def transformActive(from: CLMatrix[V], fn: (V) => V) {
        transform(from, fn)
      }
    }
  }

  implicit def canMapKeyValuePairs[V, R:ClassTag] = {
    new CanMapKeyValuePairs[CLMatrix[V],(Int,Int),V,R,CLMatrix[R]] {
      override def map(from : CLMatrix[V], fn : (((Int,Int),V)=>R)) = {
        val data = new Array[R](from.data.length)
        var j = 0
        var off = 0
        while (j < from.cols) {
          var i = 0
          while(i < from.rows) {
            data(off) = fn(i -> j, from(i, j))
            off += 1
            i += 1
          }
          j += 1
        }
        new CLMatrix(from.rows, from.cols, data)
      }

      override def mapActive(from : CLMatrix[V], fn : (((Int,Int),V)=>R)) =
        map(from, fn)
    }
  }
  */

  implicit def canTranspose[V]: CanTranspose[CLMatrix[V], CLMatrix[V]] = {
    new CanTranspose[CLMatrix[V], CLMatrix[V]] {
      def apply(from: CLMatrix[V]) = {
        new CLMatrix(data = from.data, offset = from.offset, cols = from.rows, rows = from.cols, majorStride = from.majorStride, isTranspose = !from.isTranspose)(from.queue)
      }
    }
  }

  /*
  implicit def canTransposeComplex: CanTranspose[CLMatrix[Complex], CLMatrix[Complex]] = {
    new CanTranspose[CLMatrix[Complex], CLMatrix[Complex]] {
      def apply(from: CLMatrix[Complex]) = {
        new CLMatrix(data = from.data map { _.conjugate },
          offset = from.offset,
          cols = from.rows,
          rows = from.cols,
          majorStride = from.majorStride,
          isTranspose = !from.isTranspose)
      }
    }
  }
  */

  def binaryOpFromBinaryUpdateOp[V, Other, Op<:OpType](implicit copy: CanCopy[CLMatrix[V]], op: BinaryUpdateOp[CLMatrix[V], Other, Op], man: ClassTag[V]) = {
    new BinaryOp[CLMatrix[V], Other, Op, CLMatrix[V]] {
      override def apply(a : CLMatrix[V], b : Other) = {
        val c = copy(a)
        op(c, b)
        c
      }
    }
  }

  /**
   * Maps the columns into a new dense matrix
   * @tparam V
   * @tparam R
   * @return
  implicit def canMapRows[V:ClassTag:DefaultArrayValue]: CanCollapseAxis[CLMatrix[V], Axis._0.type, CLMatrix[V], CLMatrix[V], CLMatrix[V]]  = new CanCollapseAxis[CLMatrix[V], Axis._0.type, CLMatrix[V], CLMatrix[V], CLMatrix[V]] {
    def apply(from: CLMatrix[V], axis: Axis._0.type)(f: (CLMatrix[V]) => CLMatrix[V]): CLMatrix[V] = {
      var result:CLMatrix[V] = null
      for(c <- 0 until from.cols) {
        val col = f(from(::, c))
        if(result eq null) {
          result = CLMatrix.zeros[V](col.length, from.cols)
        }
        result(::, c) := col
      }
      if(result eq null){
        CLMatrix.zeros[V](0, from.cols)
      } else {
        result
      }
    }
  }

  /**
   * Returns a numRows CLMatrix
   * @tparam V
   * @tparam R
   * @return
   */
  implicit def canMapCols[V:ClassTag:DefaultArrayValue] = new CanCollapseAxis[CLMatrix[V], Axis._1.type, CLMatrix[V], CLMatrix[V], CLMatrix[V]] {
    def apply(from: CLMatrix[V], axis: Axis._1.type)(f: (CLMatrix[V]) => CLMatrix[V]): CLMatrix[V] = {
      var result:CLMatrix[V] = null
      val t = from.t
      for(r <- 0 until from.rows) {
        val row = f(t(::, r))
        if(result eq null) {
          result = CLMatrix.zeros[V](from.rows, row.length)
        }
        result.t apply (::, r) := row
      }
      result
    }
  }


  //  implicit val setMM_D: BinaryUpdateOp[CLMatrix[Double], CLMatrix[Double], OpSet] = new SetCLMCLMOp[Double]
  //  implicit val setMM_F: BinaryUpdateOp[CLMatrix[Float], CLMatrix[Float], OpSet]  = new SetCLMCLMOp[Float]
  //  implicit val setMM_I: BinaryUpdateOp[CLMatrix[Int], CLMatrix[Int], OpSet]  = new SetCLMCLMOp[Int]

/*
  implicit def canGaxpy[V: Semiring]: CanAxpy[V, CLMatrix[V], CLMatrix[V]] = {
    new CanAxpy[V, CLMatrix[V], CLMatrix[V]] {
      val ring = implicitly[Semiring[V]]
      def apply(s: V, b: CLMatrix[V], a: CLMatrix[V]) {
        require(a.rows == b.rows, "Vector row dimensions must match!")
        require(a.cols == b.cols, "Vector col dimensions must match!")

        var i = 0
        while (i < a.rows) {
          var j = 0
          while (j < a.cols) {
            a(i, j) = ring.+(a(i, j), ring.*(s, b(i, j)))
            j += 1
          }
          i += 1
        }
      }
    }
  }
  */
   */
}

trait LowPriorityNativeMatrix1 {
//  class SetMMOp[@specialized(Int, Double, Float) V] extends BinaryUpdateOp[CLMatrix[V], Matrix[V], OpSet] {
//    def apply(a: CLMatrix[V], b: Matrix[V]) {
//      require(a.rows == b.rows, "Matrixs must have same number of rows")
//      require(a.cols == b.cols, "Matrixs must have same number of columns")
//
//      // slow path when we don't have a trivial matrix
//      val ad = a.data
//      var c = 0
//      while(c < a.cols) {
//        var r = 0
//        while(r < a.rows) {
//          ad(a.linearIndex(r, c)) = b(r, c)
//          r += 1
//        }
//        c += 1
//      }
//    }
//  }



//  class SetDMVOp[@specialized(Int, Double, Float) V] extends BinaryUpdateOp[CLMatrix[V], Vector[V], OpSet] {
//    def apply(a: CLMatrix[V], b: Vector[V]) {
//      require(a.rows == b.length && a.cols == 1 || a.cols == b.length && a.rows == 1, "CLMatrix must have same number of rows, or same number of columns, as CLMatrix, and the other dim must be 1.")
//      val ad = a.data
//      var i = 0
//      var c = 0
//      while(c < a.cols) {
//        var r = 0
//        while(r < a.rows) {
//          ad(a.linearIndex(r, c)) = b(i)
//          r += 1
//          i += 1
//        }
//        c += 1
//      }
//    }
//  }
//
//  implicit def setMM[V]: BinaryUpdateOp[CLMatrix[V], Matrix[V], OpSet] = new SetMMOp[V]
//  implicit def setMV[V]: BinaryUpdateOp[CLMatrix[V], Vector[V], OpSet] = new SetDMVOp[V]
}

trait LowPriorityNativeMatrix extends LowPriorityNativeMatrix1 {

  implicit def canSliceWeirdRows[V]:CanSlice2[CLMatrix[V], Seq[Int], ::.type, SliceMatrix[Int, Int, V]] = {
    new CanSlice2[CLMatrix[V], Seq[Int], ::.type, SliceMatrix[Int, Int, V]] {
      def apply(from: CLMatrix[V], slice: Seq[Int], slice2: ::.type): SliceMatrix[Int, Int, V] = {
        new SliceMatrix(from, slice.toIndexedSeq, (0 until from.cols))
      }
    }
  }

  class SetCLMCLMVOp[V] extends BinaryUpdateOp[CLMatrix[V], CLMatrix[V], OpSet] {
    def apply(a: CLMatrix[V], b: CLMatrix[V]) {
      a.writeFrom(b, true)
    }
  }

  implicit object SetCLMDMFloatOp extends BinaryUpdateOp[CLMatrix[Float], DenseMatrix[Float], OpSet] {
    def apply(a: CLMatrix[Float], b: DenseMatrix[Float]) {
      a.writeFrom(b, true)
    }
  }



  implicit object setCLMCLMFloat extends SetCLMCLMVOp[Float]
  implicit object setCLMCLMLong extends SetCLMCLMVOp[Long]
  implicit object setCLMCLMInt extends SetCLMCLMVOp[Int]
  implicit object setCLMCLMDouble extends SetCLMCLMVOp[Double]

  /*
  class SetDMDVOp[@specialized(Int, Double, Float) V] extends BinaryUpdateOp[CLMatrix[V], CLMatrix[V], OpSet] {
    def apply(a: CLMatrix[V], b: CLMatrix[V]) {
      require(a.rows == b.length && a.cols == 1 || a.cols == b.length && a.rows == 1, "CLMatrix must have same number of rows, or same number of columns, as CLMatrix, and the other dim must be 1.")
      val ad = a.data
      val bd = b.data
      var c = 0
      var boff = b.offset
      while(c < a.cols) {
        var r = 0
        while(r < a.rows) {
          ad(a.linearIndex(r, c)) = bd(boff)
          r += 1
          boff += b.stride
        }
        c += 1
      }
    }
  }
  */


  implicit object SetMSFloatOp extends BinaryUpdateOp[CLMatrix[Float], Float, OpSet] {
    def apply(a: CLMatrix[Float], b: Float) {
      val zmk = ZeroMemoryKernel()(a.queue.getContext)
      import a.queue
      // nicely shaped matrix
      if( (!a.isTranspose && a.majorStride == a.rows)  ||(a.isTranspose && a.majorStride == a.cols)) {
        val ev = zmk.fillMemory(a.data, b, a.offset, a.rows * a.cols)
        ev.waitFor()
      } else {
        val rr = if(a.isTranspose) a.cols else a.rows
        val cc = if(a.isTranspose) a.rows else a.cols
        val events = for(i <- 0 until cc) yield {
          zmk.fillMemory(a.data, b, a.offset + i * a.majorStride, rr)
        }

        events.foreach(_.waitFor())
      }
    }
  }

}


