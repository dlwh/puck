package puck.linalg

import breeze.linalg._
import scala.reflect.ClassTag
import scala.Vector
import breeze.util.ArrayUtil
import breeze.generic.CanTransformValues
import breeze.storage.DefaultArrayValue
import breeze.math.Ring
import breeze.linalg.operators._
import breeze.linalg.support._
import puck.util.NativeArray

/**
 * A NativeMatrix is a matrix with all elements found in an NativeArray. It is column major unless isTranspose is true,
 * It is designed to be fast: Double- (and potentially Float-)valued NativeMatrices
 * can be used with blas, and support operations to that effect.
 *
 * @author dlwh
 * @param rows number of rows
 * @param cols number of cols
 * @param data The underlying data.
 *             Column-major unless isTranpose is true.
 *             Mutate at your own risk.
 *             Note that this matrix may be a view of the data.
 *             Use linearIndex(r,c) to calculate indices.
 * @param offset starting point into array
 * @param majorStride distance separating columns (or rows, for isTranspose). should be >= rows (or cols, for isTranspose)
 * @param isTranspose if true, then the matrix is considered to be "transposed" (that is, row major)
 */
@SerialVersionUID(1L)
final class NativeMatrix[@specialized(Int, Float, Double) V](val rows: Int,
                                                             val cols: Int,
                                                             val data: NativeArray[V],
                                                             val offset: Int,
                                                             val majorStride: Int,
                                                             val isTranspose: Boolean = false)
  extends Matrix[V] with MatrixLike[V, NativeMatrix[V]] with Serializable {
  /** Creates a matrix with the specified data array, rows, and columns. */
  def this(rows: Int, cols: Int)(implicit man: ClassTag[V]) = this(rows, cols, NativeArray[V](rows * cols), 0, rows)
  /** Creates a matrix with the specified data array, rows, and columns. Data must be column major */
  def this(rows: Int, cols: Int, data: NativeArray[V], offset: Int = 0) = this(rows, cols, data, offset, rows)

  def apply(row: Int, col: Int) = {
    if(row < 0 || row >= rows) throw new IndexOutOfBoundsException((row,col) + " not in [0,"+rows+") x [0," + cols+")")
    if(col < 0 || col >= cols) throw new IndexOutOfBoundsException((row,col) + " not in [0,"+rows+") x [0," + cols+")")
    data(linearIndex(row, col))
  }

  def pointer = data.pointer


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
    data(linearIndex(row, col)) = v
  }

  def repr = this

  def activeIterator = iterator

  def activeValuesIterator = valuesIterator

  def activeKeysIterator = keysIterator

  override def equals(p1: Any) = p1 match {
    case x: NativeMatrix[_] =>
      // todo: make this faster in obvious cases
      rows == x.rows && cols == x.cols && (valuesIterator sameElements x.valuesIterator )

    case _ => false
  }

  def activeSize = data.length.toInt

  def valueAt(i: Int) = data(i)

  def indexAt(i: Int) = i

  def isActive(i: Int) = true
  def allVisitableIndicesActive = true

  /*
  override def ureduce[A](f: URFunc[V, A]): A = {
    val idealMajorStride = if(isTranspose) cols else rows
    if(majorStride == idealMajorStride && offset == 0) f(data, rows*cols)
    else if(majorStride == idealMajorStride) f(data, offset, 1, rows*cols, {_ => {true}})
    else f(valuesIterator)
  }
  */

  def copy: NativeMatrix[V] = {
    implicit val man = ClassTag[V](data.getClass.getComponentType.asInstanceOf[Class[V]])
    val result = new NativeMatrix[V](rows, cols, NativeArray[V](size))
    result := this
    result
  }
}

object NativeMatrix extends LowPriorityNativeMatrix with MatrixConstructors[NativeMatrix]  {
  /**
   * The standard way to create an empty matrix, size is rows * cols
   */
  def zeros[@specialized(Int, Float, Double) V:ClassTag:DefaultArrayValue](rows: Int, cols: Int) = {
    val data = new Array[V](rows * cols)
    if(rows * cols != 0 && data(0) != implicitly[DefaultArrayValue[V]].value)
      ArrayUtil.fill(data, 0, data.length, implicitly[DefaultArrayValue[V]].value)
    new NativeMatrix(rows, cols, NativeArray(data))
  }

  def create[@specialized(Int, Float, Double) V:DefaultArrayValue](rows: Int, cols: Int, data: Array[V]): NativeMatrix[V] = {
    new NativeMatrix(rows, cols, NativeArray[V](data))
  }


  /** Horizontally tiles some matrices. They must have the same number of rows */
  def horzcat[M,V](matrices: M*)(implicit ev: M <:< Matrix[V], opset: BinaryUpdateOp[NativeMatrix[V], M, OpSet], vman: ClassTag[V], dav: DefaultArrayValue[V]) = {
    if(matrices.isEmpty) zeros[V](0,0)
    else {
      require(matrices.forall(m => m.rows == matrices(0).rows),"Not all matrices have the same number of rows")
      val numCols = matrices.foldLeft(0)(_ + _.cols)
      val numRows = matrices(0).rows
      val res = NativeMatrix.zeros[V](numRows,numCols)
      var offset = 0
      for(m <- matrices) {
        res(0 until numRows,(offset) until (offset + m.cols)) := m
        offset+= m.cols
      }
      res
    }
  }

  /** Vertically tiles some matrices. They must have the same number of columns */
  def vertcat[V](matrices: NativeMatrix[V]*)(implicit opset: BinaryUpdateOp[NativeMatrix[V], NativeMatrix[V], OpSet], vman: ClassTag[V], dav: DefaultArrayValue[V]) = {
    if(matrices.isEmpty) zeros[V](0,0)
    else {
      require(matrices.forall(m => m.cols == matrices(0).cols),"Not all matrices have the same number of columns")
      val numRows = matrices.foldLeft(0)(_ + _.rows)
      val numCols = matrices(0).cols
      val res = NativeMatrix.zeros[V](numRows,numCols)
      var offset = 0
      for(m <- matrices) {
        res((offset) until (offset + m.rows),0 until numCols) := m
        offset+= m.rows
      }
      res
    }
  }


  // slices
  implicit def canSliceRow[V]: CanSlice2[NativeMatrix[V], Int, ::.type, NativeMatrix[V]] = {
    new CanSlice2[NativeMatrix[V], Int, ::.type, NativeMatrix[V]] {
      def apply(m: NativeMatrix[V], row: Int, ignored: ::.type) = {
        if(row < 0 || row >= m.rows) throw new ArrayIndexOutOfBoundsException("Row must be in bounds for slice!")
        if(!m.isTranspose)
          new NativeMatrix(1, m.cols, m.data, m.offset + row, m.majorStride)
        else
          new NativeMatrix(1, m.cols, m.data, m.offset + row * m.cols, 1)
      }
    }
  }

  implicit def canSliceCol[V]: CanSlice2[NativeMatrix[V], ::.type, Int, NativeMatrix[V]] = {
    new CanSlice2[NativeMatrix[V], ::.type, Int, NativeMatrix[V]] {
      def apply(m: NativeMatrix[V], ignored: ::.type, col: Int) = {
        if(col < 0 || col >= m.cols) throw new ArrayIndexOutOfBoundsException("Column must be in bounds for slice!")
        if(!m.isTranspose)
          new NativeMatrix(m.rows, 1, m.data, col * m.majorStride + m.offset)
        else
          new NativeMatrix(1, m.cols, m.data, offset = m.offset + col, majorStride = m.majorStride)
      }
    }
  }

  implicit def canSliceRows[V]: CanSlice2[NativeMatrix[V], Range, ::.type, NativeMatrix[V]] = {
    new CanSlice2[NativeMatrix[V], Range, ::.type, NativeMatrix[V]] {
      def apply(m: NativeMatrix[V], rows: Range, ignored: ::.type) = {
        if(rows.isEmpty) new NativeMatrix(0, 0, m.data, 0, 0)
        else if(!m.isTranspose) {
          require(rows.step == 1, "Sorry, we can't support row ranges with step sizes other than 1")
          val first = rows.head
          new NativeMatrix(rows.length, m.cols, m.data, m.offset + first, m.majorStride)
        } else {
          canSliceCols(m.t, ::, rows).t
        }
      }
    }
  }

  implicit def canSliceCols[V]: CanSlice2[NativeMatrix[V], ::.type, Range, NativeMatrix[V]] = {
    new CanSlice2[NativeMatrix[V], ::.type, Range, NativeMatrix[V]] {
      def apply(m: NativeMatrix[V], ignored: ::.type, cols: Range) = {
        if(cols.isEmpty) new NativeMatrix(0, 0, m.data, 0, 1)
        else if(!m.isTranspose) {
          val first = cols.head
          new NativeMatrix(m.rows, cols.length, m.data, m.offset + first * m.majorStride, m.majorStride * cols.step)
        } else {
          canSliceRows(m.t, cols, ::).t
        }
      }
    }
  }

  implicit def canSliceColsAndRows[V]: CanSlice2[NativeMatrix[V], Range, Range, NativeMatrix[V]] = {
    new CanSlice2[NativeMatrix[V], Range, Range, NativeMatrix[V]] {
      def apply(m: NativeMatrix[V], rows: Range, cols: Range) = {
        if(rows.isEmpty || cols.isEmpty) new NativeMatrix(0, 0, m.data, 0, 1)
        else if(!m.isTranspose) {
          require(rows.step == 1, "Sorry, we can't support row ranges with step sizes other than 1 for non transposed matrices")
          val first = cols.head
          new NativeMatrix(rows.length, cols.length, m.data, m.offset + first * m.rows + rows.head, m.majorStride * cols.step)
        } else {
          require(cols.step == 1, "Sorry, we can't support col ranges with step sizes other than 1 for transposed matrices")
          canSliceColsAndRows(m.t, cols, rows).t
        }
      }
    }
  }



  implicit def negFromScale[V](implicit scale: BinaryOp[NativeMatrix[V], V, OpMulScalar, NativeMatrix[V]], field: Ring[V]) = {
    new UnaryOp[NativeMatrix[V], OpNeg, NativeMatrix[V]] {
      override def apply(a : NativeMatrix[V]) = {
        scale(a, field.negate(field.one))
      }
    }
  }

  implicit def canSlicePartOfRow[V]: CanSlice2[NativeMatrix[V], Int, Range, NativeMatrix[V]] = {
    new CanSlice2[NativeMatrix[V], Int, Range, NativeMatrix[V]] {
      def apply(m: NativeMatrix[V], row: Int, cols: Range) = {
        if(row < 0  || row > m.rows) throw new IndexOutOfBoundsException("Slice with out of bounds row! " + row)
        if(cols.isEmpty) new NativeMatrix(0, 0, m.data, 0, 1)
        else if(!m.isTranspose) {
          val first = cols.head
          new NativeMatrix(1, cols.length, m.data, m.offset + first * m.rows + row, m.majorStride * cols.step)
        } else {
          require(cols.step == 1, "Sorry, we can't support col ranges with step sizes other than 1 for transposed matrices")
          canSlicePartOfCol(m.t, cols, row).t
        }
      }
    }
  }

  implicit def canSlicePartOfCol[V]: CanSlice2[NativeMatrix[V], Range, Int, NativeMatrix[V]] = {
    new CanSlice2[NativeMatrix[V], Range, Int, NativeMatrix[V]] {
      def apply(m: NativeMatrix[V], rows: Range, col: Int) = {
        if(rows.isEmpty) new NativeMatrix(0, 0, m.data, 0)
        else if(!m.isTranspose) {
          new NativeMatrix(col * m.rows + m.offset + rows.head, 1, m.data, rows.step, rows.length)
        } else {
          val m2 = canSlicePartOfRow(m.t, col, rows).t
          m2(::, 0)
        }
      }
    }
  }

  /*
  implicit def canMapValues[V, R:ClassTag] = {
    new CanMapValues[NativeMatrix[V],V,R,NativeMatrix[R]] {
      override def map(from : NativeMatrix[V], fn : (V=>R)) = {
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
        new NativeMatrix[R](from.rows, from.cols, data)
      }

      override def mapActive(from : NativeMatrix[V], fn : (V=>R)) =
        map(from, fn)
    }
  }
  */


  implicit def canTransformValues[V]:CanTransformValues[NativeMatrix[V], V, V] = {
    new CanTransformValues[NativeMatrix[V], V, V] {
      def transform(from: NativeMatrix[V], fn: (V) => V) {
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

      def transformActive(from: NativeMatrix[V], fn: (V) => V) {
        transform(from, fn)
      }
    }
  }

  /*
  implicit def canMapKeyValuePairs[V, R:ClassTag] = {
    new CanMapKeyValuePairs[NativeMatrix[V],(Int,Int),V,R,NativeMatrix[R]] {
      override def map(from : NativeMatrix[V], fn : (((Int,Int),V)=>R)) = {
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
        new NativeMatrix(from.rows, from.cols, data)
      }

      override def mapActive(from : NativeMatrix[V], fn : (((Int,Int),V)=>R)) =
        map(from, fn)
    }
  }
  */

  implicit def canTranspose[V]: CanTranspose[NativeMatrix[V], NativeMatrix[V]] = {
    new CanTranspose[NativeMatrix[V], NativeMatrix[V]] {
      def apply(from: NativeMatrix[V]) = {
        new NativeMatrix(data = from.data, offset = from.offset, cols = from.rows, rows = from.cols, majorStride = from.majorStride, isTranspose = !from.isTranspose)
      }
    }
  }

  /*
  implicit def canTransposeComplex: CanTranspose[NativeMatrix[Complex], NativeMatrix[Complex]] = {
    new CanTranspose[NativeMatrix[Complex], NativeMatrix[Complex]] {
      def apply(from: NativeMatrix[Complex]) = {
        new NativeMatrix(data = from.data map { _.conjugate },
          offset = from.offset,
          cols = from.rows,
          rows = from.cols,
          majorStride = from.majorStride,
          isTranspose = !from.isTranspose)
      }
    }
  }
  */

  implicit def canCopyNativeMatrix[V:ClassTag] = new CanCopy[NativeMatrix[V]] {
    def apply(v1: NativeMatrix[V]) = {
      v1.copy
    }
  }

  def binaryOpFromBinaryUpdateOp[V, Other, Op<:OpType](implicit copy: CanCopy[NativeMatrix[V]], op: BinaryUpdateOp[NativeMatrix[V], Other, Op], man: ClassTag[V]) = {
    new BinaryOp[NativeMatrix[V], Other, Op, NativeMatrix[V]] {
      override def apply(a : NativeMatrix[V], b : Other) = {
        val c = copy(a)
        op(c, b)
        c
      }
    }
  }

  implicit def binaryLeftMulOpFromBinaryRightOp[V, Op<:OpType](implicit op: BinaryOp[NativeMatrix[V], V, OpMulScalar, NativeMatrix[V]]) = {
    new BinaryOp[V, NativeMatrix[V], Op, NativeMatrix[V]] {
      override def apply(a : V, b: NativeMatrix[V]) = {
        op(b, a)
      }
    }
  }

  /**
   * Maps the columns into a new dense matrix
   * @tparam V
   * @tparam R
   * @return
  implicit def canMapRows[V:ClassTag:DefaultArrayValue]: CanCollapseAxis[NativeMatrix[V], Axis._0.type, NativeMatrix[V], NativeMatrix[V], NativeMatrix[V]]  = new CanCollapseAxis[NativeMatrix[V], Axis._0.type, NativeMatrix[V], NativeMatrix[V], NativeMatrix[V]] {
    def apply(from: NativeMatrix[V], axis: Axis._0.type)(f: (NativeMatrix[V]) => NativeMatrix[V]): NativeMatrix[V] = {
      var result:NativeMatrix[V] = null
      for(c <- 0 until from.cols) {
        val col = f(from(::, c))
        if(result eq null) {
          result = NativeMatrix.zeros[V](col.length, from.cols)
        }
        result(::, c) := col
      }
      if(result eq null){
        NativeMatrix.zeros[V](0, from.cols)
      } else {
        result
      }
    }
  }

  /**
   * Returns a numRows NativeMatrix
   * @tparam V
   * @tparam R
   * @return
   */
  implicit def canMapCols[V:ClassTag:DefaultArrayValue] = new CanCollapseAxis[NativeMatrix[V], Axis._1.type, NativeMatrix[V], NativeMatrix[V], NativeMatrix[V]] {
    def apply(from: NativeMatrix[V], axis: Axis._1.type)(f: (NativeMatrix[V]) => NativeMatrix[V]): NativeMatrix[V] = {
      var result:NativeMatrix[V] = null
      val t = from.t
      for(r <- 0 until from.rows) {
        val row = f(t(::, r))
        if(result eq null) {
          result = NativeMatrix.zeros[V](from.rows, row.length)
        }
        result.t apply (::, r) := row
      }
      result
    }
  }


  //  implicit val setMM_D: BinaryUpdateOp[NativeMatrix[Double], NativeMatrix[Double], OpSet] = new SetDMDMOp[Double]
  //  implicit val setMM_F: BinaryUpdateOp[NativeMatrix[Float], NativeMatrix[Float], OpSet]  = new SetDMDMOp[Float]
  //  implicit val setMM_I: BinaryUpdateOp[NativeMatrix[Int], NativeMatrix[Int], OpSet]  = new SetDMDMOp[Int]

  implicit val setMV_D: BinaryUpdateOp[NativeMatrix[Double], NativeMatrix[Double], OpSet] = new SetDMDVOp[Double]
  implicit val setMV_F: BinaryUpdateOp[NativeMatrix[Float], NativeMatrix[Float], OpSet]  = new SetDMDVOp[Float]
  implicit val setMV_I: BinaryUpdateOp[NativeMatrix[Int], NativeMatrix[Int], OpSet]  = new SetDMDVOp[Int]

  // There's a bizarre error specializing float's here.
  class CanZipMapValuesNativeMatrix[@specialized(Int, Double, Float) V, @specialized(Int, Double) RV: ClassTag] extends CanZipMapValues[NativeMatrix[V], V, RV, NativeMatrix[RV]] {
    def create(rows: Int, cols: Int) = new NativeMatrix(rows, cols, new Array[RV](rows * cols))

    /**Maps all corresponding values from the two collection. */
    def map(from: NativeMatrix[V], from2: NativeMatrix[V], fn: (V, V) => RV) = {
      require(from.rows == from2.rows, "Vector row dimensions must match!")
      require(from.cols == from2.cols, "Vector col dimensions must match!")
      val result = create(from.rows, from.cols)
      var i = 0
      while (i < from.rows) {
        var j = 0
        while (j < from.cols) {
          result(i, j) = fn(from(i, j), from2(i, j))
          j += 1
        }
        i += 1
      }
      result
    }
  }

  implicit def zipMap[V, R: ClassTag]: CanZipMapValuesNativeMatrix[V, R] = new CanZipMapValuesNativeMatrix[V, R]
  implicit val zipMap_d: CanZipMapValuesNativeMatrix[Double, Double] = new CanZipMapValuesNativeMatrix[Double, Double]
  implicit val zipMap_f: CanZipMapValuesNativeMatrix[Float, Float] = new CanZipMapValuesNativeMatrix[Float, Float]
  implicit val zipMap_i: CanZipMapValuesNativeMatrix[Int, Int] = new CanZipMapValuesNativeMatrix[Int, Int]

  implicit def canGaxpy[V: Semiring]: CanAxpy[V, NativeMatrix[V], NativeMatrix[V]] = {
    new CanAxpy[V, NativeMatrix[V], NativeMatrix[V]] {
      val ring = implicitly[Semiring[V]]
      def apply(s: V, b: NativeMatrix[V], a: NativeMatrix[V]) {
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
}

trait LowPriorityNativeMatrix1 {
  class SetMMOp[@specialized(Int, Double, Float) V] extends BinaryUpdateOp[NativeMatrix[V], Matrix[V], OpSet] {
    def apply(a: NativeMatrix[V], b: Matrix[V]) {
      require(a.rows == b.rows, "Matrixs must have same number of rows")
      require(a.cols == b.cols, "Matrixs must have same number of columns")

      // slow path when we don't have a trivial matrix
      val ad = a.data
      var c = 0
      while(c < a.cols) {
        var r = 0
        while(r < a.rows) {
          ad(a.linearIndex(r, c)) = b(r, c)
          r += 1
        }
        c += 1
      }
    }
  }



  class SetDMVOp[@specialized(Int, Double, Float) V] extends BinaryUpdateOp[NativeMatrix[V], Vector[V], OpSet] {
    def apply(a: NativeMatrix[V], b: Vector[V]) {
      require(a.rows == b.length && a.cols == 1 || a.cols == b.length && a.rows == 1, "NativeMatrix must have same number of rows, or same number of columns, as NativeMatrix, and the other dim must be 1.")
      val ad = a.data
      var i = 0
      var c = 0
      while(c < a.cols) {
        var r = 0
        while(r < a.rows) {
          ad(a.linearIndex(r, c)) = b(i)
          r += 1
          i += 1
        }
        c += 1
      }
    }
  }

  implicit def setMM[V]: BinaryUpdateOp[NativeMatrix[V], Matrix[V], OpSet] = new SetMMOp[V]
  implicit def setMV[V]: BinaryUpdateOp[NativeMatrix[V], Vector[V], OpSet] = new SetDMVOp[V]
}

trait LowPriorityNativeMatrix extends LowPriorityNativeMatrix1 {

  implicit def canSliceWeirdRows[V]:CanSlice2[NativeMatrix[V], Seq[Int], ::.type, SliceMatrix[Int, Int, V]] = {
    new CanSlice2[NativeMatrix[V], Seq[Int], ::.type, SliceMatrix[Int, Int, V]] {
      def apply(from: NativeMatrix[V], slice: Seq[Int], slice2: ::.type): SliceMatrix[Int, Int, V] = {
        new SliceMatrix(from, slice.toIndexedSeq, (0 until from.cols))
      }
    }
  }

  class SetDMDMOp[@specialized(Int, Double, Float) V] extends BinaryUpdateOp[NativeMatrix[V], NativeMatrix[V], OpSet] {
    def apply(a: NativeMatrix[V], b: NativeMatrix[V]) {
      require(a.rows == b.rows, "Matrices must have same number of rows")
      require(a.cols == b.cols, "Matrices must have same number of columns")
      if(a.data.length - a.offset == a.rows * a.cols
        && b.data.length - b.offset == a.rows * a.cols
        && a.majorStride == b.majorStride
        && a.isTranspose == b.isTranspose) {
        b.data.pointer.next(b.offset).copyTo(a.data.pointer.next(a.offset), a.size)
        System.arraycopy(b.data, b.offset, a.data, a.offset, a.size)
        return
      } else if(!a.isTranspose && !b.isTranspose) {
        // copy one column at a time
        for(column <- 0 until a.cols) {
          b.data.pointer.next(b.offset + b.majorStride * column).copyTo(a.data.pointer.next(a.offset + column * a.majorStride), a.rows)
        }
        return
      }

      // slow path when we don't have a trivial matrix
      val ad = a.data
      val bd = b.data
      var c = 0
      while(c < a.cols) {
        var r = 0
        while(r < a.rows) {
          ad(a.linearIndex(r, c)) = bd(b.linearIndex(r, c))
          r += 1
        }
        c += 1
      }
    }
  }

  /*
  class SetDMDVOp[@specialized(Int, Double, Float) V] extends BinaryUpdateOp[NativeMatrix[V], NativeMatrix[V], OpSet] {
    def apply(a: NativeMatrix[V], b: NativeMatrix[V]) {
      require(a.rows == b.length && a.cols == 1 || a.cols == b.length && a.rows == 1, "NativeMatrix must have same number of rows, or same number of columns, as NativeMatrix, and the other dim must be 1.")
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


  class SetMSOp[@specialized(Int, Double, Float) V:ClassTag] extends BinaryUpdateOp[NativeMatrix[V], V, OpSet] {
    private val buffer = new ThreadLocal[Array[V]] {
      override def initialValue() = new Array[V](1000)
    }
    def apply(a: NativeMatrix[V], b: V) {
      if(b == null.asInstanceOf[V]) {
        val casted = a.data.pointer.as(java.lang.Byte.TYPE)
        casted.clearBytes(casted.getValidElements)
      } else if(a.data.length - a.offset == a.rows * a.cols) {
        val buf = buffer.get()
        // There's no array fill method for native arrays, so we do this instead.
        ArrayUtil.fill(buf, 0, buf.length, b)
        for(start <- 0L until a.data.length by buf.length) {
          a.data.pointer.setArrayAtOffset(start, if(start + buf.length >= a.data.length) buf.take(a.data.length-start toInt) else buf)
        }
      } else {
        // slow path when we don't have a trivial matrix
        val ad = a.data
        var c = 0
        while(c < a.cols) {
          var r = 0
          while(r < a.rows) {
            ad(a.linearIndex(r, c)) = b
            r += 1
          }
          c += 1
        }
      }
    }
  }

  implicit def setDMDM[V]: BinaryUpdateOp[NativeMatrix[V], NativeMatrix[V], OpSet] = new SetDMDMOp[V]
//  implicit def setDMDV[V]: BinaryUpdateOp[NativeMatrix[V], NativeMatrix[V], OpSet] = new SetDMDVOp[V]
  implicit def setDMS[V:ClassTag]: BinaryUpdateOp[NativeMatrix[V], V, OpSet] = new SetMSOp[V]
}


