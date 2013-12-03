package puck.linalg.kernels

import org.bridj._
import com.nativelibs4java.opencl._
import puck.linalg.CLMatrix

// TODO... i bet kernel has a context, so this leaks the context...
class CLMatrixSliceCopy private(blockSize: Int, kernel: CLKernel, kernelOut: CLKernel) {

  def sliceCopy(dst: CLMatrix[Float],
                src: CLMatrix[Float],
                srcColumnPointers: Array[Int], numCols: Int,
                events: CLEvent*)(implicit queue: CLQueue):CLEvent = {
    val ptr = Pointer.pointerToArray[java.lang.Integer](srcColumnPointers)
    val intBuffer = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, numCols)
    val ev = intBuffer.write(queue, 0, numCols, ptr, false, events: _*)
    val res = this.sliceCopy(dst, src, intBuffer.asInstanceOf[CLBuffer[Int]], numCols, ev)
    res.invokeUponCompletion(new Runnable() {
      def run() = {
        ptr.release(); intBuffer.release()
      }
    })
    res
  }

  def sliceCopy(dst: CLMatrix[Float],
    src: CLMatrix[Float],
    srcColumnPointers: CLBuffer[Int], numCols: Int,
    events: CLEvent*)(implicit queue: CLQueue):CLEvent = {
    synchronized {
      require(dst.cols == numCols, dst.cols + " " + numCols)
      require(dst.isTranspose == src.isTranspose)
      require(dst.rows == src.rows)

      kernel.setArgs(
        dst.data.safeBuffer, Integer.valueOf(dst.offset), Integer.valueOf(dst.majorStride),
        src.data.safeBuffer, Integer.valueOf(src.offset), Integer.valueOf(src.majorStride),
        srcColumnPointers,
        Integer.valueOf(src.rows),
        Integer.valueOf(src.cols))
      val adjustedSrcRowBlocks = puck.roundUpToMultipleOf(src.rows, blockSize)
      kernel.enqueueNDRange(queue, Array(numCols, 1, 1), Array(1, 1, 1), (events): _*)
    }
  }

  def sliceCopyOut(dst: CLMatrix[Float],
    dstColPointers: Array[Int], numCols: Int,
    src: CLMatrix[Float],
    events: CLEvent*)(implicit queue: CLQueue):CLEvent = synchronized {
    require(src.cols == numCols, src.cols +" " + numCols)
    require(dst.cols == src.cols)
    require(dst.isTranspose == src.isTranspose)
    val ptr = Pointer.pointerToArray[java.lang.Integer](dstColPointers)
    val intBuffer = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, numCols)
    val ev = intBuffer.write(queue, 0, numCols, ptr, false, events:_*)
    val res = sliceCopyOut(dst, intBuffer.asInstanceOf[CLBuffer[Int]], numCols, src, ev)
    res.invokeUponCompletion(new Runnable() {
      def run() = { ptr.release(); intBuffer.release() }
    })
    res
  }

  def sliceCopyOut(dst: CLMatrix[Float],
                   dstColPointers: CLBuffer[Int], numCols: Int,
                   src: CLMatrix[Float],
                   events: CLEvent*)(implicit queue: CLQueue):CLEvent = synchronized {
    require(dst.rows == src.rows)
    require(dst.isTranspose == src.isTranspose)
    assert(numCols == src.cols)

    kernelOut.setArgs(
      dst.data.safeBuffer, Integer.valueOf(dst.offset), Integer.valueOf(dst.majorStride), 
      dstColPointers,
      src.data.safeBuffer, Integer.valueOf(src.offset), Integer.valueOf(src.majorStride), 
      Integer.valueOf(src.rows),
      Integer.valueOf(src.cols))
    val adjustedSrcCols = puck.roundUpToMultipleOf(src.cols, blockSize)
    //val res = kernelOut.enqueueNDRange(queue, Array(adjustedSrcCols, adjustedSrcRowBlocks, 1), Array(blockSize, 1, 1), (ev +: events):_*)
    kernelOut.enqueueNDRange(queue, Array(adjustedSrcCols, puck.roundUpToMultipleOf(src.rows, blockSize), 1), Array(1, blockSize, 1), events:_*)
  }




}

object CLMatrixSliceCopy {
  def apply(preferredBlockSize: Int = 32)(implicit context: CLContext) = map.synchronized {
    import scala.collection.JavaConverters._
    // TODO ??!?!??!
    // not sure what's going on, but Apple's Intel reports 1024/1/1, but can't handle more than 1/1/1...
    val blockSize = if( context.getDevices.head.toString.contains("Apple") && context.getDevices.head.toString.contains("Intel")) {
      1
    } else {
      val x = context.getDevices.head.getMaxWorkItemSizes
      val size0 = x(0)
      math.min(size0, preferredBlockSize).toInt
    }
    val prog = context.createProgram(sliceCopy(blockSize))
    val kernel = prog.createKernel("slice_copy")
    val kernel2 = prog.createKernel("slice_copy_out")

    map.asScala.getOrElseUpdate(context, new CLMatrixSliceCopy(blockSize, kernel, kernel2))
  }

  private val map = new java.util.WeakHashMap[CLContext, CLMatrixSliceCopy]

/** Copy src into dst, permuting the columns as it goes. Matrices are column major.*/
   def sliceCopy(blockSize: Int) = {
  """
#define T float
#define BLOCK_SIZE """ + blockSize + """
__kernel void slice_copy(__global T* _dst, int dstOff, int dstMajorStride,
                         __global const T* _src, int srcOff, int srcMajorStride, __global int* srcPtrs,
                             int srcRows, int srcCols) {
  // copy each col into block[i]

  __global T* dst = _dst + dstOff;
  __global const T* src = _src + srcOff;

  int dstCol = get_global_id(0);
  if(dstCol >= srcCols) return;
  int threadid = 0;//get_local_id(1);
  int local_size = 1;//get_local_size(1);

  int srcCol = srcPtrs[dstCol];

  int firstRow = 0;////get_group_id(1) * BLOCK_SIZE;
  int lastRow = srcRows;//min(BLOCK_SIZE, srcRows - firstRow);


  for(int i = firstRow + threadid; i < lastRow; i += local_size) {
    dst[dstCol * dstMajorStride + i] = src[srcCol * srcMajorStride + i];
  }

}

__kernel void slice_copy_out(
      __global T* _dst, int dstOff, int dstMajorStride, __global int* dstPtrs,
      __global T* _src, int srcOff, int srcMajorStride,
      int srcRows, int srcCols) {

    __global T* dst = _dst + dstOff;
    __global const T* src = _src + srcOff;

    int srcCol = get_global_id(0);
    if(srcCol >= srcCols) return;
    int threadid = get_local_id(1);
    int local_size = get_local_size(1);

    int dstCol = dstPtrs[srcCol];

    int firstRow = 0;//get_group_id(1) * BLOCK_SIZE;
    int lastRow = srcRows;//min(BLOCK_SIZE, srcRows - firstRow);


    for(int i = firstRow + threadid; i < lastRow; i += local_size) {
      dst[dstCol * dstMajorStride + i] = src[srcCol * srcMajorStride + i];
    }


}


                                     """
  }




}