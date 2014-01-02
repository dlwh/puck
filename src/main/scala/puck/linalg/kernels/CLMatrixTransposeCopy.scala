package puck.linalg
package kernels

import org.bridj._
import com.nativelibs4java.opencl._
import java.nio._
import java.util

// TODO... i bet kernel has a context, so this leaks the context...
class CLMatrixTransposeCopy private(wgSize: Array[Int], kernel: CLKernel, kernelOut: CLKernel) {

  def permuteTransposeCopy(dst: CLMatrix[Float],
                           src: CLMatrix[Float],
                           srcColumnPointers: Array[Int], numCols: Int,
                           events: CLEvent*)(implicit queue: CLQueue):CLEvent = {
    val ptr = Pointer.pointerToArray[java.lang.Integer](srcColumnPointers)
    val intBuffer = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, numCols)
    val ev = intBuffer.write(queue, 0, numCols, ptr, false, events: _*)
    val res = this.permuteTransposeCopy(dst, src, intBuffer.asInstanceOf[CLBuffer[Int]], 0, numCols, ev)
    res.invokeUponCompletion(new Runnable() {
      def run() = {
        ptr.release(); intBuffer.release()
      }
    })
    res
  }

  def permuteTransposeCopy(dst: CLMatrix[Float],
    src: CLMatrix[Float],
    srcColumnPointers: CLBuffer[Int], colOff: Int, numCols: Int,
    events: CLEvent*)(implicit queue: CLQueue):CLEvent = {
    synchronized {
      require(dst.rows == numCols, (dst.rows,dst.cols) + " " + (src.rows, src.cols, numCols))
      require(dst.isTranspose == src.isTranspose)

      kernel.setArgs(
        dst.data.safeBuffer, Integer.valueOf(dst.offset), Integer.valueOf(dst.majorStride),
        src.data.safeBuffer, Integer.valueOf(src.offset), Integer.valueOf(src.majorStride),
        srcColumnPointers,
        Integer.valueOf(src.rows),
        Integer.valueOf(colOff),
        Integer.valueOf(numCols))
      kernel.enqueueNDRange(queue, Array(wgSize(0) * 40, wgSize(0) * 10, 1), wgSize, (events): _*)
    }
  }

  def permuteTransposeCopyOut(dst: CLMatrix[Float],
    dstColPointers: Array[Int], numCols: Int,
    src: CLMatrix[Float],
    events: CLEvent*)(implicit queue: CLQueue):CLEvent = synchronized {
    require(src.rows == numCols, src.rows +" " + numCols)
    assert(dstColPointers.slice(0,numCols).forall(_ < dst.cols), dstColPointers.toIndexedSeq.filter(_ != 0).map(x => x -> (x < dst.cols)) -> dst.cols)
    require(dst.rows == src.cols)
    require(dst.isTranspose == src.isTranspose)
    val ptr = Pointer.pointerToArray[java.lang.Integer](dstColPointers)
    val intBuffer = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, numCols)
    val ev = intBuffer.write(queue, 0, numCols, ptr, false, events:_*)
    val res = permuteTransposeCopyOut(dst, intBuffer.asInstanceOf[CLBuffer[Int]], numCols, src, ev)
    res.invokeUponCompletion(new Runnable() {
      def run() = { ptr.release(); intBuffer.release() }
    })
    res
  }

  def permuteTransposeCopyOut(dst: CLMatrix[Float],
                              dstColPointers: CLBuffer[Int], numCols: Int,
                              src: CLMatrix[Float],
                              events: CLEvent*)(implicit queue: CLQueue):CLEvent = synchronized {
    require(dst.rows == src.cols)
    require(dst.isTranspose == src.isTranspose)
    assert(numCols == src.rows)

    kernelOut.setArgs(
      dst.data.safeBuffer, Integer.valueOf(dst.offset), Integer.valueOf(dst.majorStride), 
      dstColPointers,
      src.data.safeBuffer, Integer.valueOf(src.offset), Integer.valueOf(src.majorStride), 
      Integer.valueOf(numCols),
      Integer.valueOf(src.cols))
    val blockSize = wgSize(0)
    val adjustedSrcCols = ((src.cols + blockSize - 1)/blockSize)*blockSize
    val adjustedSrcRowBlocks = ((numCols + blockSize - 1)/blockSize)
    //val res = kernelOut.enqueueNDRange(queue, Array(adjustedSrcCols, adjustedSrcRowBlocks, 1), Array(blockSize, 1, 1), (ev +: events):_*)
    kernelOut.enqueueNDRange(queue, Array(adjustedSrcCols, adjustedSrcRowBlocks, 1), Array(blockSize, 1, 1), events:_*)
  }




}


object CLMatrixTransposeCopy {
  def apply(preferredBlockSize: Int = 32)(implicit context: CLContext) = {
    map.synchronized {
      import scala.collection.JavaConverters._
      // TODO ??!?!??!
      // not sure what's going on, but Apple's Intel reports 1024/1/1, but can't handle more than 1/1/1...
      val blockSize = 32

      val wgSize = if (context.getDevices.head.toString.contains("Apple") && context.getDevices.head.toString.contains("Intel Core")) {
        Array(1, 1, 1)
      } else {
        val wgSizes = context.getDevices.head.getMaxWorkItemSizes
        val x = wgSizes(0) min blockSize
        val maxProduct = context.getDevices.head.getMaxWorkGroupSize
        Array(x toInt, (maxProduct / x toInt) min 4, 1)
      }
      val prog = context.createProgram(permuteTransposeCopy(blockSize, wgSize))
      val kernel = prog.createKernel("transpose_copy")
      val kernel2 = prog.createKernel("transpose_copy_out")

      map.asScala.getOrElseUpdate(context, new CLMatrixTransposeCopy(wgSize, kernel, kernel2))
    }
  }

  private val map = new util.WeakHashMap[CLContext, CLMatrixTransposeCopy]

/** Transposes src into dst, permuting the columns as it goes. Matrices are column major.*/
   def permuteTransposeCopy(blockSize: Int, wgSize: Array[Int]) = {
  """
#define T float
#define BLOCK_SIZE """ + blockSize + """

__attribute__((reqd_work_group_size(""" + wgSize.mkString(", ") + """)))
__kernel void transpose_copy(__global T* _dst, int dstOff, int dstMajorStride, 
                             __global T* _src, int srcOff, int srcMajorStride, __global int* srcPtrs,
                             int srcRows, int colOff, int srcCols) {
  int numGroupsX = BLOCK_SIZE * get_num_groups(0);
  int numGroupsY = BLOCK_SIZE * get_num_groups(1);
  int firstBlockX = BLOCK_SIZE * get_group_id(0);
  int firstBlockY = BLOCK_SIZE * get_group_id(1);
  __local float tile[BLOCK_SIZE][BLOCK_SIZE+1];


  int threadid = get_local_id(0);
  int threadidy = get_local_id(1);

  __global T* dst = _dst + dstOff;
  __global T* src = _src + srcOff;

  for (int yb = firstBlockY; yb < srcCols; yb += numGroupsY) {
    for (int xb = firstBlockX; xb < srcRows; xb += numGroupsX) {
      int ylim = min(srcCols, yb + BLOCK_SIZE);
      int xlim = min(srcRows, xb + BLOCK_SIZE);
      #pragma unroll
      for (int y = threadidy + yb; y < ylim; y += get_local_size(1)) {
       #pragma unroll
        for(int x = threadid + xb; x < xlim; x += get_local_size(0)) {
          tile[x-xb][y-yb] = src[srcPtrs[colOff + y]*srcMajorStride + x];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      #pragma unroll
      for (int x = threadidy + xb; x < xlim; x += get_local_size(1)) {
       #pragma unroll
        for(int y = yb + threadid; y < ylim; y += get_local_size(0)) {
          dst[y + x*dstMajorStride] = tile[x-xb][y-yb];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

}

__kernel void transpose_copy_out(
      __global T* _dst, int dstOff, int dstMajorStride, __global int* dstPtrs,
      __global T* _src, int srcOff, int srcMajorStride, 
      int srcRows, int srcCols) {
  // copy each col into block[i]
  __local T block[BLOCK_SIZE][BLOCK_SIZE+1]; // + 1 to avoid bank conflicts
  event_t copyInEvents[BLOCK_SIZE];

  __global T* dst = _dst + dstOff;
  __global T* src = _src + srcOff;

  int srcCol = get_global_id(0);
  int threadid = get_local_id(0);
  // srcCol - threadid is the same for all threads in a workgroup.
  int firstSrcCol = get_group_id(0) * BLOCK_SIZE;
  int nColsToDo = max(min(BLOCK_SIZE, srcCols - firstSrcCol),0);

  int firstSrcRow = get_global_id(1) * BLOCK_SIZE;
  int nRowsToDo =  min(BLOCK_SIZE, srcRows - firstSrcRow);

  __local int myPtrs[BLOCK_SIZE];
  event_t copyFirstPtr = async_work_group_copy(myPtrs, dstPtrs + firstSrcRow, nRowsToDo, 0);


  for(int i = 0; i < nColsToDo; ++i) {
    copyInEvents[i] = async_work_group_copy(block[i], // block(i, ::)
      src + srcMajorStride * (firstSrcCol + i) + firstSrcRow, // src(firstSrcRow --> nRowsToDo, myPtrs(i))
      nRowsToDo, 0); //
    // TODO: why is this necessary on intel? the wait_group_events below doesn't work.
    wait_group_events(1, copyInEvents + i);
  }


  wait_group_events(nColsToDo, copyInEvents);
  wait_group_events(1, &copyFirstPtr);

  // each block[i] now contains the slice src(firstSrcRow --> nRowsToDo, firstSrcCol + i)
  // we want to move src(firstSrcRow, ::) to dst(::, dstPtrs(firstSrcRow))
  // so we want thread i to write block[i][j] to dst(dstRow, firstSrcRow + j)

  for(int j = 0; j < nRowsToDo && threadid < nColsToDo; j += 1) {
    dst[myPtrs[j] * dstMajorStride + srcCol] = block[threadid][j];
  }


}


                                                                  """
  }




}
