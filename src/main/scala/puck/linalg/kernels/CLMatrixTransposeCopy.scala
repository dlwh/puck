package puck.linalg
package kernels

import org.bridj._
import com.nativelibs4java.opencl._
import java.nio._
import java.util

// TODO... i bet kernel has a context, so this leaks the context...
class CLMatrixTranposeCopy private(blockSize: Int, kernel: CLKernel) {


  def permuteTransposeCopy(dst: CLMatrix[Float],
    src: CLMatrix[Float],
    srcColumnPointers: Array[Int], events: CLEvent*)(implicit queue: CLQueue) = synchronized {
    require(dst.cols == srcColumnPointers.length)
    require(dst.isTranspose == !src.isTranspose)

    val ptr = Pointer.pointerToArray[java.lang.Integer](srcColumnPointers)
    val intBuffer = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, srcColumnPointers.length)
    val ev = intBuffer.write(queue, 0, srcColumnPointers.length, ptr, false, events:_*)
    kernel.setArgs(
      dst.data, Integer.valueOf(dst.offset), Integer.valueOf(dst.majorStride), 
      src.data, Integer.valueOf(src.offset), Integer.valueOf(src.majorStride), 
      intBuffer,
      Integer.valueOf(src.rows),
      Integer.valueOf(srcColumnPointers.length))
    val adjustedSrcCols = ((srcColumnPointers.length + blockSize - 1)/blockSize)*blockSize
    val adjustedSrcRowBlocks = ((src.rows + blockSize - 1)/blockSize)
    val res = kernel.enqueueNDRange(queue, Array(adjustedSrcCols, adjustedSrcRowBlocks, 1), Array(blockSize, 1, 1), (ev +: events):_*)
    res.invokeUponCompletion(new Runnable() {
      def run() = { ptr.release(); intBuffer.release() }
    })
    res
  }

}


object CLMatrixTranposeCopy {
  def apply(preferredBlockSize: Int = 32)(implicit context: CLContext) = map.synchronized {
    import scala.collection.JavaConverters._
    // TODO ??!?!??!
    // not sure what's going on, but Apple's Intel reports 1024/1/1, but can't handle more than 1/1/1...
    val blockSize = if(context.getDevices.head.toString.contains("Apple") && context.getDevices.head.toString.contains("Intel")) {
      1
    } else {
      val x = context.getDevices.head.getMaxWorkItemSizes
      val size0 = x(0)
      math.min(size0, preferredBlockSize).toInt
    }
    val prog = context.createProgram(permuteTransposeCopy(blockSize))
    val kernel = prog.createKernel("transpose_copy")
    
    map.asScala.getOrElseUpdate(context, new CLMatrixTranposeCopy(blockSize, kernel))
  }

  private val map = new util.WeakHashMap[CLContext, CLMatrixTranposeCopy]

/** Transposes src into dst, permuting the columns as it goes. Matrices are column major.*/
   def permuteTransposeCopy(blockSize: Int) = {
"""
#define T float
#define BLOCK_SIZE """ + blockSize + """
__kernel void transpose_copy(__global T* _dst, int dstOff, int dstMajorStride, 
                             __global T* _src, int srcOff, int srcMajorStride, __global int* srcPtrs,
                             int srcRows, int srcCols) {
  // copy each col into block[i]
  __local T block[BLOCK_SIZE][BLOCK_SIZE+1]; // + 1 to avoid bank conflicts
  event_t copyInEvents[BLOCK_SIZE];

  __global T* dst = _dst + dstOff;
  __global T* src = _src + srcOff;
                        
  int dstRow = get_global_id(0);
  int threadid = get_local_id(0);
  // dstRow - threadid is the same for all threads in a workgroup.
  __local int myPtrs[BLOCK_SIZE];
  int ndo = max(min(BLOCK_SIZE, srcCols - (dstRow - threadid)),0);
  event_t copyFirstPtr = async_work_group_copy(myPtrs, srcPtrs + dstRow - threadid, ndo, 0);
  wait_group_events(1, &copyFirstPtr);

  int firstSrcRow = get_global_id(1) * BLOCK_SIZE;
  int nRowsToDo =  min(BLOCK_SIZE, srcRows - firstSrcRow);

  // if the srcPtrs were next to each other, we could use async_work_group_strided_copy
  // but they're not :(
  for(int i = 0; i < ndo; ++i) 
    copyInEvents[i] = async_work_group_copy(block[i], // block(i, ::)
      src + srcMajorStride * myPtrs[i] + firstSrcRow, // src(firstSrcRow --> nRowsToDo, myPtrs(i))
      nRowsToDo, 0); //

  wait_group_events(ndo, copyInEvents);
  
  // each block[i] now contains the slice src(firstSrcRow --> nRowsToDo, myPtrs[i]) 
  // so we want thread i to write block[i][j] to dst(dstRow, firstSrcRow + j)

  for(int j = 0; j < nRowsToDo && threadid < ndo; j += 1) {
    int dstCol = firstSrcRow + j;
    dst[dstCol * dstMajorStride + dstRow] = block[threadid][j];
  }


}


"""
  }




}
