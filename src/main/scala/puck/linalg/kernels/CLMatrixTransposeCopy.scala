package puck.linalg
package kernels

import org.bridj._
import com.nativelibs4java.opencl._
import java.nio._
import java.util

// TODO... i bet kernel has a context, so this leaks the context...
class CLMatrixTransposeCopy private(blockSize: Int, kernel: CLKernel, kernelOut: CLKernel) {


  def permuteTransposeCopy(dst: CLMatrix[Float],
    src: CLMatrix[Float],
    srcColumnPointers: Array[Int], events: CLEvent*)(implicit queue: CLQueue) = synchronized {
    require(dst.rows > srcColumnPointers.length, dst.rows + " " + srcColumnPointers.length)
    require(dst.isTranspose == src.isTranspose)

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

  def permuteTransposeCopyOut(dst: CLMatrix[Float],
    dstColPointers: Array[Int], 
    src: CLMatrix[Float],
    events: CLEvent*)(implicit queue: CLQueue) = synchronized {
    require(src.rows >= dstColPointers.length, src.rows +" " + dstColPointers.length)
    require(dst.rows == src.cols)
    require(dst.isTranspose == src.isTranspose)

    val ptr = Pointer.pointerToArray[java.lang.Integer](dstColPointers)
    val intBuffer = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, dstColPointers.length)
    val ev = intBuffer.write(queue, 0, dstColPointers.length, ptr, false, events:_*)
    kernelOut.setArgs(
      dst.data, Integer.valueOf(dst.offset), Integer.valueOf(dst.majorStride), 
      intBuffer,
      src.data, Integer.valueOf(src.offset), Integer.valueOf(src.majorStride), 
      Integer.valueOf(src.rows),
      Integer.valueOf(src.cols))
    val adjustedSrcCols = ((src.cols + blockSize - 1)/blockSize)*blockSize
    val adjustedSrcRowBlocks = ((dstColPointers.length + blockSize - 1)/blockSize)
    val res = kernelOut.enqueueNDRange(queue, Array(adjustedSrcCols, adjustedSrcRowBlocks, 1), Array(blockSize, 1, 1), (ev +: events):_*)
    res.invokeUponCompletion(new Runnable() {
      def run() = { ptr.release(); intBuffer.release() }
    })
    res
  }


}


object CLMatrixTransposeCopy {
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
    val kernel2 = prog.createKernel("transpose_copy_out")
    
    map.asScala.getOrElseUpdate(context, new CLMatrixTransposeCopy(blockSize, kernel, kernel2))
  }

  private val map = new util.WeakHashMap[CLContext, CLMatrixTransposeCopy]

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
  int nColsToDo = max(min(BLOCK_SIZE, srcCols - (dstRow - threadid)),0);
  event_t copyFirstPtr = async_work_group_copy(myPtrs, srcPtrs + dstRow - threadid, nColsToDo, 0);
  wait_group_events(1, &copyFirstPtr);

  int firstSrcRow = get_global_id(1) * BLOCK_SIZE;
  int nRowsToDo =  min(BLOCK_SIZE, srcRows - firstSrcRow);

  // if the srcPtrs were next to each other, we could use async_work_group_strided_copy
  // but they're not :(
  for(int i = 0; i < nColsToDo; ++i) 
    copyInEvents[i] = async_work_group_copy(block[i], // block(i, ::)
      src + srcMajorStride * myPtrs[i] + firstSrcRow, // src(firstSrcRow --> nRowsToDo, myPtrs(i))
      nRowsToDo, 0); //

  wait_group_events(nColsToDo, copyInEvents);
  
  // each block[i] now contains the slice src(firstSrcRow --> nRowsToDo, myPtrs[i]) 
  // so we want thread i to write block[i][j] to dst(dstRow, firstSrcRow + j)

  for(int j = 0; j < nRowsToDo && threadid < nColsToDo; j += 1) {
    int dstCol = firstSrcRow + j;
    dst[dstCol * dstMajorStride + dstRow] = block[threadid][j];
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
  int firstSrcCol = srcCol - threadid;
  int nColsToDo = max(min(BLOCK_SIZE, srcCols - firstSrcCol),0);

  int firstSrcRow = get_global_id(1) * BLOCK_SIZE;
  int nRowsToDo =  min(BLOCK_SIZE, srcRows - firstSrcRow);

  __local int myPtrs[BLOCK_SIZE];
  event_t copyFirstPtr = async_work_group_copy(myPtrs, dstPtrs + firstSrcRow, nRowsToDo, 0);
  wait_group_events(1, &copyFirstPtr);

  // if the srcPtrs were next to each other, we could use async_work_group_strided_copy
  // but they're not :(
  for(int i = 0; i < nColsToDo; ++i) {
    copyInEvents[i] = async_work_group_copy(block[i], // block(i, ::)
      src + srcMajorStride * (firstSrcCol + i) + firstSrcRow, // src(firstSrcRow --> nRowsToDo, myPtrs(i))
      nRowsToDo, 0); //
  }


  wait_group_events(nColsToDo, copyInEvents);
  
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
