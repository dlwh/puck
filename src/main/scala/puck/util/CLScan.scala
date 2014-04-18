package puck.util

import com.nativelibs4java.opencl._
import com.nativelibs4java.opencl.CLMem.Usage
import java.nio.IntBuffer

/**
 * TODO
 *
 * @author dlwh
 **/
class CLScan(simpleScan: CLKernel) {

  def scan(arr: Array[Int], ev: CLEvent*)(implicit queue: CLQueue):Array[Int] = simpleScan.synchronized {
    val src = queue.getContext.createIntBuffer(Usage.InputOutput, IntBuffer.wrap(arr), true)
    val dest = queue.getContext.createIntBuffer(Usage.InputOutput, arr.length)
    val evr = scan(dest, src, arr.length, ev:_*)
    evr.waitFor()
    dest.read(queue, evr).getInts
  }

  def scan(dest: CLBuffer[Integer], src: CLBuffer[Integer], numElems: Int, ev: CLEvent*)(implicit queue: CLQueue):CLEvent = simpleScan.synchronized {
    require(dest.getElementCount >= numElems)
    require(src.getElementCount >= numElems)
    simpleScan.setArgs(dest, src, Integer.valueOf(numElems))
    simpleScan.enqueueNDRange(queue,  Array(32), Array(32), ev:_*)
  }

}

object CLScan {
  def make(implicit context: CLContext) = {
    val program: CLProgram = context.createProgram(computeWorkArrayOffsetsText).build()
    new CLScan(program.createKernel("simpleScan"))
  }

  val computeWorkArrayOffsetsText =
    """
      |
      |#define WORKGROUP_SIZE 32
      |inline int warpScanInclusive(int idata, volatile __local int *l_Data, int size){
      |  int pos = 2 * get_local_id(0) - (get_local_id(0) & (size - 1));
      |  barrier(CLK_LOCAL_MEM_FENCE);
      |  l_Data[pos] = 0;
      |  barrier(CLK_LOCAL_MEM_FENCE);
      |  pos += size;
      |  l_Data[pos] = idata;
      |  barrier(CLK_LOCAL_MEM_FENCE);
      |
      |   for(uint offset = 1; offset < size; offset <<= 1){
      |        barrier(CLK_LOCAL_MEM_FENCE); //Fails with Intel openCL
      |        uint t = l_Data[pos] + l_Data[pos - offset];
      |        barrier(CLK_LOCAL_MEM_FENCE); //Fails with Intel openCL
      |        l_Data[pos] = t;
      |    }
      |
      |
      |/*
      |  if(size >=  2) l_Data[pos] += l_Data[pos -  1];
      |  barrier(CLK_LOCAL_MEM_FENCE);
      |  if(size >=  4) l_Data[pos] += l_Data[pos -  2];
      |  barrier(CLK_LOCAL_MEM_FENCE);
      |  if(size >=  8) l_Data[pos] += l_Data[pos -  4];
      |  barrier(CLK_LOCAL_MEM_FENCE);
      |  if(size >= 16) l_Data[pos] += l_Data[pos -  8];
      |  barrier(CLK_LOCAL_MEM_FENCE);
      |  if(size >= 32) l_Data[pos] += l_Data[pos - 16];
      |  barrier(CLK_LOCAL_MEM_FENCE);
      |  */
      |
      |  return l_Data[pos];
      | }
      |
      |
      |__kernel void simpleScan(__global int* dest, __global const int* src, int n) {
      |  __local int buffer[WORKGROUP_SIZE * 2];
      |  int local_id = get_local_id(0);
      |  __local int increment;
      |  increment = 0;
      |  for(int base = 0; base < n; base += WORKGROUP_SIZE) {
      |     int myid = base + local_id;
      |     int size = min(WORKGROUP_SIZE, n - base);
      |     int myresult = warpScanInclusive((local_id < size) ? src[myid] : 0, buffer, WORKGROUP_SIZE) + increment;
      |
      |     if(myid < n)
      |       dest[myid] = myresult;
      |
      |     if(local_id == size - 1)
      |       increment = myresult;
      |    barrier(CLK_LOCAL_MEM_FENCE);
      |  }
      |
      |
      |}
    """.stripMargin

}