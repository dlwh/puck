package puck.parser.gen

import com.nativelibs4java.opencl._
import java.nio.IntBuffer
import com.nativelibs4java.opencl.CLMem.Usage

/**
 *
 *
 * @author dlwh
 */
class CLParserQueue(simpleScan: CLKernel, insideDenseOffsets: CLKernel, outsideDenseOffsets: CLKernel) {

  def computeTerminalOffsets(lengths: CLBuffer[Integer], numSentences: Integer,
                             scratch: CLBuffer[Integer], workArrayOffsets: CLBuffer[Integer], ev: CLEvent*)(implicit queue: CLQueue):CLEvent = simpleScan.synchronized {
    require(lengths.getElementCount >= numSentences)
    require(scratch.getElementCount  >= numSentences)
    require(workArrayOffsets.getElementCount  >= numSentences)
    insideDenseOffsets.setArgs(scratch, lengths, numSentences, 2)
    val evoff = insideDenseOffsets.enqueueNDRange(queue, Array(1024), Array(32), ev:_*)
    val evr = scan(workArrayOffsets,  scratch, numSentences, evoff)
    evr
  }

  def computeDenseInsideOffsets(lengths: CLBuffer[Integer],numSentences: Integer,
                                spanLength: Integer, scratch: CLBuffer[Integer], workArrayOffsets: CLBuffer[Integer], ev: CLEvent*)(implicit queue: CLQueue):CLEvent = simpleScan.synchronized {
    require(lengths.getElementCount >= numSentences)
    require(scratch.getElementCount  >= numSentences)
    require(workArrayOffsets.getElementCount  >= numSentences)
    insideDenseOffsets.setArgs(scratch, lengths, numSentences, spanLength)
    val evoff = insideDenseOffsets.enqueueNDRange(queue, Array(1024), Array(32), ev:_*)
    val evr = scan(workArrayOffsets,  scratch, numSentences, evoff)
    evr
  }

    def computeDenseOutsideOffsets(lengths: CLBuffer[Integer], numSentences: Integer,
                                spanLength: Integer, scratch: CLBuffer[Integer], workArrayOffsets: CLBuffer[Integer], ev: CLEvent*)(implicit queue: CLQueue):CLEvent = simpleScan.synchronized {
    require(lengths.getElementCount >= numSentences)
    require(scratch.getElementCount  >= numSentences)
    require(workArrayOffsets.getElementCount  >= numSentences)
    insideDenseOffsets.setArgs(scratch, lengths, numSentences, spanLength)
    val evoff = insideDenseOffsets.enqueueNDRange(queue, Array(1024), Array(32), ev:_*)
    val evr = scan(workArrayOffsets,  scratch, numSentences, evoff)
    evr
  }


  def scan(arr: Array[Int], ev: CLEvent*)(implicit queue: CLQueue):Array[Int] = simpleScan.synchronized {
    val src = queue.getContext.createIntBuffer(Usage.InputOutput, IntBuffer.wrap(arr), true)
    val dest = queue.getContext.createIntBuffer(Usage.InputOutput, arr.length)
    val evr = scan(dest, src, arr.length, ev:_*)
    evr.waitFor()
    dest.read(queue, evr).getInts
  }

  def scan(dest: CLBuffer[Integer], src: CLBuffer[Integer], numElems: Int, ev: CLEvent*)(implicit queue: CLQueue):CLEvent = simpleScan.synchronized {
    simpleScan.setArgs(dest, src, Integer.valueOf(numElems))
    simpleScan.enqueueNDRange(queue,  Array(32), Array(32), ev:_*)
  }

}


object CLParserQueue {

  def make(implicit context: CLContext) = {
    val program: CLProgram = context.createProgram(computeWorkArrayOffsetsText).build()
    new CLParserQueue(program.createKernel("simpleScan"), program.createKernel("offsetsNeededInsideDense"),  program.createKernel("offsetsNeededOutsideDense"))

  }
  val computeWorkArrayOffsetsText =
    """
      |
      |#define WORKGROUP_SIZE 32
      |   //Almost the same as naive scan1Inclusive but doesn't need barriers
      |    //and works only for size <= WORKGROUP_SIZE
      |inline int warpScanInclusive(int idata, volatile __local int *l_Data, int size){
      |  int pos = 2 * get_local_id(0) - (get_local_id(0) & (size - 1));
      |  l_Data[pos] = 0;
      |  pos += size;
      |  l_Data[pos] = idata;
      |
      |  if(size >=  2) l_Data[pos] += l_Data[pos -  1];
      |  if(size >=  4) l_Data[pos] += l_Data[pos -  2];
      |  if(size >=  8) l_Data[pos] += l_Data[pos -  4];
      |  if(size >= 16) l_Data[pos] += l_Data[pos -  8];
      |  if(size >= 32) l_Data[pos] += l_Data[pos - 16];
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
      |
      |
      |__kernel void offsetsNeededInsideDense(__global int* dest, __global int* lengths, int n, int spanLength) {
      |  int id = get_global_id(0);
      |  for(int i = id; i < n; i += get_global_size(0)) {
      |    int len = lengths[i] ;
      |    // numSpans * (numSplitPoints)
      |    dest[i] = (len - spanLength + 1) * (spanLength - 1);
      |  }
      |}
      |
      |__kernel void offsetsNeededOutsideDense(__global int* dest, __global int* lengths, int n, int spanLength) {
      |  int id = get_global_id(0);
      |  for(int i = id; i < n; i += get_global_size(0)) {
      |    int len = lengths[i] ;
      |    // \sum_{endPoint=spanLength}^len (len - endPoint) == the below
      |    dest[i] = (len - spanLength + 1) * (len - spanLength) / 2;
      |  }
      |}
    """.stripMargin
}
