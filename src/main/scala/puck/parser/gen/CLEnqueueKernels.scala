package puck.parser.gen

import com.nativelibs4java.opencl._
import puck.util.CLScan
import puck.parser.{PruningMask, Batch, WorkSpace}
import org.bridj.Pointer
import puck.parser.Batch

/**
 *
 *
 * @author dlwh
 */
class CLEnqueueKernels(enqueueKernel: CLKernel)(implicit ctxt: CLContext) {
  private val scanKernel = CLScan.make

  /*
  def enqueueDense[W](ws: WorkSpace,
                      batch: Batch,
                      spanLength: Int,
                      pTop: Boolean, lTop: Boolean, rTop: Boolean,
                      events: CLEvent*)(implicit queue: CLQueue):CLEvent = synchronized {
    def I(x: Boolean) = if(x) 1 else 0
    enqueueKernel.setArgs(ws.parentQueue, ws.leftQueue, ws.rightQueue, ws.scratch,
      batch.cellOffsetsDev, batch.lengthsDev, null, null, Integer.valueOf(0), I(pTop), I(lTop), I(rTop), spanLength, Integer.valueOf(1))
    val computeNeeded = enqueueKernel.enqueueNDRange(queue, Array(batch.numSentences), Array(1), events:_*)
    val evScan = scanKernel.scan(ws.queueOffsets, ws.scratch, batch.numSentences, computeNeeded)
    enqueueKernel.setArg(3, ws.queueOffsets)
    enqueueKernel.setArg(13, 1)

    enqueueKernel.enqueueNDRange(queue, Array(batch.numSentences), Array(1), evScan)

  }
  */



}

object CLEnqueueKernels {
  def text(numCoarseSyms: Int) = CLMaskKernels.maskHeader(numCoarseSyms) +
    """
      | typedef struct { int low, high;} range_t;
      |
      | range_t computeSplitRange(int begin, int end, int length);
      |
      | inline int computeCell(int begin, int end, int length) {
      |    int span = end - begin - 1;
      |    return begin + span * length - span * (span - 1) / 2;
      | }
      |
      | __kernel void computeQueueOffsets(__global const int*
      |
      | __kernel void enqueue(__global int* parentQueue,
      |                       __global int* leftQueue,
      |                       __global int* rightQueue,
      |                       __global int* queueOffsets,
      |                       __global const int* cellOffsets,
      |                       __global const int* lengths,
      |                       __global const mask_t* masks,
      |                       __global const mask_t* targetMask,
      |                       int useMasks,
      |                       int parentTop,
      |                       int leftTop,
      |                       int rightTop,
      |                       const int spanLength,
      |                       int justComputeNumNeeded) {
      |   int sent = get_group_id(0);
      |   const int firstQueueOffset = (sent == 0) ? 0 : queueOffsets[sent - 1];
      |   const int nextQueueOffset = queueOffsets[sent];
      |   const int length = lengths[sent];
      |   const int cellOffset = cellOffsets[sent];
      |   const int lastCell = cellOffsets[sent + 1];
      |
      |   int topChart = (lastCell - cellOffset)/2 + cellOffset;
      |   int topChart = (lastCell - cellOffset)/2 + cellOffset;
      |
      |   int parentOffset = (parentTop) ? topChart : cellOffset;
      |   int rightOffset = (rightTop) ? topChart : cellOffset;
      |   int leftOffset = (leftTop) ? topChart : cellOffset;
      |
      |   int queueOffset = firstQueueOffset;
      |
      |   mask_t pMask;
      |   mask_t tMask;
      |   if(useMasks) tMask = *targetMask;
      |
      |   mask_t lMask;
      |   mask_t rMask;
      |
      |   int numNeeded = 0;
      |
      |   for(int begin = 0, end = spanLength; end <= length; begin++, end++) {
      |     int parentCell = parentOffset + computeCell(begin, end, length);
      |
      |     if(useMasks)
      |       pMask = *masks[parentCell];
      |
      |     if(!useMasks || maskIntersects(&pMask, &tmask)) {
      |       range_t splitRange = computeSplitRange(begin, end, length);
      |
      |       for(int split = splitRange.low; split < splitRange.high; ++split) {
      |          int leftCell = leftOffset + (
      |                             (split < begin) ? computeCell(split, begin, length)
      |                                             : computeCell(begin, split, length)
      |                           );
      |          int rightCell = rightOffset + (
      |                             (split < end) ? computeCell(split, end, length)
      |                                             : computeCell(end, split, length)
      |                           );
      |
      |
      |         if(!useMasks || (maskAny(&(lMask = masks[leftCell])) && maskAny(&(rMask = masks[rightCell]))) {
      |            if(justComputeNumNeeded) {
      |              numNeeded++;
      |            } else {
      |              parentQueue[queueOffset] = parentCell;
      |              leftQueue[queueOffset] = leftCell;
      |              rightQueue[queueOffset] = rightCell;
      |              queueOffset++;
      |            }
      |         }
      |       }
      |     }
      |   }
      |
      |   if(justComputeNumNeeded) {
      |     queueOffsets[sent] = numNeeded;
      |   }
      | }
    """.stripMargin
}