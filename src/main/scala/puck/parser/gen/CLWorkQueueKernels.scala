package puck.parser.gen

import com.nativelibs4java.opencl._
import puck.util.{ZipUtil, CLScan}
import puck.parser.{PruningMask, Batch, WorkSpace}
import org.bridj.Pointer
import puck.parser.Batch
import puck.PointerFreer
import java.util.zip.{ZipFile, ZipOutputStream}

/**
 *
 *
 * @author dlwh
 */
class CLWorkQueueKernels(enqueueKernel: CLKernel)(implicit ctxt: CLContext) {
  private val scanKernel = CLScan.make

  def enqueueDense[W](ws: WorkSpace,
                      batch: Batch[W],
                      spanLength: Int,
                      pTop: Boolean, lTop: Boolean, rTop: Boolean,
                      events: CLEvent*)(implicit queue: CLQueue):(CLEvent, Int) = synchronized {
    val scratch = ctxt.createIntBuffer(CLMem.Usage.InputOutput, batch.numSentences + 1)
    def I(x: Boolean) = if(x) Integer.valueOf(1) else Integer.valueOf(0)
    enqueueKernel.setArgs(ws.pPtrBuffer, ws.lPtrBuffer, ws.rPtrBuffer, scratch,
      batch.cellOffsetsDev, batch.lengthsDev, null, null, Integer.valueOf(0), I(pTop), I(lTop), I(rTop), Integer.valueOf(spanLength), I(true))

    val computeNeededEv = enqueueKernel.enqueueNDRange(queue, Array(batch.numSentences), Array(1), events:_*)

    val evScan = scanKernel.scan(ws.queueOffsets, scratch, batch.numSentences, computeNeededEv)
    PointerFreer.enqueue({scratch.release()}, evScan)

    enqueueKernel.setArg(3, ws.queueOffsets)
    enqueueKernel.setArg(13, I(false))

    val res = enqueueKernel.enqueueNDRange(queue, Array(batch.numSentences), Array(1), evScan)
    val numNeeded = scratch.read(queue, batch.numSentences, 1, evScan).get()

    (res, numNeeded.intValue())
  }

  def write(prefix: String, out: ZipOutputStream) {
    ZipUtil.addKernel(out, s"$prefix/enqueue", enqueueKernel)
  }
}

object CLWorkQueueKernels {
  def forLoopType(numCoarseSyms: Int, loopType: LoopType)(implicit ctxt: CLContext):CLWorkQueueKernels = {
    loopType match {
      case LoopType.Inside => forInside(numCoarseSyms)
      case LoopType.OutsideL => forOutsideL(numCoarseSyms)
      case LoopType.OutsideR => forOutsideR(numCoarseSyms)
      case LoopType.OutsideRTerm => forOutsideRTerm(numCoarseSyms)
      case LoopType.OutsideLTerm => forOutsideLTerm(numCoarseSyms)
    }
  }

  def forSplitRange(numCoarseSyms: Int, splitRangeFunc: String)(implicit ctxt: CLContext):CLWorkQueueKernels = {
    val text = this.text(numCoarseSyms) + "\n\n\n" + splitRangeFunc

    val prg = ctxt.createProgram(text).build()
    val k = prg.createKernel("enqueue")
    new CLWorkQueueKernels(k)
  }

  def read(prefix: String, in: ZipFile)(implicit context: CLContext) = {
    val k = ZipUtil.readKernel(in, s"$prefix/enqueue")
    new CLWorkQueueKernels(k)
  }

  def forInside(numCoarseSyms: Int)(implicit ctxt: CLContext) = {
    forSplitRange(numCoarseSyms, insideSplitRange)
  }

  def forOutsideL(numCoarseSyms: Int)(implicit ctxt: CLContext) = {
    forSplitRange(numCoarseSyms, outsideLSplitRange)
  }

  def forOutsideLTerm(numCoarseSyms: Int)(implicit ctxt: CLContext) = {
    forSplitRange(numCoarseSyms, outsideLTermSplitRange)
  }


  def forOutsideR(numCoarseSyms: Int)(implicit ctxt: CLContext) = {
    forSplitRange(numCoarseSyms, outsideRSplitRange)
  }

  def forOutsideRTerm(numCoarseSyms: Int)(implicit ctxt: CLContext) = {
    forSplitRange(numCoarseSyms, outsideRTermSplitRange)
  }


  def insideSplitRange =
    """
      | range_t computeSplitRange(int begin, int end, int length) {
      |   range_t r= {begin + 1, end - 1};
      |   return r;
      | }
    """.stripMargin

  def outsideRSplitRange =
    """
      | range_t computeSplitRange(int begin, int end, int length) {
      |   range_t r = {0, begin};
      |   return r;
      | }
    """.stripMargin

  def outsideLSplitRange =
    """
      | range_t computeSplitRange(int begin, int end, int length) {
      |   range_t r =  {end + 1, length};
      |   return r;
      | }
    """.stripMargin

  def outsideLTermSplitRange =
    """
      | range_t computeSplitRange(int begin, int end, int length) {
      |   range_t r = {end + 1, end + 1};
      |   return r;
      | }
    """.stripMargin

  def outsideRTermSplitRange =
    """
      | range_t computeSplitRange(int begin, int end, int length) {
      |   range_t r = {begin - 1, begin - 1};
      |   return r;
      | }
    """.stripMargin

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
      |   //const int nextQueueOffset = queueOffsets[sent];
      |   const int length = lengths[sent];
      |   const int cellOffset = cellOffsets[sent];
      |   const int lastCell = cellOffsets[sent + 1];
      |
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
      |       pMask = masks[parentCell];
      |
      |     if(!useMasks || maskIntersects(&pMask, &tMask)) {
      |       range_t splitRange = computeSplitRange(begin, end, length);
      |
      |       for(int split = splitRange.low; split <= splitRange.high; ++split) {
      |          int leftCell = leftOffset + (
      |                             (split < begin) ? computeCell(split, begin, length)
      |                                             : computeCell(begin, split, length)
      |                           );
      |          int rightCell = rightOffset + (
      |                             (split < end) ? computeCell(split, end, length)
      |                                             : computeCell(end, split, length)
      |                           );
      |
      |         int includeThisOne = useMasks;
      |         if(!includeThisOne) {
      |           lMask = masks[leftCell];
      |           int doLeft = maskAny(&lMask);
      |           if (doLeft) {
      |             rMask = masks[rightCell];
      |             int doRight = maskAny(&rMask);
      |             if(doRight) includeThisOne = true;
      |           }
      |         }
      |
      |         if(includeThisOne) {
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