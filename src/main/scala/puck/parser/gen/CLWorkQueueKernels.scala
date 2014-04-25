package puck.parser.gen

import com.nativelibs4java.opencl._
import puck.util.{PointerFreer, ZipUtil, CLScan}
import puck.parser.{PruningMask, Batch, WorkSpace}
import org.bridj.Pointer
import puck.parser.Batch
import java.util.zip.{ZipFile, ZipOutputStream}

/**
 *
 *
 * @author dlwh
 */
class CLWorkQueueKernels(enqueueKernel: CLKernel, maskSize: Int)(implicit ctxt: CLContext) {
  private val scanKernel = CLScan.make

  def enqueue[W](ws: WorkSpace,
                 batch: Batch[W],
                 spanLength: Int,
                 pTop: Boolean,
                 lTop: Boolean,
                 rTop: Boolean,
                 mask: Option[Array[Int]],
                 events: CLEvent*)(implicit queue: CLQueue):(CLEvent, Int) = synchronized {
    val scratch = ctxt.createIntBuffer(CLMem.Usage.InputOutput, batch.numSentences + 1)
    def I(x: Boolean) = if(x) Integer.valueOf(1) else Integer.valueOf(0)
    enqueueKernel.setArgs(ws.pPtrBuffer, ws.lPtrBuffer, ws.rPtrBuffer, scratch,
      batch.cellOffsetsDev, batch.lengthsDev, batch.masksDev.map(_.data.safeBuffer).getOrElse(ws.lPtrBuffer), mask.getOrElse(Array.fill(maskSize)(-1)), I(mask.nonEmpty), I(pTop), I(lTop), I(rTop), Integer.valueOf(spanLength), I(true))

    val computeNeededEv = enqueueKernel.enqueueNDRange(queue, Array(batch.numSentences), Array(1), events:_*)


    val evScan = scanKernel.scan(ws.queueOffsets, scratch, batch.numSentences, computeNeededEv)
    PointerFreer.enqueue({scratch.release()}, evScan)

    enqueueKernel.setArg(3, ws.queueOffsets)
    enqueueKernel.setArg(13, I(false))

    val res = enqueueKernel.enqueueNDRange(queue, Array(batch.numSentences), Array(1), evScan)
    val numNeeded = ws.queueOffsets.read(queue, batch.numSentences - 1, 1, evScan).get()

    (res, numNeeded.intValue())
  }

  def write(prefix: String, out: ZipOutputStream) {
    ZipUtil.addKernel(out, s"$prefix/enqueue", enqueueKernel)
    ZipUtil.serializedEntry(out, s"$prefix/ints", Array(maskSize))
  }
}

object CLWorkQueueKernels {
  def forLoopType(numCoarseSyms: Int, loopType: LoopType)(implicit ctxt: CLContext):CLWorkQueueKernels = {
    loopType match {
      case LoopType.Inside => forInside(numCoarseSyms)
      case LoopType.InsideNT => forInsideNT(numCoarseSyms)
      case LoopType.InsideTN => forInsideTN(numCoarseSyms)
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
    new CLWorkQueueKernels(k, CLMaskKernels.maskSizeFor(numCoarseSyms))
  }

  def read(prefix: String, in: ZipFile)(implicit context: CLContext) = {
    val k = ZipUtil.readKernel(in, s"$prefix/enqueue")
    val ints = ZipUtil.deserializeEntry[Array[Int]](in.getInputStream(in.getEntry(s"$prefix/ints")))
    new CLWorkQueueKernels(k, ints(0))
  }

  def forInside(numCoarseSyms: Int)(implicit ctxt: CLContext) = {
    forSplitRange(numCoarseSyms, insideSplitRange)
  }

  def forInsideNT(numCoarseSyms: Int)(implicit ctxt: CLContext) = {
    forSplitRange(numCoarseSyms, insideNTSplitRange)
  }


  def forInsideTN(numCoarseSyms: Int)(implicit ctxt: CLContext) = {
    forSplitRange(numCoarseSyms, insideTNSplitRange)
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

    def insideNTSplitRange =
    """
      | range_t computeSplitRange(int begin, int end, int length) {
      |   range_t r= {end - 1, end - 1};
      |   return r;
      | }
    """.stripMargin


    def insideTNSplitRange =
    """
      | range_t computeSplitRange(int begin, int end, int length) {
      |   range_t r= {begin + 1, begin + 1};
      |   return r;
      | }
    """.stripMargin

  def outsideRSplitRange =
    """
      | range_t computeSplitRange(int begin, int end, int length) {
      |   range_t r = {0, begin - 1};
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
      |                       mask_t tMask,
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
      |
      |   int parentOffset = (parentTop) ? topChart : cellOffset;
      |   int rightOffset = (rightTop) ? topChart : cellOffset;
      |   int leftOffset = (leftTop) ? topChart : cellOffset;
      |
      |   int queueOffset = firstQueueOffset;
      |
      |   mask_t pMask;
      //|   mask_t tMask;
      //|   if(useMasks) tMask = *targetMask;
      |
      |   mask_t lMask;
      |   mask_t rMask;
      |
      |   int numNeeded = 0;
      |
      |   if(!justComputeNumNeeded && (nextQueueOffset - firstQueueOffset) == 0) return;
      |
      |   for(int begin = 0, end = spanLength; end <= length; begin++, end++) {
      |     int parentCell = parentOffset + computeCell(begin, end, length);
      |
      |     if(useMasks)
      |       pMask = masks[parentCell];
      |
//      |     if(useMasks) {
//      |       printf("%d %d :: %d intersects?\n", begin, end, maskIntersects(&pMask, &tMask));
//      |     }
      |
      |     if(!useMasks || maskIntersects(&pMask, &tMask)) {
      |       range_t splitRange = computeSplitRange(begin, end, length);
      |       splitRange.low = max(splitRange.low, 0);
      |       splitRange.high = min(splitRange.high, length);
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
      |         int includeThisOne = !useMasks;
      |         if(!includeThisOne) {
      |           lMask = masks[leftCell];
      |           int doLeft = maskAny(&lMask);
//      |           printf("left says %d\n",doLeft);
      |           if (doLeft) {
      |             rMask = masks[rightCell];
      |             int doRight = maskAny(&rMask);
//      |             printf("right says %d\n",doRight);
      |             if(doRight) includeThisOne = true;
      |           }
      |         }
      |
      |         if(includeThisOne) {
//      |            printf("<%d %d %d %d: p%d l%d r%d>\n",numNeeded,begin,split,end, parentCell, leftCell, rightCell);
      |            numNeeded++;
      |            if(!justComputeNumNeeded) {
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
//      |     printf("?? %d %d\n",sent, numNeeded);
      |     queueOffsets[sent] = numNeeded;
      |   }
      | }
    """.stripMargin
}