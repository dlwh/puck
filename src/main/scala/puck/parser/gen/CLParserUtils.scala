package puck.parser.gen

import com.nativelibs4java.opencl._
import java.util.zip.{ZipFile, ZipOutputStream}
import puck.util.ZipUtil
import puck.linalg.CLMatrix
import org.bridj.Pointer
import puck.parser.{RuleSemiring, RuleStructure}

case class CLParserUtils(sumGrammarKernel: CLKernel, sumSplitPointsKernel: CLKernel, setRootScoresKernel: CLKernel, getMasksKernel: CLKernel,
                               splitPointsBlockSize: Int, groupSize: Int, fieldSize: Int) {
  def write(out: ZipOutputStream) {
    ZipUtil.addKernel(out, "sumGrammarKernel", sumGrammarKernel)
    ZipUtil.addKernel(out, "sumSplitPointsKernel", sumSplitPointsKernel)
    ZipUtil.addKernel(out, "setRootScoresKernel", setRootScoresKernel)
    ZipUtil.addKernel(out, "getMasksKernel", getMasksKernel)
    ZipUtil.serializedEntry(out, "ints", Array(splitPointsBlockSize, groupSize, fieldSize))
  }

  def sumSplitPoints(parent: CLMatrix[Float], chart: CLMatrix[Float], chartIndices: Array[Int], parentIndices: Array[Int], chartIndicesPerGroup: Int, events: CLEvent*)(implicit queue: CLQueue) = {
    require(chartIndices.length == parentIndices.length - 1)
    val parentStride = parent.rows
    val numSyms = parent.cols

    val ptrCI = Pointer.pointerToArray[java.lang.Integer](chartIndices)
    val intBufferCI = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, chartIndices.length)
    val evCI = intBufferCI.write(queue, 0, chartIndices.length, ptrCI, false, events:_*)

    val ptrPI = Pointer.pointerToArray[java.lang.Integer](parentIndices)
    val intBufferPI = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, parentIndices.length)
    val evPI = intBufferPI.write(queue, 0, parentIndices.length, ptrPI, false, events:_*)

    sumSplitPointsKernel.setArgs(parent.data.safeBuffer, chart.data.safeBuffer, intBufferCI, intBufferPI,
      Integer.valueOf(parentStride), Integer.valueOf(chartIndices.length), Integer.valueOf(chartIndicesPerGroup),
      Integer.valueOf(numSyms))

    val numGroups = (chartIndices.length + chartIndicesPerGroup - 1)/chartIndicesPerGroup * chartIndicesPerGroup
    val rowBlocks = (numSyms + splitPointsBlockSize - 1)/splitPointsBlockSize

    val ev = sumSplitPointsKernel.enqueueNDRange(queue, Array(numGroups * groupSize, rowBlocks), Array(groupSize, 1), Seq(evCI, evPI)++events :_*)

    ev.invokeUponCompletion(new Runnable() {
      def run() = { ptrCI.release(); intBufferCI.release(); ptrPI.release(); intBufferPI.release(); }
    })
    ev
  }

  def setRootScores(charts: CLMatrix[Float],
                    chartIndices: Array[Int],
                    root: Int,
                    one: Float,
                    events: CLEvent*)(implicit queue: CLQueue):CLEvent = {

    val ptrCI = Pointer.pointerToArray[java.lang.Integer](chartIndices)
    val intBufferCI = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, chartIndices.length)
    val evCI = intBufferCI.write(queue, 0, chartIndices.length, ptrCI, false, events:_*)

    setRootScoresKernel.setArgs(charts.data.safeBuffer, intBufferCI,
      Integer.valueOf(chartIndices.length), Integer.valueOf(charts.rows),
      Integer.valueOf(root), java.lang.Float.valueOf(one))

    val ev = setRootScoresKernel.enqueueNDRange(queue, Array(chartIndices.length), evCI)

    ev.invokeUponCompletion(new Runnable() {
      def run() = { ptrCI.release(); intBufferCI.release();}
    })
    ev
  }

  def getMasks(masks: CLMatrix[Int],
               inside: CLMatrix[Float],
               outside: CLMatrix[Float],
               firstOutside: Int,
               chartIndices: Array[Int],
               root: Int, threshold: Float,
               events: CLEvent*)(implicit queue: CLQueue):CLEvent = {
    require(masks.rows == fieldSize)
    require(masks.cols == inside.cols)
    require(masks.cols == outside.cols)
    queue.finish()

    val ptrCI = Pointer.pointerToArray[java.lang.Integer](chartIndices)
    val intBufferCI = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, chartIndices.length)
    val evCI = intBufferCI.write(queue, 0, chartIndices.length, ptrCI, false, events:_*)

    println(chartIndices.toIndexedSeq)

    getMasksKernel.setArgs(masks.data.safeBuffer,
      inside.data.safeBuffer, outside.data.safeBuffer, Integer.valueOf(outside.offset), intBufferCI,
      Integer.valueOf(chartIndices(chartIndices.length-1)), Integer.valueOf(inside.rows),
      Integer.valueOf(root), java.lang.Float.valueOf(threshold))
      //, LocalSize.ofIntArray(fieldSize * groupSize * 5))

    val ev = getMasksKernel.enqueueNDRange(queue, Array(chartIndices.length-1, 1), Array(1, 1), evCI)

    ev.invokeUponCompletion(new Runnable() {
      def run() = { ptrCI.release(); intBufferCI.release();}
    })
    ev
  }


}

object CLParserUtils {
  def read(zf: ZipFile)(implicit ctxt: CLContext) = {
    val ints = ZipUtil.deserializeEntry[Array[Int]](zf.getInputStream(zf.getEntry("ints")))
    CLParserUtils(ZipUtil.readKernel(zf, "sumGrammarKernel"),
      ZipUtil.readKernel(zf, "sumSplitPointsKernel"),
      ZipUtil.readKernel(zf, "setRootScoresKernel"),
      ZipUtil.readKernel(zf, "getMasksKernel"),
      ints(0), ints(1), ints(2)
    )
  }

  def make[C, L](structure: RuleStructure[C, L])(implicit context: CLContext, semiring: RuleSemiring) = {
    lazy val sumCellsKernel = context.createProgram {
      s"""
      |__kernel void sumCells(__global float* dest, int destOff, int destRowSize,
      |                       __global float* src, int srcOff, int srcRowSize, int numLabels, int rowsToDo) {
      |    int row = get_global_id(0);
      |    int label = get_global_id(1);
      |    if ((row < rowsToDo) & (label < numLabels)) {
      |      float score = ${semiring.add("dest[label * destRowSize + row + destOff]", "src[label * srcRowSize + row + srcOff]")};
      |      dest[label * destRowSize + row + destOff] = score;
      |    }
      |}
    """.stripMargin
    }.build().createKernels().head


    val blockSize = 32
    val groupSize = if(context.getDevices.head.toString.contains("Apple") && context.getDevices.head.toString.contains("Intel")) {
      1
    } else {
      val x = context.getDevices.head.getMaxWorkItemSizes
      val size0 = x(0)
      math.min(size0, blockSize).toInt
    }

    val numSyms = structure.termIndex.size.max(structure.nontermIndex.size)
    val numSymsRounded = (numSyms + 31)/32 * 32

    val prog = context.createProgram(splitPointSumKernel(blockSize, numSymsRounded, numSyms))

    CLParserUtils(sumCellsKernel,
      prog.createKernel("splitPointSum"),
      prog.createKernel("setRootScores"),
      prog.createKernel("computeMasks"),
      blockSize, groupSize, numSymsRounded/32)
  }

  def splitPointSumKernel(blockSize: Int, numSymsRounded: Int, numSyms: Int) = {
    """#define BLOCK_SIZE """ + blockSize + """

   float sumUp(__local float* scores, float _acc, int first, int last) {
     float m = _acc;
     for(int i = first; i < last; i += 1) {
        m = max(m, scores[i]);
     }

#ifdef LOGSUM
     if(m != -INFINITY) {
       float adj = exp(_acc - m);
       for(int i = first; i < last; i += 1) {
         adj += exp(scores[i] - m);
       }
       m += log(adj);
     }
#endif

     return m;
   }

__kernel void splitPointSum(__global float* parent, __global float* chart,
                        __global int* chartIndex, __global int* parentIndex,
                        int parentStride, int numChartIndices, int chartIndicesPerGroup, int numSyms) {

  int groupid = get_group_id(0);
  int tid = get_local_id(0);
  int numThreads = get_local_size(0);

  int totalChartIndicesToDo = clamp(numChartIndices - groupid * chartIndicesPerGroup, 0, chartIndicesPerGroup);

  __local int myParentIndices[BLOCK_SIZE+1];
  __local int myChartIndices[BLOCK_SIZE];

  int lastChartIndex = groupid * chartIndicesPerGroup + totalChartIndicesToDo;
  for(int firstChartIndex = groupid * chartIndicesPerGroup; firstChartIndex < lastChartIndex; firstChartIndex += BLOCK_SIZE) {
    int chartIndicesToDo = clamp(lastChartIndex - firstChartIndex, 0, BLOCK_SIZE);

    event_t e_parents = async_work_group_copy(myParentIndices, parentIndex + firstChartIndex, chartIndicesToDo + 1, 0);
    // TODO: intel needs this here and not after the next line, for some reason
    wait_group_events(1, &e_parents);
    event_t e_charts = async_work_group_copy(myChartIndices, chartIndex + firstChartIndex, chartIndicesToDo, 0);

    int numRowsToDo = myParentIndices[chartIndicesToDo] - myParentIndices[0];
    int rowOffset = myParentIndices[0];

    int firstSym = get_global_id(1) * BLOCK_SIZE;
    int lastSym = min(firstSym + BLOCK_SIZE, numSyms);

    __local float scores[BLOCK_SIZE][BLOCK_SIZE+1];

    wait_group_events(1, &e_charts);

    int currentChartIndex = 0;
    // each row is a single split point.
    for(int firstRow = 0; firstRow < numRowsToDo; firstRow += BLOCK_SIZE) {
      int todo = min(numRowsToDo - firstRow, BLOCK_SIZE);
      // copy scores in
      for(int sym = firstSym;  sym < lastSym; sym += 1) {
        int localSym = sym - firstSym;
        event_t event = async_work_group_copy(scores[localSym], parent + parentStride * sym + rowOffset + firstRow, todo, 0);
        wait_group_events(1, &event);
      }

      // at this point, todo columns of scores are read in.
      // min(myParentIndices[currentChartIndex+1] - myParentIndices[0] - firstRow, todo) belong to the currentChartIndex.
      // what's left belongs to the next chartIndex (or the ones thereafter)
      // TODO: if the number of rows for one chart index is bigger than BLOCK_SIZE, then we'll need to
      // write to global memory multiple times. we might consider caching in local memory instead.
      // this will only happen on nvidia cards for numSplits > 32, and since we're mostly doing length 40,
      // whatever.
      int row = 0;
      while(currentChartIndex < chartIndicesToDo && row < todo) {
        int lastRowForThisChartCell = min(myParentIndices[currentChartIndex+1] - myParentIndices[0] - firstRow, todo);
        // for each symbol, sum over all rows (split points) for this chart index
        // then flush the score back to the right place for this chart symbol.
        for(int sym = tid + firstSym;  sym < lastSym; sym += numThreads) {
          int localSym = sym - firstSym;
          float result = sumUp(scores[localSym], chart[myChartIndices[currentChartIndex] * numSyms + sym], row, lastRowForThisChartCell);
          chart[myChartIndices[currentChartIndex] * numSyms + sym] = result;
        }

        row = lastRowForThisChartCell;

        if (row >= myParentIndices[currentChartIndex+1] - myParentIndices[0] - firstRow) {
          ++currentChartIndex;
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    }

  }
}


__kernel void setRootScores(__global float* charts, __global int* indices, int numIndices, int numSyms, int root, float value) {
  int id = get_global_id(0);
  if(id < numIndices)
      charts[numSyms * indices[id] + root] = value;
}


#define NUM_SYMS """ + numSyms + """
#define NUM_FIELDS """ + ( (numSyms + 31)/32) + """
typedef struct { int fields[NUM_FIELDS]; } mask_t;

inline void set_bit(int* field, int bit, int shouldSet) {
    *field = (*field & ~(1 << (bit))) | (shouldSet<<(bit));
}

// each global_id(0) corresponds to a single sentence.
// we have some number of workers for each sentence, global_size(1)
// for each cell in the sentence, each worker in parallel reads a sym from a cell, thresholds it, and then sets
// the mask if the threshold is exceeded. Each worker has its own mask for its share of the cells. At the
// end, the masks are or'd together and written out.
// the masks are then
// indices(i) is the first cell in the i'th sentence
// indices(i+1)-1 is the last cell in the i'th sentence
// the last cell has the root score.
//
__kernel void computeMasks(__global mask_t* masksOut,
                           __global const float* inside,
                           __global const float* _outside,
                           const int _outsideOff,
                           __global const int* indices,
                           const int numIndices,
                           int numSyms,
                           int root,
                           float thresh) {
  const int sentence = get_global_id(0);
  const int firstCell = indices[sentence];
  const int lastCell = indices[sentence + 1];
  __global const float* outside = _outside + _outsideOff;
  const float root_score = inside[(lastCell-1) * numSyms + root];

  float cutoff = root_score + thresh;

  for(int cell = firstCell; cell < lastCell; cell++) {
    __global const float* in = inside + (cell * numSyms);
    __global const float* out = outside + (cell * numSyms);
    mask_t myMask;
    for(int i = 0; i < NUM_FIELDS; ++i) {
      myMask.fields[i] = 0;
    }

    for(int sym = 0; sym < NUM_SYMS; ++sym) {
      float score = (in[sym] + out[sym]);
      int keep = score >= cutoff;

      set_bit(myMask.fields + (sym/32), sym%32, keep);
    }
    masksOut[cell] = myMask;
  }

}
                                                """



  }
}


