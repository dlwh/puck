package puck.parser.gen

import puck.roundUpToMultipleOf

import com.nativelibs4java.opencl._
import java.util.zip.{ZipFile, ZipOutputStream}
import puck.util.ZipUtil
import puck.linalg.CLMatrix
import org.bridj.Pointer
import puck.parser.{ViterbiRuleSemiring, RuleSemiring, RuleStructure}

case class CLParserUtils(sumGrammarKernel: CLKernel, sumSplitPointsKernel: CLKernel, setRootScoresKernel: CLKernel,
                               splitPointsBlockSize: Int, groupSize: Int) {
  def write(out: ZipOutputStream) {
    ZipUtil.addKernel(out, "sumGrammarKernel", sumGrammarKernel)
    ZipUtil.addKernel(out, "sumSplitPointsKernel", sumSplitPointsKernel)
    ZipUtil.addKernel(out, "setRootScoresKernel", setRootScoresKernel)
    ZipUtil.serializedEntry(out, "ints", Array(splitPointsBlockSize, groupSize))
  }

  def sumSplitPoints(parent: CLMatrix[Float], chart: CLMatrix[Float], chartIndices: CLBuffer[Integer], numUniqueParents: Int, splitPointIndicesIntoWorkArray: CLBuffer[Integer], uniqueIndicesPerGroup: Int, numSymsToDo: Int, events: CLEvent*)(implicit queue: CLQueue) = {
    val parentStride = parent.rows
    val majorStride = parent.cols

    sumSplitPointsKernel.setArgs(parent.data.safeBuffer, chart.data.safeBuffer, chartIndices, splitPointIndicesIntoWorkArray,
      Integer.valueOf(parentStride), Integer.valueOf(numUniqueParents), Integer.valueOf(groupSize),//Integer.valueOf(uniqueIndicesPerGroup),
     Integer.valueOf(numSymsToDo),   Integer.valueOf(majorStride))

    val numGroups = roundUpToMultipleOf(numUniqueParents, 32)/groupSize//uniqueIndicesPerGroup)
    val rowBlocks = (numSymsToDo + splitPointsBlockSize - 1)/splitPointsBlockSize


    sumSplitPointsKernel.enqueueNDRange(queue, Array(numGroups * groupSize, rowBlocks), Array(groupSize, 1), events :_*)
  }

  def setRootScores(charts: CLMatrix[Float],
                    chartIndices: CLBuffer[Integer], numUniqueParents: Int,
                    root: Int,
                    one: Float,
                    events: CLEvent*)(implicit queue: CLQueue):CLEvent = {

    setRootScoresKernel.setArgs(charts.data.safeBuffer, chartIndices,
      Integer.valueOf(numUniqueParents), Integer.valueOf(charts.rows),
      Integer.valueOf(root), java.lang.Float.valueOf(one))

    setRootScoresKernel.enqueueNDRange(queue, Array(numUniqueParents), events:_*)
  }

}

object CLParserUtils {
  def read(zf: ZipFile)(implicit ctxt: CLContext) = {
    val structure = ZipUtil.deserializeEntry[RuleStructure[_, _]](zf.getInputStream(zf.getEntry("structure")))
    implicit val semi = ViterbiRuleSemiring
    make(structure)
//    val ints = ZipUtil.deserializeEntry[Array[Int]](zf.getInputStream(zf.getEntry("ints")))
//    CLParserUtils(ZipUtil.readKernel(zf, "sumGrammarKernel"),
//      ZipUtil.readKernel(zf, "sumSplitPointsKernel"),
//      ZipUtil.readKernel(zf, "setRootScoresKernel"),
//      ints(0), ints(1)
//    )
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
    val groupSize = if(context.getDevices.head.toString.contains("Apple") && context.getDevices.head.toString.contains("Intel") && context.getDevices.head.toString.contains("Core")) {
      1
    } else {
      val x = context.getDevices.head.getMaxWorkItemSizes
      val size0 = x(0)
      math.min(size0, blockSize).toInt
    }

    val prog = context.createProgram(splitPointSumKernel(blockSize, structure))

    CLParserUtils(sumCellsKernel,
      prog.createKernel("splitPointSum"),
      prog.createKernel("setRootScores"),
      blockSize, groupSize)
  }

  def splitPointSumKernel[C, L](blockSize: Int, structure: RuleStructure[C, L]) = {
    """#define BLOCK_SIZE """ + blockSize + """

   inline float sumUp(__local float* scores, float _acc, int first, int last) {
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

// sum split points from the parent array into the place they go in chart.
// the i'th entry in chartIndex tells us which index into chart the row in parent is associated with
// rows in the work array are blocked so that all rows that index to the same chart cell are next to one another.
// splitPointIndex tells us where each one begins and ends. splitPointIndex is of length numUniqueParents+1
__kernel void splitPointSum(__global float* parent, __global float* chart,
                        __global int* chartIndex, __global int* splitPointIndex,
                        int parentStride, int numUniqueParents, int uniqueIndicesPerGroup, int numSyms, int majorStride) {

  int groupid = get_group_id(0);
  int tid = get_local_id(0);
  int numThreads = get_local_size(0);

  int uniqueParentsToDo = clamp(numUniqueParents - groupid * uniqueIndicesPerGroup, 0, uniqueIndicesPerGroup);

  __local int mySplitPointIndices[BLOCK_SIZE+1];

  int lastUniqueParent = groupid * uniqueIndicesPerGroup + uniqueParentsToDo;
  for(int firstUniqueParent = groupid * uniqueIndicesPerGroup; firstUniqueParent < lastUniqueParent; firstUniqueParent += BLOCK_SIZE) {
    int parentsToDo = clamp(lastUniqueParent - firstUniqueParent, 0, BLOCK_SIZE);

    event_t e_parents = async_work_group_copy(mySplitPointIndices, splitPointIndex + firstUniqueParent, parentsToDo + 1, 0);
    // TODO: intel needs this here and not after the next line, for some reason
    wait_group_events(1, &e_parents);

    int numRowsToDo = mySplitPointIndices[parentsToDo] - mySplitPointIndices[0];
    int rowOffset = mySplitPointIndices[0];

    int firstSym = get_global_id(1) * BLOCK_SIZE;
    int lastSym = min(firstSym + BLOCK_SIZE, numSyms);

    __local float scores[BLOCK_SIZE][BLOCK_SIZE+1];

    int currentParent = 0;
    // each row is a single split point.
    for(int firstRow = 0; firstRow < numRowsToDo; firstRow += BLOCK_SIZE) {

      int todo = min(numRowsToDo - firstRow, BLOCK_SIZE);
      // copy scores in from the global parent array to local storage, so we can transpose
      for(int sym = firstSym;  sym < lastSym; sym += 1) {
        int localSym = sym - firstSym;
        event_t event = async_work_group_copy(scores[localSym], parent + parentStride * sym + rowOffset + firstRow, todo, 0);
        // TODO: should be able to batch these up... but stupid intel is stupid
        wait_group_events(1, &event);
//          for(int row = tid; row < todo; row += numThreads) {
//            scores[localSym][row] = parent[parentStride * sym + rowOffset + firstRow + row];
//          }
      }

      // at this point, todo columns of scores are read in.
      // min(mySplitPointIndices[currentParent+1] - mySplitPointIndices[0] - firstRow, todo) belong to the currentParent.
      // what's left belongs to the next chartIndex (or the ones thereafter)
      // TODO: if the number of rows for one chart index is bigger than BLOCK_SIZE, then we'll need to
      // write to global memory multiple times. we might consider caching in local memory instead.
      // this will only happen on nvidia cards for numSplits > 32, and since we're mostly doing length 40,
      // whatever.
      int row = 0;
      while(currentParent < parentsToDo && row < todo) {
        __global float* chartCell = chart + majorStride * chartIndex[rowOffset + firstRow + row];
        int lastRowForThisChartCell = min(mySplitPointIndices[currentParent+1] - mySplitPointIndices[0] - firstRow, todo);
        // for each symbol, sum over all rows (split points) for this chart index
        // then flush the score back to the right place for this chart symbol.
        for(int sym = tid + firstSym;  sym < lastSym; sym += numThreads) {
          int localSym = sym - firstSym;
          float result = sumUp(scores[localSym], chartCell[sym], row, lastRowForThisChartCell);
          chartCell[sym] = result;
        }

        row = lastRowForThisChartCell;

        if (row >= mySplitPointIndices[currentParent+1] - mySplitPointIndices[0] - firstRow) {
          ++currentParent;
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

                                            """



  }
}


