package puck.newparser.generator

import puck.util._
import com.nativelibs4java.opencl._
import java.util.zip._
import scala.collection.JavaConverters._
import trochee.basic._
import trochee.kernels._
import scala.reflect.runtime.universe._
import puck.linalg._

case class CLInsideKernels(insideNNKernels: IndexedSeq[CLKernel],
                           insideNTKernels: IndexedSeq[CLKernel],
                           insideTNKernels: IndexedSeq[CLKernel],
                           insideTTKernels: IndexedSeq[CLKernel],
                           insideNUKernels: IndexedSeq[CLKernel],
                           insideTUKernels: IndexedSeq[CLKernel]) {

  def write(out: ZipOutputStream) {
    ZipUtil.addKernelSet(out, "insideNN", insideNNKernels)
    ZipUtil.addKernelSet(out, "insideNT", insideNTKernels)
    ZipUtil.addKernelSet(out, "insideTN", insideTNKernels)
    ZipUtil.addKernelSet(out, "insideTT", insideTTKernels)
    ZipUtil.addKernelSet(out, "insideNU", insideNUKernels)
    ZipUtil.addKernelSet(out, "insideTU", insideTUKernels)
  }
}

object CLInsideKernels {
  def read(in: ZipFile)(implicit context: CLContext) = {
    val insideNN = ZipUtil.readKernelSet(in, "insideNN")
    val insideNT = ZipUtil.readKernelSet(in, "insideNT")
    val insideTN = ZipUtil.readKernelSet(in, "insideTN")
    val insideTT = ZipUtil.readKernelSet(in, "insideTT")
    val insideNU = ZipUtil.readKernelSet(in, "insideNU")
    val insideTU = ZipUtil.readKernelSet(in, "insideTU")
    CLInsideKernels(insideNN, insideNT, insideTN, insideTT, insideNU, insideTU)
  }


  def make[C, L](parserGen: CLParserKernelGenerator[C, L])(implicit context: CLContext) = {
    import parserGen._
    val insideNNKernels = structure.partitionsParent.zipWithIndex.map { case(partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition, "inside_nn_binaries_"+i))
    }

    val insideNTKernels = structure.partitionsRightTermRules.zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition, "inside_nt_binaries_"+i))
    }

    val insideTNKernels = structure.partitionsLeftTermRules.zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition, "inside_tn_binaries"+i))
    }

    val insideTTKernels = structure.partitionsBothTermRules.zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition, "inside_tt_binaries_"+i))
    }

    val insideNUKernels = IndexedSeq(structure.unaryRules).zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.unaryRuleKernel(partition, "inside_nn_unaries"+i))
    }

    val insideTUKernels = IndexedSeq(structure.unaryTermRules).zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.unaryRuleKernel(partition, "inside_nt_unaries"+i))
    }

    CLInsideKernels(insideNNKernels,
                    insideNTKernels,
                    insideTNKernels,
                    insideTTKernels,
                    insideNUKernels,
                    insideTUKernels)
  }
}

case class CLParserUtilKernels(sumGrammarKernel: CLKernel, sumSplitPointsKernel: CLKernel, splitPointsBlockSize: Int) {
  def write(out: ZipOutputStream) {
    ZipUtil.addKernel(out, "sumGrammarKernel", sumGrammarKernel)
    ZipUtil.addKernel(out, "sumSplitPointsKernel", sumSplitPointsKernel)
    sys.error("haven't figured out the right way to store ints!")
  }

  def sumSplitPoints(parent: CLMatrix[Float], chart: CLMatrix[Float], chartIndices: Array[Int], parentIndices: Array[Int], chartIndicesPerGroup: Int, events: CLEvent*)(implicit queue: CLQueue) = {
    require(chartIndices.length == parentIndices.length - 1)
    val parentStride = parent.rows
    val numSyms = parent.cols
    sumSplitPointsKernel.setArgs(parent.data, chart.data, chartIndices,  parentIndices,
      Integer.valueOf(parentStride), Integer.valueOf(chartIndices.length), Integer.valueOf(chartIndicesPerGroup min splitPointsBlockSize), 
      Integer.valueOf(numSyms))
      val workSize = (chartIndices.length + splitPointsBlockSize - 1)/splitPointsBlockSize * splitPointsBlockSize
      val rowBlocks = (numSyms + splitPointsBlockSize - 1)/splitPointsBlockSize
    sumSplitPointsKernel.enqueueNDRange(queue, Array(workSize, rowBlocks), Array(splitPointsBlockSize, 1), events:_*)
  }

}

object CLParserUtilKernels {
  def read(zf: ZipFile)(implicit ctxt: CLContext) = {
    sys.error("haven't figured out the right way to store ints!")
    //CLParserUtilKernels(ZipUtil.readKernel(zf, "sumGrammarKernel"), ZipUtil.readKernel(zf, "sumSplitPointsKernel"))
  }

  def make[C, L](generator: CLParserKernelGenerator[C, L])(implicit context: CLContext) = {
  val preferredBlockSize = 32
    val blockSize = if( context.getDevices.head.toString.contains("Apple") && context.getDevices.head.toString.contains("Intel")) {
      1
    } else {
      val x = context.getDevices.head.getMaxWorkItemSizes
      val size0 = x(0)
      math.min(size0, preferredBlockSize).toInt
    }

    CLParserUtilKernels(generator.gen.mkKernel(generator.gen.IR.sumGrammarCellsKernel), context.createProgram(splitPointSumKernel(blockSize)).createKernel("splitPointSum"), blockSize)
  }

  def splitPointSumKernel(blockSize: Int) = {
    """#define BLOCK_SIZE """ + blockSize + """

   static float sumUp(__local float* scores, float _acc, int first, int last) {
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

  int firstChartIndex = groupid * chartIndicesPerGroup;
  int chartIndicesToDo = clamp(numChartIndices - firstChartIndex, 0, chartIndicesPerGroup);

  __local int myParentIndices[BLOCK_SIZE+1];
  __local int myChartIndices[BLOCK_SIZE];

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
  // rows = single split point.
  for(int firstRow = 0; firstRow < numRowsToDo; firstRow += BLOCK_SIZE) {
    int todo = firstRow + min(numRowsToDo - firstRow, BLOCK_SIZE);
    // copy scores in
    for(int sym = firstSym;  sym < lastSym; sym += 1) {
      int localSym = sym - firstSym;
      event_t event = async_work_group_copy(scores[localSym], parent + parentStride * sym + rowOffset, todo, 0);
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

      int todoForThisChartCell = min(myParentIndices[currentChartIndex+1]- myParentIndices[0] - firstRow, todo);
      // for each symbol, sum over all rows (split points) for this chart index
      // then flush the score back to the right place for this chart symbol.
      for(int sym = tid + firstSym;  sym < lastSym; sym += numThreads) {
        int localSym = sym - firstSym; 
        float result = sumUp(scores[localSym], chart[myChartIndices[currentChartIndex] * numSyms + sym], row, row + todoForThisChartCell);
        chart[myChartIndices[currentChartIndex] * numSyms + sym] = result;
      }

      row += todoForThisChartCell;

      if (row >= myParentIndices[currentChartIndex+1] - myParentIndices[0] - firstRow) {
        ++currentChartIndex;
      }
    }
  }

}"""
  }
}


