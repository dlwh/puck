package puck.parser.gen

import puck.parser.{SymId, RuleSemiring, RuleStructure}
import scala.collection.JavaConverters._
import puck.linalg.CLMatrix
import com.nativelibs4java.opencl._
import org.bridj.Pointer
import java.util.zip.{ZipOutputStream, ZipFile}
import puck.util.ZipUtil
import scala.Array

/**
 * TODO
 *
 * @author dlwh
 **/
case class CLMaskKernels(maskSize: Int, getMasksKernel: CLKernel) {


  def write(out: ZipOutputStream) {
    ZipUtil.addKernel(out, "computeMasksKernel", getMasksKernel)
    ZipUtil.serializedEntry(out, "MasksInts", Array(maskSize))
  }

  def getMasks(masks: CLMatrix[Int],
               inside: CLMatrix[Float],
               outside: CLMatrix[Float],
               firstOutside: Int,
               chartIndices: Array[Int],
               root: Int, threshold: Float,
               events: CLEvent*)(implicit queue: CLQueue):CLEvent = {
    require(masks.rows == maskSize, masks.rows + " " + maskSize)
    require(masks.cols == inside.cols)
    require(masks.cols == outside.cols)
    queue.finish()



    val ptrCI = Pointer.pointerToArray[java.lang.Integer](chartIndices)
    val intBufferCI = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, chartIndices.length)
    val evCI = intBufferCI.write(queue, 0, chartIndices.length, ptrCI, false, events:_*)

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

object CLMaskKernels {
  def read(zf: ZipFile)(implicit ctxt: CLContext) = {
    val ints = ZipUtil.deserializeEntry[Array[Int]](zf.getInputStream(zf.getEntry("MasksInts")))
    CLMaskKernels(ints(0), ZipUtil.readKernel(zf, "computeMasksKernel"))
  }

  def make[C, L](structure: RuleStructure[C, L])(implicit context: CLContext, semiring: RuleSemiring) = {
    val cellSize = (structure.numNonTerms max structure.numTerms)
    val maskSize = puck.roundUpToMultipleOf(structure.numCoarseSyms, 32) / 32

    val prog = context.createProgram(programText(cellSize, structure))

    CLMaskKernels(maskSize, prog.createKernel("computeMasks"))
  }


  def maskHeader[C, L](structure: RuleStructure[C, L]) = {
    val maskSize = puck.roundUpToMultipleOf(structure.numCoarseSyms, 32) / 32
    """#define NUM_FIELDS """ + maskSize + """

  typedef struct { int fields[NUM_FIELDS]; } mask_t;

  inline void set_bit(mask_t* mask, int bit, int shouldSet) {
    int field = (bit/32);
    int modulus = bit%32;
    mask->fields[field] = mask->fields[field] | (shouldSet<<(modulus));
  }

  /* Intel gets sad from this one?
  inline int is_set(mask_t* mask, int bit) {
    int field = (bit/32);
    int modulus = bit%32;
    return mask->fields[field] & (1<<(modulus));
  }
  */

   #define is_set(mask, bit)  ((mask)->fields[(bit)/32] & (1<<((bit)%32)))

                                           """
  }


  def generateCheckMaskString[C, L](structure: RuleStructure[C, L],
                                    nameOfMaskVariable: String,
                                    symbols: java.util.Set[SymId[C, L]]): String = {
    generateCheckMaskString(structure, nameOfMaskVariable, symbols.asScala.toSet)
  }

  def generateCheckMaskString[C, L](structure: RuleStructure[C, L],
                               nameOfMaskVariable: String,
                               symbols: Set[SymId[C, L]]): String = {
    // set up the mask
    val maskStrings = for {
      (field, parentsInField) <- symbols
        .map(s => structure.refinements.labels.project(s.system))
        .groupBy(_ / 32)
    } yield parentsInField.map(p => s"(1<<($p%32))").mkString(s"$nameOfMaskVariable.fields[$field] & (", "|", ")")

    maskStrings.mkString("if (!((", ") | (", ")) ) return;")
  }

  def programText[L, C](cellSize: Int, structure: RuleStructure[C, L]): String = {


    maskHeader(structure) ++ """
      #define NUM_SYMS """ + cellSize + """

                                        """ + structure.projectedTerminalMap.padTo(cellSize, 0).mkString("__constant int terminalProjections[] = {", ", ", "};") +
      """
      """ + structure.projectedNonterminalMap.padTo(cellSize, 0).mkString("__constant int nonterminalProjections[] = {", ", ", "};") +
      """

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
/** TODO this isn't optimized at all */
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
    __constant const int* projections = (masksOut[cell].fields[0] == 0) ? nonterminalProjections : terminalProjections;

    __global const float* in = inside + (cell * numSyms);
    __global const float* out = outside + (cell * numSyms);
    mask_t myMask;
    for(int i = 0; i < NUM_FIELDS; ++i) {
      myMask.fields[i] = 0;
    }

    for(int sym = 0; sym < NUM_SYMS; ++sym) {
      float score = (in[sym] + out[sym]);
      int keep = score >= cutoff;
      int field = projections[sym];

      //if(cell == lastCell - 1 && score != -INFINITY)
      //  printf("%d %d %d %f %f\n", lastCell, sym, field, score, cutoff);

      set_bit(&myMask, field, keep);
    }


    masksOut[cell] = myMask;
    masksOut[cell + _outsideOff/numSyms] = myMask;
//    printf("%d %d\n", cell, cell + _outsideOff/numSyms);
  }

}
      """
  }
}
