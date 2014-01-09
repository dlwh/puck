package puck.parser.gen

import puck.parser.{LogSumRuleSemiring, SymId, RuleSemiring, RuleStructure}
import scala.collection.JavaConverters._
import puck.linalg.CLMatrix
import com.nativelibs4java.opencl._
import org.bridj.Pointer
import java.util.zip.{ZipOutputStream, ZipFile}
import puck.util.ZipUtil
import scala.Array
import puck.PointerFreer

/**
 * TODO
 *
 * @author dlwh
 **/
case class CLMBRKernels(maskSize: Int, getMasksKernel: CLKernel) {


  def write(out: ZipOutputStream) {
    ZipUtil.addKernel(out, "computeMBRKernel", getMasksKernel)
    ZipUtil.serializedEntry(out, "MasksInts", Array(maskSize))
  }

  def getMasks(masks: CLMatrix[Int],
               inside: CLMatrix[Float],
               outside: CLMatrix[Float],
               chartIndices: Array[Int],
               lengths: Array[Int],
               root: Int,
               events: CLEvent*)(implicit queue: CLQueue):CLEvent = {
    require(masks.rows == maskSize, masks.rows + " " + maskSize)
    require(masks.cols == inside.cols)
    require(masks.cols == outside.cols)
    queue.finish()



    val ptrCI = Pointer.pointerToArray[java.lang.Integer](chartIndices)
    val intBufferCI = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, chartIndices.length)
    val evCI = intBufferCI.write(queue, 0, chartIndices.length, ptrCI, false, events:_*)

    val ptrL = Pointer.pointerToArray[java.lang.Integer](lengths)
    val intBufferL = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, lengths.length)
    val evL = intBufferL.write(queue, 0, lengths.length, ptrL, false, events:_*)

    getMasksKernel.setArgs(masks.data.safeBuffer,
      inside.data.safeBuffer, outside.data.safeBuffer, intBufferCI, intBufferL,
      Integer.valueOf(chartIndices(chartIndices.length-1)), Integer.valueOf(inside.rows),
      Integer.valueOf(root))
    //, LocalSize.ofIntArray(fieldSize * groupSize * 5))

    val ev = getMasksKernel.enqueueNDRange(queue, Array(chartIndices.length-1, 1), Array(1, 1), evCI, evL)
//    queue.finish()
    PointerFreer.enqueue(ptrCI.release(), ev)
    PointerFreer.enqueue(intBufferCI.release(), ev)

    PointerFreer.enqueue(ptrL.release(), ev)
    PointerFreer.enqueue(intBufferL.release(), ev)
    ev
  }

}

object CLMBRKernels {
  def read(zf: ZipFile)(implicit ctxt: CLContext) = {
    val ints = ZipUtil.deserializeEntry[Array[Int]](zf.getInputStream(zf.getEntry("MasksInts")))
    CLMBRKernels(ints(0), ZipUtil.readKernel(zf, "computeMBRKernel"))
  }

  def make[C, L](structure: RuleStructure[C, L])(implicit context: CLContext, semiring: RuleSemiring) = {
    val cellSize = (structure.numNonTerms max structure.numTerms)
    val maskSize = puck.roundUpToMultipleOf(structure.numCoarseSyms, 32) / 32

    val prog = context.createProgram(programText(cellSize, structure, semiring))

    CLMBRKernels(maskSize, prog.createKernel("computeMBR"))
  }


  def genCheckIfMaskIsEmpty[C, L](structure: RuleStructure[C, L],
                                    nameOfMaskVariable: String,
                                    symbols: java.util.Set[SymId[C, L]]): String = {
    genCheckIfMaskIsEmpty(structure, nameOfMaskVariable, symbols.asScala.toSet)
  }

  def genCheckIfMaskIsEmpty[C, L](structure: RuleStructure[C, L],
                               nameOfMaskVariable: String,
                               symbols: Set[SymId[C, L]]): String = {
    // set up the mask
    val maskStrings = for {
      (field, parentsInField) <- symbols
        .map(s => structure.refinements.labels.project(s.system))
        .groupBy(_ / 32)
    } yield parentsInField.map(p => s"(1<<($p%32))").mkString(s"$nameOfMaskVariable.fields[$field] & (", "|", ")")

    maskStrings.mkString("(!((", ") | (", ")) )")
  }

  def programText[L, C](cellSize: Int, structure: RuleStructure[C, L], semiring: RuleSemiring): String = {
    """
    typedef struct {float score; int symbol; int ignoreMe[2];} decode_t;
      #define NUM_SYMS """ + cellSize + """

                                        """ + structure.projectedTerminalMap.padTo(cellSize, 0).mkString("__constant int terminalProjections[] = {", ", ", "};") +
      """
      """ + structure.projectedNonterminalMap.padTo(cellSize, 0).mkString("__constant int nonterminalProjections[] = {", ", ", "};") +
      """

// each global_id(0) corresponds to a single sentence.
// we have some number of workers for each sentence, global_size(1)
// indices(i) is the first cell in the i'th sentence
// indices(i+1)-1 is the last cell in the i'th sentence
// the last cell has the root score.
//
/** TODO this isn't optimized at all */
__kernel void computeMBR(__global decode_t* decodeOut,
                           __global const float* inside,
                           __global const float* outside,
                           __global const int* indices,
                           __global const int* lengths,
                           const int numIndices,
                           int numSyms,
                           int root) {
  const int sentence = get_global_id(0);
  const int firstCell = indices[sentence];
  const int lastCell = indices[sentence + 1];
  int length = lengths[sentence];
  const float root_score = inside[(lastCell-1) * numSyms + root];


  for(int cell = firstCell; cell < lastCell; cell++) {
    __constant const int* projections = (cell-firstCell >= length) ? nonterminalProjections : terminalProjections;

    __global const float* in = inside + (cell * numSyms);
    __global const float* out = outside + (cell * numSyms);
      
    float coarseMargs["""+structure.numCoarseSyms+"""];
    for(int coarseSym = 0; coarseSym < """+structure.numCoarseSyms+"""; ++coarseSym) {
      coarseMargs[coarseSym] = 0.0f;
    }
    for(int sym = 0; sym < NUM_SYMS; ++sym) {
""" + {if (semiring.isInstanceOf[LogSumRuleSemiring.type]) {
    "coarseMargs[projections[sym]] += exp(in[sym] - root_score + out[sym]);\n"
  } else {
    "coarseMargs[projections[sym]] += in[sym]/root_score * out[sym];\n"
  } }+ """
    }
      
    decode_t myDecode;
    myDecode.score = -INFINITY;
    myDecode.symbol = -1;
    for(int coarseSym = 0; coarseSym < """+structure.numCoarseSyms+"""; ++coarseSym) {
      if (coarseMargs[coarseSym] > myDecode.score) {
      	myDecode.score = coarseMargs[coarseSym];
      	myDecode.symbol = coarseSym;
      }
    }

    decodeOut[cell] = myDecode;
  }

}
      """
  }
}
