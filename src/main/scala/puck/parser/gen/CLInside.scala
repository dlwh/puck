package puck.parser.gen

import epic.parser.SimpleRefinedGrammar
import com.nativelibs4java.opencl.{CLContext, CLKernel}
import java.util.zip.{ZipFile, ZipOutputStream}
import puck.util.ZipUtil
import puck.parser.{CLUnaryRuleUpdater, CLBinaryRuleUpdater, RuleSemiring, RuleStructure}


case class CLInsideKernels(insideNNKernels: CLBinaryRuleUpdater,
                           insideNTKernels: CLBinaryRuleUpdater,
                           insideTNKernels: CLBinaryRuleUpdater,
                           insideTTKernels: CLBinaryRuleUpdater,
                           insideNUKernels: CLUnaryRuleUpdater,
                           insideTUKernels: CLUnaryRuleUpdater) {

  def write(out: ZipOutputStream) {
     insideNTKernels.write("insideNT", out)
     insideNNKernels.write("insideNN", out)
     insideTNKernels.write("insideTN", out)
     insideTTKernels.write("insideTT", out)
    insideNUKernels.write("insideNU", out)
    insideTUKernels.write("insideTU", out)
  }
}

trait GenType {
  def generator[C, L](structure: RuleStructure[C, L]):GenRuleMultiply[C, L]
}

object GenType {
  object VariableLength extends GenType {
//    def generator[C, L](structure: RuleStructure[C, L]): GenRuleMultiply[C, L] = new SmartVariableGen(structure)
    def generator[C, L](structure: RuleStructure[C, L]): GenRuleMultiply[C, L] = new VariableSizeGreedyGenRuleMultiply(structure)
  }
  object Canny extends GenType {
    def generator[C, L](structure: RuleStructure[C, L]): GenRuleMultiply[C, L] = new CannySegmentationGenRuleMultiply(structure)
  }

  object Random extends GenType {
    def generator[C, L](structure: RuleStructure[C, L]): GenRuleMultiply[C, L] = new RandomSegmentationGenRuleMultiply(structure)
  }

  object CoarseParent extends GenType {
    def generator[C, L](structure: RuleStructure[C, L]): GenRuleMultiply[C, L] = new CoarseParentSymbolSegmentationGenRuleMultiply(structure)
  }


  object Greedy extends GenType {
    def generator[C, L](structure: RuleStructure[C, L]): GenRuleMultiply[C, L] = new GreedySegmentationGenRuleMultiply(structure)
  }
}

object CLInsideKernels {
  def read(in: ZipFile)(implicit context: CLContext) = {
    val insideNT = CLBinaryRuleUpdater.read(in, "insideNT")
    val insideNN = CLBinaryRuleUpdater.read(in, "insideNN")
    val insideTN = CLBinaryRuleUpdater.read(in, "insideTN")
    val insideTT = CLBinaryRuleUpdater.read(in, "insideTT")
    val insideNU = CLUnaryRuleUpdater.read(in, "insideNU")
    val insideTU = CLUnaryRuleUpdater.read(in, "insideTU")
    CLInsideKernels(insideNN, insideNT, insideTN, insideTT, insideNU, insideTU)
  }



  def make[C, L](structure: RuleStructure[C, L], genType: GenType = GenType.VariableLength)(implicit context: CLContext, semiring: RuleSemiring) = {
//    val parserGen = new LHSGenRuleMultiply[C, L](structure)
//    val parserGen = new RandomSegmentationGenRuleMultiply[C, L](structure)
//    val parserGen = new CannySegmentationGenRuleMultiply[C, L](structure)
    val parserGen = genType.generator(structure)
//    val parserGen = new CoarseParentSymbolSegmentationGenRuleMultiply[C, L](structure)
//	  val parserGen = new GreedySegmentationGenRuleMultiply[C, L](structure)
//    val parserGen = new NoninlinedRuleMultiply(structure)
    val insideNNKernels =  parserGen.binaryRuleApplication(structure.nontermRules, "inside_nn_binaries")

    val insideNTKernels =  parserGen.binaryRuleApplication(structure.rightTermRules, "inside_nt_binaries")

    val insideTNKernels =  parserGen.binaryRuleApplication(structure.leftTermRules, "inside_tn_binaries")

    val insideTTKernels =  parserGen.binaryRuleApplication(structure.bothTermRules, "inside_tt_binaries")

    val insideNUKernels =  parserGen.unaryRuleApplication(structure.unaryRules, "inside_n_unaries")

    val insideTUKernels =  parserGen.unaryRuleApplication(structure.unaryTermRules, "inside_t_unaries")

    CLInsideKernels(insideNNKernels,
      insideNTKernels,
      insideTNKernels,
      insideTTKernels,
      insideNUKernels,
      insideTUKernels)
  }
}
