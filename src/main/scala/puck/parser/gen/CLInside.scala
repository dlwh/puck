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

  def write(prefix: String, out: ZipOutputStream) {
    insideNTKernels.write(s"$prefix/insideNT", out)
    insideNNKernels.write(s"$prefix/insideNN", out)
    insideTNKernels.write(s"$prefix/insideTN", out)
    insideTTKernels.write(s"$prefix/insideTT", out)
    insideNUKernels.write(s"$prefix/insideNU", out)
    insideTUKernels.write(s"$prefix/insideTU", out)
  }
}

trait GenType {
  def generator[C, L](structure: RuleStructure[C, L], directWrite: Boolean, semiring: RuleSemiring):GenRuleMultiply[C, L]
}

object GenType {
  object VariableLength extends GenType {
//    def generator[C, L](structure: RuleStructure[C, L]): GenRuleMultiply[C, L] = new SmartVariableGen(structure)
    def generator[C, L](structure: RuleStructure[C, L], directWrite: Boolean, semiring: RuleSemiring): GenRuleMultiply[C, L] = new VariableSizeGreedyGenRuleMultiply(structure, directWrite, semiring)
  }
  object Canny extends GenType {
    def generator[C, L](structure: RuleStructure[C, L], directWrite: Boolean, semiring: RuleSemiring): GenRuleMultiply[C, L] = new CannySegmentationGenRuleMultiply(structure, directWrite, semiring)
  }

  object Random extends GenType {
    def generator[C, L](structure: RuleStructure[C, L], directWrite: Boolean, semiring: RuleSemiring): GenRuleMultiply[C, L] = new RandomSegmentationGenRuleMultiply(structure, directWrite, semiring)
  }

  object CoarseParent extends GenType {
    def generator[C, L](structure: RuleStructure[C, L], directWrite: Boolean, semiring: RuleSemiring): GenRuleMultiply[C, L] = new CoarseParentSymbolSegmentationGenRuleMultiply(structure, directWrite, semiring)
  }
  
  object VariableLengthCoarseParent extends GenType {
    def generator[C, L](structure: RuleStructure[C, L], directWrite: Boolean, semiring: RuleSemiring): GenRuleMultiply[C, L] = new VariableSizeCoarseParentSymbolSegmentationGenRuleMultiply(structure, directWrite, semiring)
  }

  object Greedy extends GenType {
    def generator[C, L](structure: RuleStructure[C, L], directWrite: Boolean, semiring: RuleSemiring): GenRuleMultiply[C, L] = new GreedySegmentationGenRuleMultiply(structure, directWrite, semiring)
  }
}

object CLInsideKernels {
  def read(prefix: String, in: ZipFile)(implicit context: CLContext) = {
    val insideNT = CLBinaryRuleUpdater.read(in, s"$prefix/insideNT")
    val insideNN = CLBinaryRuleUpdater.read(in, s"$prefix/insideNN")
    val insideTN = CLBinaryRuleUpdater.read(in, s"$prefix/insideTN")
    val insideTT = CLBinaryRuleUpdater.read(in, s"$prefix/insideTT")
    val insideNU = CLUnaryRuleUpdater.read(in, s"$prefix/insideNU")
    val insideTU = CLUnaryRuleUpdater.read(in, s"$prefix/insideTU")
    CLInsideKernels(insideNN, insideNT, insideTN, insideTT, insideNU, insideTU)
  }



  def make[C, L](structure: RuleStructure[C, L], directWrite: Boolean, semiring: RuleSemiring, genType: GenType = GenType.VariableLength)(implicit context: CLContext) = {
//    val parserGen = new LHSGenRuleMultiply[C, L](structure)
//    val parserGen = new RandomSegmentationGenRuleMultiply[C, L](structure)
//    val parserGen = new CannySegmentationGenRuleMultiply[C, L](structure)
    val parserGen = genType.generator(structure, directWrite, semiring)
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
