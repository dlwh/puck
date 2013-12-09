package puck.parser.gen

import epic.parser.SimpleRefinedGrammar
import com.nativelibs4java.opencl.{CLContext, CLKernel}
import java.util.zip.{ZipFile, ZipOutputStream}
import puck.util.ZipUtil
import puck.parser.{RuleSemiring, RuleStructure}


case class CLInsideKernels(insideNNKernels: CLBinaryRuleUpdater,
                           insideNTKernels: CLBinaryRuleUpdater,
                           insideTNKernels: CLBinaryRuleUpdater,
                           insideTTKernels: CLBinaryRuleUpdater,
                           insideNUKernels: IndexedSeq[CLKernel],
                           insideTUKernels: IndexedSeq[CLKernel]) {

  def write(out: ZipOutputStream) {
     insideNTKernels.write("insideNT", out)
     insideNNKernels.write("insideNN", out)
     insideTNKernels.write("insideTN", out)
     insideTTKernels.write("insideTT", out)
    ZipUtil.addKernelSet(out, "insideNU", insideNUKernels)
    ZipUtil.addKernelSet(out, "insideTU", insideTUKernels)
  }
}

object CLInsideKernels {
  def read(in: ZipFile)(implicit context: CLContext) = {
    val insideNT = CLBinaryRuleUpdater.read(in, "insideNT")
    val insideNN = CLBinaryRuleUpdater.read(in, "insideNN")
    val insideTN = CLBinaryRuleUpdater.read(in, "insideTN")
    val insideTT = CLBinaryRuleUpdater.read(in, "insideTT")
    val insideNU = ZipUtil.readKernelSet(in, "insideNU")
    val insideTU = ZipUtil.readKernelSet(in, "insideTU")
    CLInsideKernels(insideNN, insideNT, insideTN, insideTT, insideNU, insideTU)
  }


  def make[C, L](structure: RuleStructure[C, L])(implicit context: CLContext, semiring: RuleSemiring) = {
//    val parserGen = new LHSGenRuleMultiply[C, L](structure)
    val parserGen = new LHSGenRuleMultiply[C, L](structure)
    val insideNNKernels =  parserGen.binaryRuleApplication(structure.nontermRules, "inside_nn_binaries")

    val insideNTKernels =  parserGen.binaryRuleApplication(structure.rightTermRules, "inside_nt_binaries")

    val insideTNKernels =  parserGen.binaryRuleApplication(structure.leftTermRules, "inside_tn_binaries")

    val insideTTKernels =  parserGen.binaryRuleApplication(structure.bothTermRules, "inside_tt_binaries")

    val insideNUKernels = structure.nontermUnariesParent.zipWithIndex.map { case (partition, i) =>
      parserGen.unaryRuleApplication(partition, "inside_n_unaries"+i)
    }

    val insideTUKernels = structure.termUnariesParent.zipWithIndex.map { case (partition, i) =>
      parserGen.unaryRuleApplication(partition, "inside_t_unaries"+i)
    }

    CLInsideKernels(insideNNKernels,
      insideNTKernels,
      insideTNKernels,
      insideTTKernels,
      insideNUKernels,
      insideTUKernels)
  }
}
