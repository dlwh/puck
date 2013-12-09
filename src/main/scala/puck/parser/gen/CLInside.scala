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
    val insideNNKernels = structure.partitionsParent.zipWithIndex.map { case(partition, i) =>
      parserGen.binaryRuleApplication(partition, "inside_nn_binaries_"+i)
    }

    val insideNTKernels = structure.partitionsRightTermRules.zipWithIndex.map { case (partition, i) =>
      parserGen.binaryRuleApplication(partition, "inside_nt_binaries_"+i)
    }

    val insideTNKernels = structure.partitionsLeftTermRules.zipWithIndex.map { case (partition, i) =>
      parserGen.binaryRuleApplication(partition, "inside_tn_binaries_"+i)
    }

    val insideTTKernels = structure.partitionsBothTermRules.zipWithIndex.map { case (partition, i) =>
      parserGen.binaryRuleApplication(partition, "inside_tt_binaries_"+i)
    }

    val insideNUKernels = structure.nontermUnariesParent.zipWithIndex.map { case (partition, i) =>
      parserGen.unaryRuleApplication(partition, "inside_n_unaries"+i)
    }

    val insideTUKernels = structure.termUnariesParent.zipWithIndex.map { case (partition, i) =>
      parserGen.unaryRuleApplication(partition, "inside_t_unaries"+i)
    }

    CLInsideKernels(new CLBinaryRuleUpdater(insideNNKernels),
      new CLBinaryRuleUpdater(insideNTKernels),
      new CLBinaryRuleUpdater(insideTNKernels),
      new CLBinaryRuleUpdater(insideTTKernels),
      insideNUKernels,
      insideTUKernels)
  }
}
