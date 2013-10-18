package puck.parser.gen

import epic.parser.SimpleRefinedGrammar
import com.nativelibs4java.opencl.{CLContext, CLKernel}
import java.util.zip.{ZipFile, ZipOutputStream}
import puck.util.ZipUtil
import puck.parser.{RuleSemiring, RuleStructure}


case class CLInsideKernels(insideNNKernels: IndexedSeq[CLKernel],
                           insideNTKernels: IndexedSeq[CLKernel],
                           insideTNKernels: IndexedSeq[CLKernel],
                           insideTTKernels: IndexedSeq[CLKernel],
                           insideNUKernels: IndexedSeq[CLKernel],
                           insideTUKernels: IndexedSeq[CLKernel]) {

  def write(out: ZipOutputStream) {
    ZipUtil.addKernelSet(out, "insideNT", insideNTKernels)
    ZipUtil.addKernelSet(out, "insideNN", insideNNKernels)
    ZipUtil.addKernelSet(out, "insideTN", insideTNKernels)
    ZipUtil.addKernelSet(out, "insideTT", insideTTKernels)
    ZipUtil.addKernelSet(out, "insideNU", insideNUKernels)
    ZipUtil.addKernelSet(out, "insideTU", insideTUKernels)
  }
}

object CLInsideKernels {
  def read(in: ZipFile)(implicit context: CLContext) = {
    val insideNT = ZipUtil.readKernelSet(in, "insideNT")
    val insideNN = ZipUtil.readKernelSet(in, "insideNN")
    val insideTN = ZipUtil.readKernelSet(in, "insideTN")
    val insideTT = ZipUtil.readKernelSet(in, "insideTT")
    val insideNU = ZipUtil.readKernelSet(in, "insideNU")
    val insideTU = ZipUtil.readKernelSet(in, "insideTU")
    CLInsideKernels(insideNN, insideNT, insideTN, insideTT, insideNU, insideTU)
  }


  def make[C, L](structure: RuleStructure[C, L])(implicit context: CLContext, semiring: RuleSemiring) = {
    val parserGen = new GenRuleMultiply()
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

    val insideNUKernels = IndexedSeq(structure.unaryRules).zipWithIndex.map { case (partition, i) =>
      parserGen.unaryRuleApplication(partition, "inside_n_unaries"+i)
    }

    val insideTUKernels = IndexedSeq(structure.unaryTermRules).zipWithIndex.map { case (partition, i) =>
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
