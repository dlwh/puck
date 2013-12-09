package puck.parser.gen

import puck.util._
import com.nativelibs4java.opencl._
import java.util.zip._
import epic.trees._
import puck.parser.{SymId, RuleSemiring, RuleStructure}
import java.io.{PrintStream, FileOutputStream}

// These kernels assume that the parent and the child named (L or R) are swapped
// in the workspace tables.
case class CLOutsideKernels(outside_L_NNKernels: CLBinaryRuleUpdater,
                            outside_R_NNKernels: CLBinaryRuleUpdater,
                            outside_L_NTKernels: CLBinaryRuleUpdater,
                            outside_R_NTKernels: CLBinaryRuleUpdater,
                            outside_L_TNKernels: CLBinaryRuleUpdater,
                            outside_R_TNKernels: CLBinaryRuleUpdater,
                            outside_L_TTKernels: CLBinaryRuleUpdater,
                            outside_R_TTKernels: CLBinaryRuleUpdater,
                            outsideNUKernels: IndexedSeq[CLKernel],
                            outsideTUKernels: IndexedSeq[CLKernel]) {

  def write(out: ZipOutputStream) {
     outside_L_NNKernels.write("outside_L_NN", out)
     outside_R_NNKernels.write("outside_R_NN", out)
     outside_L_NTKernels.write("outside_L_NT", out)
     outside_R_NTKernels.write("outside_R_NT", out)
     outside_L_TNKernels.write("outside_L_TN", out)
     outside_R_TNKernels.write("outside_R_TN", out)
     outside_L_TTKernels.write("outside_L_TT", out)
     outside_R_TTKernels.write("outside_R_TT", out)
    ZipUtil.addKernelSet(out, "outsideNU", outsideNUKernels)
    ZipUtil.addKernelSet(out, "outsideTU", outsideTUKernels)
  }
}

object CLOutsideKernels {

  def read(in: ZipFile)(implicit context: CLContext) = {
    val outside_L_NN = CLBinaryRuleUpdater.read(in, "outside_L_NN")
    val outside_R_NN = CLBinaryRuleUpdater.read(in, "outside_R_NN")
    val outside_L_NT = CLBinaryRuleUpdater.read(in, "outside_L_NT")
    val outside_R_NT = CLBinaryRuleUpdater.read(in, "outside_R_NT")
    val outside_L_TN = CLBinaryRuleUpdater.read(in, "outside_L_TN")
    val outside_R_TN = CLBinaryRuleUpdater.read(in, "outside_R_TN")
    val outside_L_TT = CLBinaryRuleUpdater.read(in, "outside_L_TT")
    val outside_R_TT = CLBinaryRuleUpdater.read(in, "outside_R_TT")
    val outsideNU = ZipUtil.readKernelSet(in, "outsideNU")
    val outsideTU = ZipUtil.readKernelSet(in, "outsideTU")
    CLOutsideKernels(outside_L_NN, outside_R_NN,
      outside_L_NT, outside_R_NT,
      outside_L_TN, outside_R_TN,
      outside_L_TT, outside_R_TT,
      outsideNU, outsideTU)
  }

  def rotateLeftToParent[C, L](r: (BinaryRule[SymId[C, L]], Int)) = {
    BinaryRule(r._1.left, r._1.parent, r._1.right) -> r._2
  }

  def rotateRightToParent[C, L](r: (BinaryRule[SymId[C, L]], Int)) = {
    BinaryRule(r._1.right, r._1.left, r._1.parent) -> r._2
  }

  def rotateChildToParent[C, L](r: (UnaryRule[SymId[C, L]], Int)) = {
    UnaryRule(r._1.child, r._1.parent, r._1.chain) -> r._2
  }

  def make[C, L](structure: RuleStructure[C, L])(implicit context: CLContext, semiring: RuleSemiring) = {
//    val parserGen = new LHSGenRuleMultiply[C, L](structure)
    val parserGen = new LHSGenRuleMultiply[C, L](structure)
    val outside_L_NNKernels = structure.partitionsLeftChild.zipWithIndex.map { case(partition, i) =>
      parserGen.binaryRuleApplication(partition.map(rotateLeftToParent), "outside_L_nn_binaries"+i)
    }

    val outside_R_NNKernels = structure.partitionsRightChild.zipWithIndex.map { case(partition, i) =>
      parserGen.binaryRuleApplication(partition.map(rotateRightToParent), "outside_R_nn_binaries"+i)
    }

    val outside_L_NTKernels = structure.partitionsRightTermRules_LeftChild.zipWithIndex.map { case (partition, i) =>
      parserGen.binaryRuleApplication(partition.map(rotateLeftToParent), "outside_L_nt_binaries"+i)
    }

    val outside_R_NTKernels = structure.partitionsRightTermRules_RightChild.zipWithIndex.map { case (partition, i) =>
      parserGen.binaryRuleApplication(partition.map(rotateRightToParent), "outside_R_nt_binaries"+i)
    }

    val outside_L_TNKernels = structure.partitionsLeftTermRules_LeftChild.zipWithIndex.map { case (partition, i) =>
      parserGen.binaryRuleApplication(partition.map(rotateLeftToParent), "outside_L_tn_binaries"+i)
    }

    val outside_R_TNKernels = structure.partitionsLeftTermRules_RightChild.zipWithIndex.map { case (partition, i) =>
      parserGen.binaryRuleApplication(partition.map(rotateRightToParent), "outside_R_tn_binaries"+i)
    }

    val outside_L_TTKernels = structure.partitionsBothTermRules_LeftChild.zipWithIndex.map { case (partition, i) =>
       parserGen.binaryRuleApplication(partition.map(rotateLeftToParent), "outside_L_tt_binaries"+i)
    }

    val outside_R_TTKernels = structure.partitionsBothTermRules_RightChild.zipWithIndex.map { case (partition, i) =>
      parserGen.binaryRuleApplication(partition.map(rotateRightToParent), "outside_R_tt_binaries"+i)
    }

    val outsideNUKernels = structure.nontermUnariesChild.zipWithIndex.map { case (partition, i) =>
      parserGen.unaryRuleApplication(partition.map(rotateChildToParent), "outside_nn_unaries"+i)
    }

    val outsideTUKernels = structure.termUnariesChild.zipWithIndex.map { case (partition, i) =>
      parserGen.unaryRuleApplication(partition.map(rotateChildToParent), "outside_nt_unaries"+i)
    }

    CLOutsideKernels(new CLBinaryRuleUpdater(outside_L_NNKernels),
      new CLBinaryRuleUpdater(outside_R_NNKernels),
      new CLBinaryRuleUpdater(outside_L_NTKernels),
      new CLBinaryRuleUpdater(outside_R_NTKernels),
      new CLBinaryRuleUpdater(outside_L_TNKernels),
      new CLBinaryRuleUpdater(outside_R_TNKernels),
      new CLBinaryRuleUpdater(outside_L_TTKernels),
      new CLBinaryRuleUpdater(outside_R_TTKernels),
      outsideNUKernels,
      outsideTUKernels)
  }
}
