package puck.newparser.generator

import puck.util._
import com.nativelibs4java.opencl._
import java.util.zip._
import scala.collection.JavaConverters._
import trochee.basic._
import trochee.kernels._
import scala.reflect.runtime.universe._
import epic.trees._

case class CLOutsideKernels(outside_L_NNKernels: IndexedSeq[CLKernel],
                            outside_R_NNKernels: IndexedSeq[CLKernel],
                            outside_L_NTKernels: IndexedSeq[CLKernel],
                            outside_R_NTKernels: IndexedSeq[CLKernel],
                            outside_L_TNKernels: IndexedSeq[CLKernel],
                            outside_R_TNKernels: IndexedSeq[CLKernel],
                            outside_L_TTKernels: IndexedSeq[CLKernel],
                            outside_R_TTKernels: IndexedSeq[CLKernel],
                            outsideNUKernels: IndexedSeq[CLKernel],
                            outsideTUKernels: IndexedSeq[CLKernel]) {

  def write(out: ZipOutputStream) {
    ZipUtil.addKernelSet(out, "outside_L_NN", outside_L_NNKernels)
    ZipUtil.addKernelSet(out, "outside_R_NN", outside_R_NNKernels)
    ZipUtil.addKernelSet(out, "outside_L_NT", outside_L_NTKernels)
    ZipUtil.addKernelSet(out, "outside_R_NT", outside_R_NTKernels)
    ZipUtil.addKernelSet(out, "outside_L_TN", outside_L_TNKernels)
    ZipUtil.addKernelSet(out, "outside_R_TN", outside_R_TNKernels)
    ZipUtil.addKernelSet(out, "outside_L_TT", outside_L_TTKernels)
    ZipUtil.addKernelSet(out, "outside_R_TT", outside_R_TTKernels)
    ZipUtil.addKernelSet(out, "outsideNU", outsideNUKernels)
    ZipUtil.addKernelSet(out, "outsideTU", outsideTUKernels)
  }
}

object CLOutsideKernels {
  def read(in: ZipFile)(implicit context: CLContext) = {
    val outside_L_NN = ZipUtil.readKernelSet(in, "outside_L_NN")
    val outside_R_NN = ZipUtil.readKernelSet(in, "outside_R_NN")
    val outside_L_NT = ZipUtil.readKernelSet(in, "outside_L_NT")
    val outside_R_NT = ZipUtil.readKernelSet(in, "outside_R_NT")
    val outside_L_TN = ZipUtil.readKernelSet(in, "outside_L_TN")
    val outside_R_TN = ZipUtil.readKernelSet(in, "outside_R_TN")
    val outside_L_TT = ZipUtil.readKernelSet(in, "outside_L_TT")
    val outside_R_TT = ZipUtil.readKernelSet(in, "outside_R_TT")
    val outsideNU = ZipUtil.readKernelSet(in, "outsideNU")
    val outsideTU = ZipUtil.readKernelSet(in, "outsideTU")
    CLOutsideKernels(outside_L_NN, outside_R_NN,
      outside_L_NT, outside_R_NT,
      outside_L_TN, outside_R_TN,
      outside_L_TT, outside_R_TT,
      outsideNU, outsideTU)
  }

  def rotateLeftToParent(r: (BinaryRule[Int], Int)) = {
    BinaryRule(r._1.left, r._1.parent, r._1.right) -> r._2
  }

  def rotateRightToParent(r: (BinaryRule[Int], Int)) = {
    BinaryRule(r._1.right, r._1.left, r._1.parent) -> r._2
  }

  def rotateChildToParent(r: (UnaryRule[Int], Int)) = {
    UnaryRule(r._1.child, r._1.parent, r._1.chain) -> r._2
  }

  def make[C, L](parserGen: CLParserKernelGenerator[C, L])(implicit context: CLContext) = {
    import parserGen._
    val outside_L_NNKernels = structure.partitionsLeftChild.zipWithIndex.map { case(partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition.map(rotateLeftToParent _), "outside_L_nn_binaries_"+i))
    }

    val outside_R_NNKernels = structure.partitionsRightChild.zipWithIndex.map { case(partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition.map(rotateRightToParent _), "outside_R_nn_binaries_"+i))
    }

    val outside_L_NTKernels = structure.partitionsRightTermRules_LeftChild.zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition.map(rotateLeftToParent _), "outside_L_nt_binaries_"+i))
    }

    val outside_R_NTKernels = structure.partitionsRightTermRules_RightChild.zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition.map(rotateRightToParent _), "outside_R_nt_binaries_"+i))
    }

    val outside_L_TNKernels = structure.partitionsLeftTermRules_LeftChild.zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition.map(rotateLeftToParent _), "outside_L_tn_binaries"+i))
    }

    val outside_R_TNKernels = structure.partitionsLeftTermRules_RightChild.zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition.map(rotateRightToParent _), "outside_R_tn_binaries"+i))
    }

    val outside_L_TTKernels = structure.partitionsBothTermRules_LeftChild.zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition.map(rotateLeftToParent _), "outside_L_tt_binaries_"+i))
    }

    val outside_R_TTKernels = structure.partitionsBothTermRules_RightChild.zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.binaryRuleKernel(partition.map(rotateRightToParent _), "outside_R_tt_binaries_"+i))
    }

    val outsideNUKernels = IndexedSeq(structure.unaryRules).zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.unaryRuleKernel(partition.map(rotateChildToParent), "outside_nn_unaries"+i))
    }

    val outsideTUKernels = IndexedSeq(structure.unaryTermRules).zipWithIndex.map { case (partition, i) =>
      gen.mkKernel(gen.IR.unaryRuleKernel(partition.map(rotateChildToParent), "outside_nt_unaries"+i))
    }

    CLOutsideKernels(outside_L_NNKernels,
      outside_R_NNKernels,
      outside_L_NTKernels,
      outside_R_NTKernels,
      outside_L_TNKernels,
      outside_R_TNKernels,
      outside_L_TTKernels,
      outside_R_TTKernels,
      outsideNUKernels,
      outsideTUKernels)
  }
}
