package puck.newparser.generator

import puck.util._
import com.nativelibs4java.opencl._
import java.util.zip._
import scala.collection.JavaConverters._
import trochee.basic._
import trochee.kernels._
import scala.reflect.runtime.universe._

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

case class CLParserUtilKernels(sumGrammarKernel: CLKernel) {
  def write(out: ZipOutputStream) {
    ZipUtil.addKernel(out, "sumGrammarKernel", sumGrammarKernel)
  }

}

object CLParserUtilKernels {
  def read(zf: ZipFile)(implicit ctxt: CLContext) = {
    CLParserUtilKernels(ZipUtil.readKernel(zf, "sumGrammarKernel"))
  }

  def make[C, L](generator: CLParserKernelGenerator[C, L])(implicit context: CLContext) = {
    CLParserUtilKernels(generator.gen.mkKernel(generator.gen.IR.sumGrammarCellsKernel))
  }


}
