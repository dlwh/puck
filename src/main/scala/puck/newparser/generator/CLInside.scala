package puck.newparser.generator

import com.nativelibs4java.opencl.CLContext
import trochee.kernels.Global

/**
 * TODO
 *
 * @author dlwh
 **/
class CLInside[C, L](ruleStructure: RuleStructure[C, L], val gen: ParserGen[L])(implicit context: CLContext) {

  val insideNNKernels = ruleStructure.partitionsParent.zipWithIndex.map { case(partition, i) =>
    gen.mkKernel(gen.IR.binaryRuleKernel(partition, "inside_nn_binaries_"+i))
  }

  val insideNTKernels = ruleStructure.partitionsRightTermRules.zipWithIndex.map { case (partition, i) =>
    gen.mkKernel(gen.IR.binaryRuleKernel(partition, "inside_nt_binaries_"+i))
  }

  val insideTNKernels = ruleStructure.partitionsLeftTermRules.zipWithIndex.map { case (partition, i) =>
    gen.mkKernel(gen.IR.binaryRuleKernel(partition, "inside_tn_binaries"+i))
  }

  val insideTTKernels = ruleStructure.partitionsBothTermRules.zipWithIndex.map { case (partition, i) =>
    gen.mkKernel(gen.IR.binaryRuleKernel(partition, "inside_tt_binaries_"+i))
  }

  val insideNUKernels = IndexedSeq(ruleStructure.unaryRules).zipWithIndex.map { case (partition, i) =>
    gen.mkKernel(gen.IR.unaryRuleKernel(partition, "inside_nn_unaries"+i))
  }

  val insideTUKernels = IndexedSeq(ruleStructure.unaryTermRules).zipWithIndex.map { case (partition, i) =>
    gen.mkKernel(gen.IR.unaryRuleKernel(partition, "inside_nt_unaries"+i))
  }


  import gen.IR._
  val sumGrammarCellsKernel = gen.mkKernel( gen.IR.kernel[Array[Real] with Global, Int, Array[Real] with Global, Int, Int, Int, Int]("sumGrammars", { (dest: Rep[Array[Real] with Global],
                                                                                                                                                       destOff: Rep[Int],
                                                                                                                                                       source: Rep[Array[Real] with Global],
                                                                                                                                                       srcOff: Rep[Int],
                                                                                                                                                       rowSize: Rep[Int],
                                                                                                                                                       numLabels: Rep[Int],
                                                                                                                                                       rowsToDo: Rep[Int]) =>
    val row = globalId(0)
    val label = globalId(1)
    if(row < rowsToDo) if(label < numLabels) {
      dest(label * rowSize + row + destOff) =  dest(label * rowSize + row + destOff) + source(label * rowSize + row + destOff)
    }

  }))
}
