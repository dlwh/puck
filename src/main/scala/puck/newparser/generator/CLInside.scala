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
  val sumGrammarCellsKernel = gen.mkKernel( gen.IR.kernel[Array[Real] with Global, Int, Int, Array[Real] with Global, Int, Int, Int, Int]("sumGrammars", { (dest: Rep[Array[Real] with Global],
                                                                                                                                                       destOff: Rep[Int],
                                                                                                                                                       destRowSize: Rep[Int],
                                                                                                                                                       source: Rep[Array[Real] with Global],
                                                                                                                                                       srcOff: Rep[Int],
                                                                                                                                                       srcRowSize: Rep[Int],
                                                                                                                                                       numLabels: Rep[Int],
                                                                                                                                                       rowsToDo: Rep[Int]) =>
    val row = globalId(0)
    val label = globalId(1)
    if(row < rowsToDo) if(label < numLabels) {
      val score = dest(label * destRowSize + row + destOff) + source(label * srcRowSize + row + srcOff)
      if(score !== zero) {
        printf("%d %d %f %f %f\n", row, label, score, dest(label * destRowSize + row + destOff), source(label * srcRowSize + row + srcOff))
      }
      dest(label * destRowSize + row + destOff) =  dest(label * destRowSize + row + destOff) + source(label * srcRowSize + row + srcOff)
    }

  }))

/*
   // TODO: before zeroing out on the last unary, copy partitions to devRight or something.
  lazy val partitionKernel = gen.mkKernel(gen.IR.kernel("compute_partitions", { (partitions: Rep[Array[Float] with Global],
                                                                 insideTops: Rep[Float with Global],
                                                                 offsets: Rep[Array[Int] with Global],
                                                                 lengths: Rep[Array[Int] with Global]) =>
    val sentence = globalId(0)
    val grammar = globalId(1)
    val length = lengths(sentence)
    val offset = offsets(sentence)
    val cell = insideTops(offset, 0, length, grammar)
    partitions(sentence * numGrammars + grammar) = (cell(root)).toLogSpace
  }))
*/
}
