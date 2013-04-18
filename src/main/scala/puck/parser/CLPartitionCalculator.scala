package puck.parser

import com.nativelibs4java.opencl._
import java.lang.{Float=>JFloat, Integer=>JInt}
import puck.parser.gen.ParserGenerator
import trochee.kernels.Global

/**
 *
 * @author dlwh
 */
trait CLPartitionCalculator[C, L] {
  val parserGen : ParserGenerator[L]
  import parserGen._

  import parserGen.grammar._
  implicit def context: CLContext


  def partitions(insideTop: CLBuffer[Float], offsets: CLBuffer[Int], lengths: CLBuffer[Int], numSentences: Int, events: CLEvent*)(implicit queue: CLQueue) = synchronized {
    val partitions = context.createFloatBuffer(CLMem.Usage.Output, numSentences * numGrammars)
    kernel.setArgs(partitions, insideTop, offsets, lengths)
    val toWait = kernel.enqueueNDRange(queue, Array(numSentences, numGrammars), Array(1, numGrammars), events:_*)

    val fPtr = partitions.read(queue, toWait)
    val floats = fPtr.getFloats
    partitions.release()
    fPtr.release()
    floats
  }

  lazy val kernel = codegen.mkKernel(kernel4("compute_partitions") { (partitions: Rep[Array[Float] with Global],
                                                                 insideTops: Rep[ParseChart with Global],
                                                                 offsets: Rep[Array[Int] with Global],
                                                                 lengths: Rep[Array[Int] with Global]) =>
    val sentence = globalId(0)
    val grammar = globalId(1)
    val length = lengths(sentence)
    val offset = offsets(sentence)
    val cell = insideTops(offset, 0, length, grammar)
    partitions(sentence * numGrammars + grammar) = (cell(root)).toLogSpace
  })


}

