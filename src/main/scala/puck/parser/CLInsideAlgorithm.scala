package puck.parser

import scala.collection.mutable.ArrayBuffer
import puck.parser.gen.ParserGenerator
import com.nativelibs4java.opencl._

trait CLInsideAlgorithm[C, L] {

  val parserGen : ParserGenerator[L]
  import parserGen._

  import parserGen.grammar._

  implicit def context: CLContext

  def insidePass(numSentences: Int,
                 inside: GPUCharts[C, L],
                 offsets: CLBuffer[Int],
                 lengths: CLBuffer[Int],
                 maxLength: Int,
                 lengthOffsets: CLBuffer[Int],
                 rules: CLBuffer[Float],
                 events: CLEvent*)(implicit queue: CLQueue) = synchronized {
    binaries.foreach(_.setArgs(inside.bot.dev, inside.top.dev, offsets, lengths, Integer.valueOf(1),  rules))
    unaries.setArgs(inside.bot.dev, inside.top.dev, offsets, lengths, Integer.valueOf(2), rules)
    leftTermBinaries.foreach(_.setArgs(inside.bot.dev, inside.top.dev, inside.tags.dev, offsets, lengths, lengthOffsets, Integer.valueOf(1), rules))
    rightTermBinaries.foreach(_.setArgs(inside.bot.dev, inside.top.dev, inside.tags.dev, offsets, lengths, lengthOffsets, Integer.valueOf(1), rules))
    bothTermBinaries.map(_.setArgs(inside.bot.dev, inside.tags.dev, offsets, lengths, lengthOffsets, rules))
    termUnaries.setArgs(inside.tags.dev, inside.top.dev, offsets, lengths, lengthOffsets, rules)
    val iu, ib, it, hooks = new ArrayBuffer[CLEvent]()

    var lastU:CLEvent = null

    lastU = termUnaries.enqueueNDRange(queue, Array(numSentences, maxLength, numGrammars), Array(1, 1, numGrammars), events:_*)
    iu += lastU

    val termBs = bothTermBinaries.map(_.enqueueNDRange(queue, Array(numSentences, maxLength, numGrammars), Array(1, 1, numGrammars), lastU))
    queue.finish()
    it ++= termBs

    for (len <- 2 to maxLength) {
       binaries.foreach(_.setArg(4, len))
       val b = binaries.map(_.enqueueNDRange(queue, Array(numSentences, maxLength + 1 - len, numGrammars), Array(1, 1, numGrammars), lastU))
       ib ++= b

       leftTermBinaries.foreach(_.setArg(6, len))
       rightTermBinaries.foreach(_.setArg(6, len))
       val t = (leftTermBinaries ++ rightTermBinaries).foldLeft(null:CLEvent){(ev, k) =>
         if(ev != null) {
          it += ev
         }
         k.enqueueNDRange(queue, Array(numSentences, maxLength + 1 - len, numGrammars), Array(1, 1, numGrammars), (ev +: b):_*)
       }


       unaries.setArg(4, len)
       lastU = unaries.enqueueNDRange(queue, Array(numSentences, maxLength + 1 - len, numGrammars), Array(1, 1, numGrammars), t)
       iu += lastU

     }


     if(queue.getProperties.contains(CLDevice.QueueProperties.ProfilingEnable)) {
       queue.finish()
       val iuCount = iu.map(e => e.getProfilingCommandEnd - e.getProfilingCommandStart).sum / 1E9
       val ibCount = ib.map(e => e.getProfilingCommandEnd - e.getProfilingCommandStart).sum / 1E9
       val itCount = it.map(e => e.getProfilingCommandEnd - e.getProfilingCommandStart).sum / 1E9
       val hc = hooks.filter(_ ne null).map(e => e.getProfilingCommandEnd - e.getProfilingCommandStart).sum / 1E9
       println("inside: " + iuCount + " " + ibCount + " " + itCount + " " + hc)
     }

     lastU

   }

  private lazy val binaries = Array.tabulate(partitionsParent.length){i =>codegen.mkKernel(insideNonterms(i, partitionsParent(i)))}
  private lazy val leftTermBinaries = insideLeftTermBinaries.map{x => codegen.mkKernel(x)}
  private lazy val rightTermBinaries = insideRightTermBinaries.map{x => codegen.mkKernel(x)}
  private lazy val bothTermBinaries = insideBothTerms.map(codegen.mkKernel _ )
  private lazy val unaries = { codegen.mkKernel(insideUnaries)}
  private lazy val termUnaries = {codegen.mkKernel(insideTermUnaries)}
}
