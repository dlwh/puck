package puck.parser

import scala.collection.mutable.ArrayBuffer
import puck.parser.gen.ParserGenerator
import com.nativelibs4java.opencl._

trait CLOutsideAlgorithm[C, L] {

  val parserGen : ParserGenerator[L]
  import parserGen._

  import parserGen.grammar._

  implicit def context: CLContext

  def outsidePass(numSentences: Int,
                  outside: GPUCharts[C, L],
                 inside: GPUCharts[C, L],
                 offsets: CLBuffer[Int],
                 lengths: CLBuffer[Int],
                 lengthOffsets: CLBuffer[Int],
                 maxLength: Int,
                 rules: CLBuffer[Float],
                 events: CLEvent*)(implicit queue: CLQueue) = synchronized {
             val ou, ob, otb, ot  = new ArrayBuffer[CLEvent]()
    var lastU = null:CLEvent
    lbinaries.foreach(_.setArgs(outside.top.dev, outside.bot.dev, inside.top.dev, offsets, lengths, Integer.valueOf(maxLength), rules))
    rbinaries.foreach(_.setArgs(outside.top.dev, outside.bot.dev, inside.top.dev, offsets, lengths, Integer.valueOf(maxLength), rules))
    termBinaries.foreach(_.setArgs(outside.bot.dev, outside.tags.dev, outside.top.dev, inside.top.dev, inside.tags.dev, offsets, lengths, lengthOffsets, Integer.valueOf(maxLength), rules))
    bothTermBinaries.foreach(_.setArgs(outside.bot.dev, outside.tags.dev, inside.tags.dev, offsets, lengths, lengthOffsets, rules))
    unaries.setArgs(outside.top.dev, outside.bot.dev, offsets, lengths, Integer.valueOf(maxLength), rules)
    tunaries.setArgs(outside.top.dev, outside.tags.dev, offsets, lengthOffsets, lengths, rules)

    lastU = unaries.enqueueNDRange(queue, Array(numSentences, 1, numGrammars), Array(1, 1, numGrammars), events:_*)
    ou += lastU

    for (len <- (maxLength - 1) to 1 by -1) {
      lbinaries.foreach(_.setArg(5, len))
      rbinaries.foreach(_.setArg(5, len))
      val lastLB = lbinaries.map(_.enqueueNDRange(queue, Array(numSentences, maxLength + 1 - len, numGrammars), Array(1, 1, numGrammars), lastU))
      ob ++= lastLB
      val lastRB:Seq[CLEvent] = for( (rb, block) <- rbinaries zip lastLB) yield rb.enqueueNDRange(queue, Array(numSentences, maxLength + 1 - len, numGrammars), Array(1, 1, numGrammars), block)
      ob ++= lastRB
      termBinaries.foreach(_.setArg(8, len))
      lastU = termBinaries.foldLeft(lastU){ (e,k) =>
        val ee = k.enqueueNDRange(queue, Array(numSentences, maxLength + 1 - len, numGrammars), Array(1, 1, numGrammars), (if(e eq null) lastRB else Seq(e)):_*)
        if (ee ne null) otb += ee
        ee
      }
      if(len == 2) {
        val events = bothTermBinaries.map(_.enqueueNDRange(queue, Array(numSentences, maxLength, numGrammars), Array(1, 1, numGrammars), lastU))
        ot ++= events
        queue.finish()
      } else  if(len == 1) {
        lastU = tunaries.enqueueNDRange(queue, Array(numSentences, maxLength, numGrammars), Array(1, 1, numGrammars), lastU)
      }
      unaries.setArg(4, len)
      lastU = unaries.enqueueNDRange(queue, Array(numSentences, maxLength + 1 - len, numGrammars), Array(1, 1, numGrammars), lastU)

      ou += lastU
    }

    if(queue.getProperties.contains(CLDevice.QueueProperties.ProfilingEnable)) {
      queue.finish()
      val ouCount = ou.map(e => e.getProfilingCommandEnd - e.getProfilingCommandStart).sum / 1E9
      val obCount = ob.map(e => e.getProfilingCommandEnd - e.getProfilingCommandStart).sum / 1E9
      val otbCount = otb.map(e => e.getProfilingCommandEnd - e.getProfilingCommandStart).sum / 1E9
      val otCount = ot.map(e => e.getProfilingCommandEnd - e.getProfilingCommandStart).sum / 1E9
      println("outside: " + ouCount + " " + obCount + " " + otCount + " " + otbCount)
    }

    lastU

   }

  private lazy val lbinaries = Array.tabulate(partitionsLeft.length)(i => codegen.mkKernel(outsideNontermsLeft(i, partitionsLeft(i))))
  private lazy val rbinaries = Array.tabulate(partitionsRight.length)(i =>codegen.mkKernel(outsideNontermsRight(i, partitionsRight(i))))
  private lazy val leftTermBinaries = outsideLeftTermBinaries.map{x => codegen.mkKernel(x)}
  private lazy val rightTermBinaries = outsideRightTermBinaries.map{x => codegen.mkKernel(x)}
  private lazy val termBinaries = (leftTermBinaries ++ rightTermBinaries)
  private lazy val bothTermBinaries = outsideBothTerms.map(codegen.mkKernel _ )
  private lazy val unaries = { println(codegen.emitKernelSource(outsideUnaries)); codegen.mkKernel(outsideUnaries)}
  private lazy val tunaries = {codegen.mkKernel(outsideTermUnaries)}

}
