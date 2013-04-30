package puck.parser

import scala.collection.mutable.ArrayBuffer
import puck.parser.gen.ParserGenerator
import com.nativelibs4java.opencl._
import puck.util.CollectionImplicits._

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

    // updates bot(len)
    lastU = unaries.enqueueNDRange(queue, Array(numSentences, 1, numGrammars), Array(1, 1, numGrammars), events:_*)
    ou += lastU

    for (len <- (maxLength - 1) to 1 by -1) {
      lbinaries.foreach(_.setArg(5, len))
      rbinaries.foreach(_.setArg(5, len))
      // updates top(len) depending on bot(>len)
      val lastLB = (lbinaries ++ rbinaries).scanLeft(lastU)( (e,k) => k.enqueueNDRange(queue, Array(numSentences, maxLength + 1 - len, numGrammars), Array(1, 1, numGrammars), e)).drop(1)
      ob ++= lastLB
      lastU = lastLB.lastOption.getOrElse(lastU)

      // updates top(len) and pos depending on bot(len+1)
      termBinaries.foreach(_.setArg(8, len))
      val tbs = termBinaries.scanLeft(lastU){ (e,k) =>
         k.enqueueNDRange(queue, Array(numSentences, maxLength + 1 - len, numGrammars), Array(1, 1, numGrammars), e)
      }
      otb ++= tbs.drop(1)
      lastU = tbs.lastOption.getOrElse(lastU)


      // updates bot(len) depending on top(len)
      unaries.setArg(4, len)
      lastU = unaries.enqueueNDRange(queue, Array(numSentences, maxLength + 1 - len, numGrammars), Array(1, 1, numGrammars), lastU)
      ou += lastU


      if(len == 2) {
        // updates pos depending on bot(2)
        val ots = bothTermBinaries.scanLeft(lastU){ (e,k) =>
          k.enqueueNDRange(queue, Array(numSentences, maxLength, numGrammars), Array(1, 1, numGrammars), e)
        }
        ot ++= ots.drop(1)
        lastU = ot.lastOption.getOrElse(lastU)
      } else  if(len == 1) {
        // updates pos depending on bot(1)
        lastU = tunaries.enqueueNDRange(queue, Array(numSentences, maxLength, numGrammars), Array(1, 1, numGrammars), lastU)
      }

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
