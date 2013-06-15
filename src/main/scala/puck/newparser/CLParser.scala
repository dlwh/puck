package puck.newparser

import breeze.config._
import com.nativelibs4java.opencl._
import com.typesafe.scalalogging.log4j._
import epic.lexicon.{TagScorer, Lexicon}
import epic.parser._
import epic.parser.projections.GrammarRefinements
import epic.trees._
import epic.trees.annotations._
import java.io._
import java.nio.FloatBuffer
import java.{lang=>jl}
import puck.linalg.NativeMatrix
import puck.parser.gen.SemiringFloatOpsExp
import puck.util.{BitHacks, ZeroMemoryKernel}
import scala.virtualization.lms.common.ArrayOpsExp
import trochee.kernels.KernelOpsExp
import puck.newparser.generator._
import collection.mutable.ArrayBuffer

/**
 * TODO
 *
 * @author dlwh
 **/
class CLParser[C, L, W](grammar: SimpleRefinedGrammar[C, L, W],
                        maxAllocSize: Long = 1<<30,
                        profile: Boolean = true)(implicit val context: CLContext) extends Logging {

  /*
  def parse(sentences: IndexedSeq[IndexedSeq[W]]):IndexedSeq[BinarizedTree[C]] = synchronized {
    {for {
      partition <- getBatches(sentences, masks).iterator
      batch = createBatch(partition)
      //    _ = getMarginals(batch)
      t <- doParse(batch)
    } yield {
      t
    }}.toIndexedSeq
  }
  */




  val structure = RuleStructure[C, L](grammar.refinements, grammar.refinedGrammar)
  private implicit val queue = if(profile) context.createDefaultProfilingQueue() else context.createDefaultOutOfOrderQueueIfPossible()

  val nrules = grammar.index.size
  // TODO: reinstate this difference if numTerms is really big.
  val cellSize =  structure.numNonTerms max structure.numTerms

  val ruleScores = Array.tabulate(grammar.refinedGrammar.index.size){r =>
    val score = grammar.ruleScoreArray(grammar.refinements.rules.project(r))(grammar.refinements.rules.localize(r))
    gen.IR.fromLogSpace(score.toFloat)
  }


  val (numGPUCells:Int, numGPUAccCells: Int) = {
    val sizeOfFloat = 4
    val fractionOfMemoryToUse = 0.8 // slack!
    val amountOfMemory = ((context.getMaxMemAllocSize min maxAllocSize) * fractionOfMemoryToUse).toInt - ruleScores.length * 4
    val maxPossibleNumberOfCells = (amountOfMemory/sizeOfFloat) / cellSize
    // We also want numGPUCells and numGPUAccCells to be divisible by 16, so that we get aligned strided access:
    //       On devices of compute capability 1.0 or 1.1, the k-th thread in a half warp must access the
    //       k-th word in a segment aligned to 16 times the size of the elements being accessed; however,
    //       not all threads need to participate... If sequential threads in a half warp access memory that is
    //       sequential but not aligned with the segments, then a separate transaction results for each element
    //       requested on a device with compute capability 1.1 or lower.
    val numberOfUnitsOf16 = maxPossibleNumberOfCells / 16
    // average sentence length of sentence, let's say n.
    // for the "acc" part of the charts, we'll only need totallength cells, so n per sentence.
    // for the "P/L/R" parts, the maximum number of relaxations (P = L * R * rules) for a fixed span
    // in a fixed sentence is (n/2)^2= n^2/4. this is (obviously) n/4 times bigger than the "acc"
    // array, if we wanted to ensure that we don't have to do anything fancy.
    // Take n = 32, then we want our P/L/R arrays to be 8 times longer.
    // 8 * 3 + 1 = 25
    val baseAccSize = numberOfUnitsOf16 / 25
    val plrSize = baseAccSize * 8
    // TODO, can probably do a better job of these calculations?
    (plrSize * 16, baseAccSize * 16)
  }





  val gen = new CLParserKernelGenerator[C, L](structure)
  import gen.insideGen

  // On the Device side we have 4 Matrices:
  // One is where we calculate P = L * R * rules, for fixed spans and split points (the "bot")
  // One is the L part of the above, and also doubles as the sink for U = P * unaries (the "top")
  // Another is the R part.
  // finally, we have the smaller "accumulator" array, where we store results from the P = L * R * rules
  // and do rescaling and stuff
  private val devParentBot, devLeft, devRight = context.createFloatBuffer(CLMem.Usage.InputOutput, numGPUCells * cellSize)
  private val devParentTop = devLeft // reuse array
  private val devParentAcc = context.createFloatBuffer(CLMem.Usage.InputOutput, numGPUAccCells * cellSize)

  // also the rules

  private val ruleDev = context.createFloatBuffer(CLMem.Usage.Input, FloatBuffer.wrap(ruleScores), false)

  // Corresponding Native Buffers on host side.
  private val hostParentBot, hostLeft, hostRight = new NativeMatrix[Float](numGPUCells, cellSize)
  private val hostParentTop = hostLeft // reuse array


  def _zero: Float = gen.IR._zero

  // other stuff
  private val zmk = new ZeroMemoryKernel(_zero)

  private case class Batch(lengthTotals: Array[Int],
                           sentences: IndexedSeq[IndexedSeq[W]],
                           charts: IndexedSeq[ParseChart]) {
    def totalLength = lengthTotals.last
    def numSentences = sentences.length

    val maxLength = sentences.map(_.length).max

    def offsetForSpanLength(sent: Int, spanLength: Int) = {
      ??? haveo 0 out spans that are longer
      lengthTotals(sent) - (spanLength-1) * sent
    }

    def totalLengthForSpan(spanLength:Int) = lengthTotals.last - charts.length * (spanLength-1)

    def offsetsForSpan(sent: Int, spanLength: Int) = Range(offsetForSpanLength(sent, spanLength), offsetForSpanLength(sent+1,spanLength))
  }

  private def getBatches(sentences: IndexedSeq[IndexedSeq[W]]): IndexedSeq[IndexedSeq[IndexedSeq[W]]] = {
    val result = ArrayBuffer[IndexedSeq[IndexedSeq[W]]]()
    var current = ArrayBuffer[IndexedSeq[W]]()
    var currentLengthTotal = 0
    for( (s, i) <- sentences.zipWithIndex) {
      currentLengthTotal += s.length
      if(currentLengthTotal > numGPUAccCells) {
        assert(current.nonEmpty)
        result += current
        currentLengthTotal = s.length
        current = ArrayBuffer()
      }
      current += s
    }

    if(current.nonEmpty) result += current
    result
  }

  private def createBatch(sentences: IndexedSeq[IndexedSeq[W]]): Batch = {
    val lengthTotals = sentences.scanLeft(0)((acc, sent) => acc + sent.length)
    val posTags = for( (s, i) <- sentences.zipWithIndex) yield {
      val anch  = grammar.tagScorer.anchor(s)
      val lexAnch = grammar.lexicon.anchor(s)
      val chart = new ParseChart(s.length, cellSize, cellSize, gen.IR._zero)
      for {
        pos <- (0 until s.length)
        a <- lexAnch.allowedTags(pos)
        refA <- grammar.refinements.labels.refinementsOf(a)
      } {
        val global = grammar.refinements.labels.globalize(a, refA)
        val score = anch.scoreTag(pos, grammar.refinedGrammar.labelIndex.get(global))
        chart.terms.array(pos, structure.labelIndexToTerminal(global)) = gen.IR.fromLogSpace(score.toFloat)
        assert(!score.isNaN)
      }
      chart
    }

    Batch(lengthTotals.toArray, sentences, posTags)
  }


  def inside(batch: Batch) = synchronized {
    var eZpb = zmk.zeroMemory(devParentBot)
    var eZpt = zmk.zeroMemory(devParentTop)
    var eZpa = zmk.zeroMemory(devParentAcc)
    hostParentTop := _zero
    hostParentBot := _zero
    val (tuEndEvents, offsets) = doTUnaryUpdates(batch, eZpt)
    copyBackToHost(devParentTop, hostParentTop, batch, _.top, 1, offsets, tuEndEvents:_*)
    val ttdone = doTTUpdates(batch, maxLength, eZpa +: eZpb +: tuEndEvents:_*)
    for(span <- 2 until maxLength) {
      val nt = doNTUpdates(batch, span, ttdone._1)
      doTNUpdates(batch, span, nt)
      val (doneNN,offsets) = doNNUpdates(batch, span, tuEndEvents:_*)
      copyBackToHost(devParentAcc, hostParentBot, batch, _.bot, span, offsets, doneNN:_*)
      val doneU = doUnaryUpdates(batch, span)
      copyBackToHost(devParentTop, hostParentTop, batch, _.top, span, offsets, doneU:_*)

      eZpb = zmk.zeroMemory(devParentBot)
      eZpt = zmk.zeroMemory(devParentTop)
      eZpa = zmk.zeroMemory(devParentAcc)
    }
  }

  def doTUnaryUpdates(batch: Batch, events: CLEvent*) = {
    import batch._
    for(sent <- 0 until charts.length) {
      val lslice = charts(sent).terms
      hostParentBot(offsetsForSpan(sent, 1), ::) := lslice.array
    }
    val wL = devParentBot.write(queue:CLQueue, 0, totalLengthForSpan(1) * cellSize, hostParentBot.pointer.as(jl.Float.TYPE), false, events:_*)
    val endEvents = insideGen.insideTUKernels.map{(kernel) =>
      kernel.setArgs(devParentTop, devParentBot, ruleDev, Integer.valueOf(numGPUCells), Integer.valueOf(numGPUCells), Integer.valueOf(offsets.last))
      kernel.enqueueNDRange(queue, Array(offsets.length), wL)
    }
     endEvents -> offsets
  }

  private def doBinaryRules(kernels: IndexedSeq[CLKernel], leftCellSize: Int, rightCellSize: Int, numToDo: Int, distinctPositions: Int, events: CLEvent*) = {
    val wL = devLeft.write(queue:CLQueue, 0, numToDo * leftCellSize, hostLeft.pointer.as(jl.Float.TYPE), false, events:_*)
    val wR = devRight.write(queue:CLQueue, 0, numToDo * rightCellSize, hostRight.pointer.as(jl.Float.TYPE), false, events:_*)
    val writeEvents = IndexedSeq(wL, wR)


    val multiplies = kernels.map{ kernel =>
      kernel.setArgs(devParentBot, devLeft, devRight, Integer.valueOf(0), ruleDev, Integer.valueOf(numGPUCells), Integer.valueOf(numToDo))
      kernel.enqueueNDRange(queue, Array(numToDo), writeEvents:_*)
    }
    assert(distinctPositions <= numGPUAccCells)
    val doneSum = {
      if(numToDo > numGPUAccCells)
        sumSplitBlocks(devParentBot, numToDo, distinctPositions, multiplies:_*)
      else
        multiplies
    }
    sumGrammarCells(devParentAcc, 0, devParentBot, 0, distinctPositions, doneSum:_*)
  }

  def doTTUpdates(charts: IndexedSeq[ParseChart], span: Int, events: CLEvent*) = {
    // layout:
    // Parents for each span of length 2 == n -1
    // [Sent0Span2Pos0, Sent0Span2Pos1, ..., Sent0Span2PosN-2, Sent1Span2Pos0, ...]
    // [Sent0Term0    , Sent0Term1    , ..., Sent0TermN-1    , Sent1Term0    , ...]
    // [Sent0Term1    , Sent0Term2    , ..., Sent0TermN      , Sent1Term1    , ...]
    val offsets = charts.scanLeft(0)((off, chart) => off + chart.length-1)
    for(sent <- 0 until charts.length) {
      val lslice = charts(sent).terms.rowSlice(0, charts(sent).length-1)
      val rslice = charts(sent).terms.rowSlice(1, charts(sent).length)
      hostLeft(offsets(sent) until offsets(sent+1), ::) := lslice
      hostRight(offsets(sent) until offsets(sent+1), ::) := rslice
    }

    val ret = doBinaryRules(insideGen.insideTTKernels, numGPUCells, numGPUCells, offsets.last, offsets.last, events:_*)
    ret -> offsets
  }


  def doTNUpdates(charts: IndexedSeq[ParseChart], span: Int, events: CLEvent*) = {
    val offsets = charts.scanLeft(0)((off, chart) => off + chart.length-span)
    val leftChildLength = 1
    for(sent <- 0 until charts.length) {
      val lslice = charts(sent).terms.rowSlice(0, charts(sent).length-span+1)
      val rslice = charts(sent).top.spanSlice(span-leftChildLength, leftChildLength, charts(sent).length)
      assert(lslice.rows == rslice.rows)
      assert(offsets(sent+1) - offsets(sent) == lslice.rows)
      hostLeft(offsets(sent) until offsets(sent+1), ::) := lslice
      hostRight(offsets(sent) until offsets(sent+1), ::) := rslice
    }
    doBinaryRules(insideGen.insideTNKernels, numGPUCells, numGPUCells, offsets.last, offsets.last, events:_*)
  }

  def doNTUpdates(charts: IndexedSeq[ParseChart], span: Int, events: CLEvent*) = {
    val offsets = charts.scanLeft(0)((off, chart) => off + chart.length-span)
    val leftChildLength = span-1
    for(sent <- 0 until charts.length) {
      val lslice = charts(sent).top.spanSlice(leftChildLength, 0, charts(sent).length - leftChildLength)
      val rslice = charts(sent).terms.rowSlice(leftChildLength, charts(sent).length)
      assert(lslice.rows == rslice.rows)
      assert(offsets(sent+1) - offsets(sent) == lslice.rows)
      hostLeft(offsets(sent) until offsets(sent+1), ::) := lslice
      hostRight(offsets(sent) until offsets(sent+1), ::) := rslice
    }
    doBinaryRules(insideGen.insideNTKernels, numGPUCells, numGPUCells, offsets.last, offsets.last, events:_*)
  }

  def doUnaryUpdates(charts: IndexedSeq[ParseChart], span: Int, events: CLEvent*): IndexedSeq[CLEvent] = {
    val offsets = charts.scanLeft(0)((off, chart) => off + chart.length-span)
    insideGen.insideNUKernels.map{(kernel) =>
      kernel.setArgs(devParentTop, devParentAcc, ruleDev, Integer.valueOf(numGPUCells), Integer.valueOf(numGPUCells), Integer.valueOf(offsets.last))
      kernel.enqueueNDRange(queue, Array(offsets.length), events:_*)
    }
  }

  def doNNUpdates(charts: IndexedSeq[ParseChart], span: Int, maxLength: Int, events: CLEvent*): (IndexedSeq[CLEvent], IndexedSeq[Int]) = {
    var offset = 0 // number of cells used so far.
    var blockSize = 0
    val offsets = charts.scanLeft(0)((off, chart) => off + chart.length-span)
    for(leftChildLength <- 1 until (maxLength - span)) {
      for(sent <- 0 until charts.length) {
        val lslice = charts(sent).top.spanSlice(leftChildLength, 0, charts(sent).length-span+1)
        val rslice = charts(sent).top.spanSlice(span-leftChildLength, leftChildLength)
        assert(lslice.rows == rslice.rows)
        hostLeft(offset until (offset + lslice.rows), ::) := lslice
        hostRight(offset until (offset + lslice.rows), ::) := charts(sent).top.spanSlice(span-leftChildLength, leftChildLength)
        offset += lslice.rows
      }
    }

    val doneNN = doBinaryRules(insideGen.insideNNKernels, numGPUCells, numGPUCells, offset, offsets.last, events:_*)

    IndexedSeq(doneNN) -> offsets
  }

  private def sumSplitBlocks(devParentBot: CLBuffer[jl.Float], targetSize: Int, currentSize: Int, events: CLEvent*):IndexedSeq[CLEvent] = {
    // TODO handle the remainder
    assert(currentSize % targetSize == 0)
    val multiple: Int = currentSize / targetSize
    val log2 = BitHacks.log2(multiple)
    if(log2 == 0) return events.toIndexedSeq
    val size2 = 1 << log2
    var ev:IndexedSeq[CLEvent] = events.toIndexedSeq
    if(log2 != multiple) {
      val difference = multiple - log2
      ev = IndexedSeq(sumGrammarCells(devParentBot, 0, devParentBot, currentSize - difference*targetSize, difference * targetSize, ev:_*))
    }
    var size = size2
    while (size > targetSize) {
      size >>= 1
      ev = IndexedSeq(sumGrammarCells(devParentBot, 0, devParentBot, size, size, ev:_*))
    }
    assert(size == targetSize)
    ev
  }

  def sumGrammarCells(dest: CLBuffer[jl.Float], destOff: Int, src: CLBuffer[jl.Float], srcOff: Int, length: Int, events: CLEvent*) = {
    insideGen.sumGrammarCellsKernel.setArgs(dest, Integer.valueOf(destOff), src, Integer.valueOf(srcOff), Integer.valueOf(length))
    insideGen.sumGrammarCellsKernel.enqueueNDRange(queue, Array(length, cellSize), Array(32, 1, 1), events:_*)
  }

  private def copyBackToHost(devParent: CLBuffer[jl.Float],
                     hostParent: NativeMatrix[Float],
                     batch: Batch,
                     level: ParseChart=>ChartHalf,
                     span: Int, events: CLEvent*) = {
    devParent.read(queue, hostParent.pointer.as(jl.Float.TYPE), true, events:_*)
    for(sent <- (0 until batch.numSentences).par) {
      val lslice = level(batch.charts(sent)).spanSlice(span)
      lslice := hostParent(batch.offsetsForSpan(sent,span) until batch.offsetsForSpan(sent, span), ::)
    }
  }
}

object CLParser extends Logging {

  case class Params(annotator: TreeAnnotator[AnnotatedLabel, String, AnnotatedLabel] = FilterAnnotations(),
                    useGPU: Boolean = true, numToParse: Int = 1000)

  def main(args: Array[String]) = {
    import ParserParams.JointParams

    val params = CommandLineParser.readIn[JointParams[Params]](args)
    import params.trainer._
    println("Training Parser...")
    println(params)
    val transformed = params.treebank.trainTrees.par.map { ti => annotator(ti) }.seq.toIndexedSeq
    val grammar: SimpleRefinedGrammar[AnnotatedLabel, AnnotatedLabel, String] = GenerativeParser.extractGrammar(AnnotatedLabel.TOP, transformed)


    implicit val context = if(useGPU) {
      JavaCL.createBestContext()
    } else {
      val cpuPlatform:CLPlatform = JavaCL.listPlatforms().filter(_.listCPUDevices(true).nonEmpty).head
      cpuPlatform.createContext(new java.util.HashMap(), cpuPlatform.listCPUDevices(true):_*)
    }
    println(context)

    val kern = fromSimpleGrammar[AnnotatedLabel, AnnotatedLabel, String](grammar)
    val train = transformed.slice(0,numToParse)

  }

  def fromSimpleGrammar[L, L2, W](grammar: SimpleRefinedGrammar[L, L2, W])(implicit context: CLContext) = {
    val kern = new CLParser(grammar)
    kern
  }
}
