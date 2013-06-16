package puck
package newparser

import breeze.config._
import breeze.collection.mutable.TriangularArray
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
import puck.linalg.CLMatrix
import puck.parser.gen.SemiringFloatOpsExp
import puck.util.{BitHacks, ZeroMemoryKernel}
import scala.virtualization.lms.common.ArrayOpsExp
import trochee.kernels.KernelOpsExp
import puck.newparser.generator._
import collection.mutable.ArrayBuffer
import java.util

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
      batch <- getBatches(sentences, masks).iterator
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
  val cellSize = structure.numNonTerms max structure.numTerms

  val ruleScores = Array.tabulate(grammar.refinedGrammar.index.size){r =>
    val score = grammar.ruleScoreArray(grammar.refinements.rules.project(r))(grammar.refinements.rules.localize(r))
    gen.IR.fromLogSpace(score.toFloat)
  }

  val (numGPUCells:Int, numGPUChartCells: Int) = {
    val sizeOfFloat = 4
    val fractionOfMemoryToUse = 0.8 // slack!
    val amountOfMemory = ((context.getMaxMemAllocSize min maxAllocSize) * fractionOfMemoryToUse).toInt - ruleScores.length * 4
    val maxPossibleNumberOfCells = (amountOfMemory/sizeOfFloat) / cellSize
    // We want numGPUCells and numGPUAccCells to be divisible by 16, so that we get aligned strided access:
    //       On devices of compute capability 1.0 or 1.1, the k-th thread in a half warp must access the
    //       k-th word in a segment aligned to 16 times the size of the elements being accessed; however,
    //       not all threads need to participate... If sequential threads in a half warp access memory that is
    //       sequential but not aligned with the segments, then a separate transaction results for each element
    //       requested on a device with compute capability 1.1 or lower.
    val numberOfUnitsOf16 = maxPossibleNumberOfCells / 16
    // average sentence length of sentence, let's say n.
    // for the gpu charts, we'll need (n choose 2) * 2 = n^2 - n cells
    // for the "P/L/R" parts, the maximum number of relaxations (P = L * R * rules) for a fixed span
    // in a fixed sentence is (n/2)^2= n^2/4.
    // Take n = 32, then we want our P/L/R arrays to be of the ratio (3 * 256):992 \approx 3/4 (3/4 exaclty if we exclude n)
    //
    val baseSize = numberOfUnitsOf16 / 7
    val extra = numberOfUnitsOf16 % 7
    val plrSize = baseSize
    // TODO, can probably do a better job of these calculations?
    (plrSize * 16, (baseSize * 4 + extra) * 16)
  }

  val gen = new CLParserKernelGenerator[C, L](structure)
  import gen.insideGen

  // On the Device side we have 4 Matrices:
  // One is where we calculate P = L * R * rules, for fixed spans and split points (the "bot")
  // One is the L part of the above
  // Another is the R part.
  // finally, we have the array of parse charts, which is insanely large. It's also where
  // we do rescaling, etc.
  private val devParent, devLeft, devRight = new CLMatrix[Float](numGPUCells, cellSize)
  private val devCharts = context.createFloatBuffer(CLMem.Usage.InputOutput, numGPUChartCells * cellSize)

  // also the rules
  private val ruleDev = context.createFloatBuffer(CLMem.Usage.Input, FloatBuffer.wrap(ruleScores), false)

  def _zero: Float = gen.IR._zero

  // other stuff
  private val zmk = new ZeroMemoryKernel()

  private case class Batch(lengthTotals: Array[Int],
                           cellTotals: Array[Int],
                           sentences: IndexedSeq[IndexedSeq[W]]) {
    def totalLength = lengthTotals.last
    def numSentences = sentences.length
    val maxLength = sentences.map(_.length).max


    val _workArrayOffsetsForSpan = Array.tabulate(maxLength)(span => sentences.scanLeft(0)((off, sent) => off + math.max(0,sent.length-span+1)))
    def workArrayOffsetsForSpan(sent: Int, span: Int) = Range(_workArrayOffsetsForSpan(span)(sent), _workArrayOffsetsForSpan(span)(sent)) 

    def totalLengthForSpan(span: Int) = _workArrayOffsetsForSpan(span).last

    lazy val gpuCharts = for(i <- 0 until numSentences) yield {
      val numCells = (cellTotals(i+1)-cellTotals(i))/2
      val chart = new ParseChart(sentences(i).length, new CLMatrix(numCells, cellSize, devCharts, cellTotals(i)), new CLMatrix(numCells, cellSize, devCharts, cellTotals(i) + numCells))
      chart.bot.spanSlice(1).write(tagScoresFor(sentences(i)), blocking=true)
      chart
    }

    def tagScoresFor(sent: IndexedSeq[W]) = {
      val anch = grammar.tagScorer.anchor(sent)
      val lexAnch = grammar.lexicon.anchor(sent)
      val tags = new Array[Float](sent.length * cellSize)
      util.Arrays.fill(tags, _zero)
      for(pos <- 0 until sent.length; t <- lexAnch.validTags(pos); ref <- grammar.refinements.labels.refinementsOf(t)) {
        val index = grammar.refinements.labels.globalize(t, ref)
        val score = lexAnch.score(pos, grammar.refinedGrammar.labelIndex.get(index))
        val gpuIndex = structure.labelIndexToTerminal(index)
        tags(gpuIndex * sent.length + pos) = gen.IR.fromLogSpace(score)
      }
      tags

    }
  }

  private def getBatches(sentences: IndexedSeq[IndexedSeq[W]]): IndexedSeq[Batch] = {
    val result = ArrayBuffer[Batch]()
    var current = ArrayBuffer[IndexedSeq[W]]()
    var currentLengthTotal = 0
    var currentCellTotal = 0
    for( (s, i) <- sentences.zipWithIndex) {
      currentLengthTotal += s.length
      currentCellTotal += TriangularArray.arraySize(s.length+1) * 2
      if(currentLengthTotal > numGPUCells || currentCellTotal > numGPUChartCells) {
        assert(current.nonEmpty)
        result += createBatch(current)
        currentLengthTotal = s.length
        currentCellTotal = TriangularArray.arraySize(s.length+1) * 2
        current = ArrayBuffer()
      }
      current += s
    }

    if(current.nonEmpty) result += createBatch(current)
    result
  }

  private def createBatch(sentences: IndexedSeq[IndexedSeq[W]]): Batch = {
    val lengthTotals = sentences.scanLeft(0)((acc, sent) => acc + sent.length)
    val cellTotals = sentences.scanLeft(0)((acc, sent) => acc + TriangularArray.arraySize(sent.length+1))
    Batch(lengthTotals.toArray, cellTotals.toArray, sentences)
  }


  private def inside(batch: Batch) = synchronized {
    var eZp = zmk.fillMemory(devParent.data, _zero)

    val doneTUnary = doUnaryUpdates(batch, 1, eZp)
    eZp = zmk.fillMemory(devParent.data, _zero, doneTUnary: _*)

    val ttdone = doTTUpdates(batch, eZp +: doneTUnary:_*)
    var events = ttdone:Seq[CLEvent]

    for(span <- 2 until batch.maxLength) {
      // TODO: there's got to be a better way. implicits?
      events = Seq[Seq[CLEvent]=>Seq[CLEvent]](
        doNTUpdates(batch, span, _ :_*),
        doTNUpdates(batch, span, _ :_*),
        doNNUpdates(batch, span, batch.maxLength, _ :_*),
        sumBackToCharts(batch, _.bot, span, _ :_*),
        {(x: Seq[CLEvent]) => Seq(zmk.fillMemory(devParent.data, _zero, x:_*))},
        doUnaryUpdates(batch, span, _ : _*),
        sumBackToCharts(batch, _.top, span, _ :_*),
        {(x: Seq[CLEvent]) => Seq(zmk.fillMemory(devParent.data, _zero, x:_*))}
      ).foldLeft(events)((a,b) => b apply a)
    }
    events
  }

  private def doBinaryRules(batch: Batch,
    kernels: IndexedSeq[CLKernel],
    span: Int,
    splitRange: Range,
    leftChart: ParseChart=>ChartHalf,
    rightChart: ParseChart=>ChartHalf,
    events: CLEvent*) = {
    var ev = events

    var offset = 0 // number of cells used so far.

    val usedPerSplit = batch.totalLengthForSpan(span)
    var maxOffset = 0 // maximum number of cells used in one kernel execution.
                      // will be a multiple of usedPerSplit

    for(leftChildLength <- splitRange) {
      // if we fill up the buffer, run a pass.
      if(offset + usedPerSplit >= numGPUCells)  {
        assert(offset != 0)
        ev = kernels.map{ kernel =>
          kernel.setArgs(devParent, devLeft, devRight, ruleDev, Integer.valueOf(numGPUCells), Integer.valueOf(offset))
          kernel.enqueueNDRange(queue, Array(offset), ev:_*)
        } 
        maxOffset = maxOffset max offset
        offset = 0
      }
      // add next split point
      val evWrite = new ArrayBuffer[CLEvent]()
      evWrite.sizeHint(batch.numSentences * 2)
      for(sent <- 0 until batch.numSentences) {
        val lslice = leftChart(batch.gpuCharts(sent)).spanSlice(leftChildLength, 0, batch.gpuCharts(sent).length-span+1)
        val rslice = rightChart(batch.gpuCharts(sent)).spanSlice(span-leftChildLength, leftChildLength)

        assert(lslice.rows == rslice.rows)
        val offsets = batch.workArrayOffsetsForSpan(sent, span) 

        val wl = devLeft(offsets.map(_ + offset), ::).write(lslice,ev:_*)
        val wr = devRight(offsets.map(_ + offset), ::).write(rslice,ev:_*)
        evWrite += wl
        evWrite += wr
      }

      ev = evWrite
      offset += usedPerSplit
    }


    if(offset > 0) {
      maxOffset = maxOffset max offset
      ev = kernels.map{ kernel =>
        kernel.setArgs(devParent, devLeft, devRight, ruleDev, Integer.valueOf(numGPUCells), Integer.valueOf(offset))
        kernel.enqueueNDRange(queue, Array(offset), ev:_*)
      }
    } 

    if(maxOffset > usedPerSplit)
      ev = sumSplitBlocks(usedPerSplit, maxOffset, ev:_*)

    ev
  }

  def doTTUpdates(batch: Batch, events: CLEvent*) = {
    // layout:
    // Parents for each span of length 2 == n -1
    // [Sent0Span2Pos0, Sent0Span2Pos1, ..., Sent0Span2PosN-2, Sent1Span2Pos0, ...]
    // [Sent0Term0    , Sent0Term1    , ..., Sent0TermN-1    , Sent1Term0    , ...]
    // [Sent0Term1    , Sent0Term2    , ..., Sent0TermN      , Sent1Term1    , ...]
    doBinaryRules(batch, insideGen.insideTTKernels, 2, 1 to 1, _.bot, _.bot, events:_*)
  }


  def doTNUpdates(batch: Batch, span: Int, events: CLEvent*) = {
    doBinaryRules(batch, insideGen.insideTNKernels, span, 1 to 1, _.bot, _.top, events:_*)
  }

  def doNTUpdates(batch: Batch, span: Int, events: CLEvent*) = {
    doBinaryRules(batch, insideGen.insideNTKernels, span, (span-1) to (span-1), _.top, _.bot, events:_*)
  }

  def doUnaryUpdates(batch: Batch, span: Int, events: CLEvent*): IndexedSeq[CLEvent] = {
    import batch._
    val writeEvents = for(sent <- 0 until batch.numSentences) yield {
      val lslice = batch.gpuCharts(sent).bot.spanSlice(span)
      devLeft(workArrayOffsetsForSpan(sent, span), ::).write(lslice, events: _*)
    }

    val kernels = if(span == 1) insideGen.insideTUKernels else insideGen.insideNUKernels
    val endEvents = kernels.map{(kernel) =>
      kernel.setArgs(devParent, devLeft, ruleDev, Integer.valueOf(numGPUCells), Integer.valueOf(numGPUCells), Integer.valueOf(totalLengthForSpan(span)))
      kernel.enqueueNDRange(queue, Array(totalLengthForSpan(span)), writeEvents:_*)
    }
    endEvents
  }

  def doNNUpdates(batch: Batch, span: Int, maxLength: Int, events: CLEvent*) = {
    doBinaryRules(batch, insideGen.insideNNKernels, span, (1 until maxLength-span), _.bot, _.bot, events:_*)
  }

  private def sumSplitBlocks(targetSize: Int, currentSize: Int, events: CLEvent*):IndexedSeq[CLEvent] = {
    assert(currentSize % targetSize == 0)
    val multiple: Int = currentSize / targetSize
    val log2 = BitHacks.log2(multiple)
    if(log2 == 0) return events.toIndexedSeq
    val size2 = 1 << log2
    var ev:IndexedSeq[CLEvent] = events.toIndexedSeq
    if(log2 != multiple) {
      val difference = multiple - log2
      ev = IndexedSeq(sumGrammarCells(devParent.data, 0, devParent.data, currentSize - difference*targetSize, difference * targetSize, ev:_*))
    }
    var size = size2
    while (size > targetSize) {
      size /= 2
      ev = IndexedSeq(sumGrammarCells(devParent.data, 0, devParent.data, size, size, ev:_*))
    }
    assert(size == targetSize)
    ev
  }

  def sumGrammarCells(dest: CLBuffer[jl.Float], destOff: Int, src: CLBuffer[jl.Float], srcOff: Int, length: Int, events: CLEvent*) = {
    insideGen.sumGrammarCellsKernel.setArgs(dest, Integer.valueOf(destOff), src, Integer.valueOf(srcOff), Integer.valueOf(length))
    insideGen.sumGrammarCellsKernel.enqueueNDRange(queue, Array(length, cellSize), Array(32, 1, 1), events:_*)
  }

  private def sumBackToCharts(batch: Batch, level: ParseChart=>ChartHalf, span: Int, events: CLEvent*) = {
    val evs = for(sent <- 0 until batch.numSentences) yield {
      val offsets = batch.workArrayOffsetsForSpan(sent, span)
      val lslice = level(batch.gpuCharts(sent)).spanSlice(span)
      sumGrammarCells(lslice.data, lslice.offset, devParent.data, offsets.start, offsets.length, events:_*)
    }
    evs
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
