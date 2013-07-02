package puck
package newparser

import breeze.linalg._
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
import puck.linalg.kernels._
import puck.parser.gen.SemiringFloatOpsExp
import puck.util._
import scala.virtualization.lms.common.ArrayOpsExp
import trochee.kernels.KernelOpsExp
import puck.newparser.generator._
import collection.mutable.ArrayBuffer
import java.util
import java.util.zip._
import scala.collection.JavaConverters._



/**
 * TODO
 *
 * @author dlwh
 **/
class CLParser[C, L, W](data: CLParserData[C, L, W],
                        maxAllocSize: Long = 1<<30, // 1 gig
                        maxSentencesPerBatch: Long = 400, 
                        profile: Boolean = true)(implicit val context: CLContext) extends Logging {
  import data._

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

  def needsOutside = data.outside.nonEmpty
  def isViterbi = data.isViterbi


  def partitions(sentences: IndexedSeq[IndexedSeq[W]]):IndexedSeq[Float] = synchronized {
    {for {
      batch <- getBatches(sentences).iterator
      evi = inside(batch)
      _ = if(needsOutside) outside(batch, evi).waitFor() else evi.waitFor
      i <- 0 until batch.numSentences
    } yield {
      batch.insideCharts(i).top(0,batch.insideCharts(i).length, structure.root)
    }}.toIndexedSeq
  }


  private implicit val queue = if(profile) context.createDefaultProfilingQueue() else context.createDefaultQueue()
  private val memFillEvents  = new CLProfiler("memfill")
  private val hdTransferEvents  = new CLProfiler("Host2Dev Transfer")
  private val transferEvents  = new CLProfiler("Transfer")
  private val binaryEvents  = new CLProfiler("Binary")
  private val unaryEvents  = new CLProfiler("Unary")
  private val sumToChartsEvents  = new CLProfiler("SumToCharts")
  private val sumEvents  = new CLProfiler("Sum")
  val allProfilers =  IndexedSeq(transferEvents, binaryEvents, unaryEvents, sumToChartsEvents, sumEvents, memFillEvents, hdTransferEvents)

  def release() {
    devParent.release()
    devCharts.release()
    devLeft.release()
    devRight.release()
    devRules.release()
    queue.release()
  }


  val nrules = grammar.index.size
  // TODO: reinstate this difference if numTerms is really big.
  val cellSize = ((structure.numNonTerms max structure.numTerms)+31)/32 * 32

  val gen = new CLParserKernelGenerator[C, L](structure)

  val ruleScores = Array.tabulate(grammar.refinedGrammar.index.size){r =>
    val score = grammar.ruleScoreArray(grammar.refinements.rules.project(r))(grammar.refinements.rules.localize(r))
    gen.IR.fromLogSpace(score.toFloat)
  }

  val (numGPUCells:Int, numGPUChartCells: Int) = {
    val sizeOfFloat = 4
    val fractionOfMemoryToUse = 0.8 // slack!
    val amountOfMemory = ((context.getMaxMemAllocSize min maxAllocSize) * fractionOfMemoryToUse).toInt - ruleScores.length * sizeOfFloat - maxSentencesPerBatch * 3 * 4;
    val maxPossibleNumberOfCells = (amountOfMemory/sizeOfFloat) / cellSize toInt
    // We want numGPUCells and numGPUChartCells to be divisible by 16, so that we get aligned strided access:
    //       On devices of compute capability 1.0 or 1.1, the k-th thread in a half warp must access the
    //       k-th word in a segment aligned to 16 times the size of the elements being accessed; however,
    //       not all threads need to participate... If sequential threads in a half warp access memory that is
    //       sequential but not aligned with the segments, then a separate transaction results for each element
    //       requested on a device with compute capability 1.1 or lower.
    val numberOfUnitsOf16 = maxPossibleNumberOfCells / 16
    // average sentence length of sentence, let's say n.
    // for the gpu charts, we'll need (n choose 2) * 2 = n^2 - n cells (*2 more if needsOutside)
    // for the "P/L/R" parts, the maximum number of relaxations (P = L * R * rules) for a fixed span
    // in a fixed sentence is (n/2)^2= n^2/4.
    // Take n = 32, then we want our P/L/R arrays to be of the ratio (3 * 256):992 \approx 3/4 (3/4 exaclty if we exclude the - n term)
    //
    val relativeSizeOfChartsToP = if(needsOutside) 8 else 4
    val baseSize = numberOfUnitsOf16 / (3 + relativeSizeOfChartsToP)
    val extra = numberOfUnitsOf16 % (3 + relativeSizeOfChartsToP)
    val plrSize = baseSize
    // TODO, can probably do a better job of these calculations?
    (plrSize * 16, (baseSize * relativeSizeOfChartsToP + extra) * 16)
  }


  // On the Device side we have 4 Matrices:
  // One is where we calculate P = L * R * rules, for fixed spans and split points (the "bot")
  // One is the L part of the above
  // Another is the R part.
  // finally, we have the array of parse charts, which is insanely large. It's also where
  // we do rescaling, etc.
  private val devParent, devLeft, devRight = new CLMatrix[Float](numGPUCells, cellSize)
  // transposed
  private val devCharts = new CLMatrix[Float](cellSize, numGPUChartCells)

  // (dest, leftSource, rightSource) (right Source if applicable)
  val pArray, lArray, rArray = new Array[Int](numGPUCells)
  val splitPointOffsets = new Array[Int](numGPUCells+1)

  // also the rules
  private val devRules = context.createFloatBuffer(CLMem.Usage.Input, FloatBuffer.wrap(ruleScores), false)

  def _zero: Float = gen.IR._zero
  def _one: Float = gen.IR._one

  // other stuff
  private val zmk = ZeroMemoryKernel()
  private val transposeCopy = CLMatrixTransposeCopy()

  private case class Batch(lengthTotals: Array[Int],
                           cellTotals: Array[Int],
                           sentences: IndexedSeq[IndexedSeq[W]]) {
    def totalLength = lengthTotals.last
    def numSentences = sentences.length
    val maxLength = sentences.map(_.length).max


    val _workArrayOffsetsForSpan = Array.tabulate(maxLength+1)(span => sentences.scanLeft(0)((off, sent) => off + math.max(0,sent.length-span+1)).toArray)
    def workArrayOffsetsForSpan(sent: Int, span: Int) = Range(_workArrayOffsetsForSpan(span)(sent), _workArrayOffsetsForSpan(span)(sent+1)) 

    def totalLengthForSpan(span: Int) = _workArrayOffsetsForSpan(span).last

    lazy val insideCharts = for(i <- 0 until numSentences) yield {
      val numCells = (cellTotals(i+1)-cellTotals(i))/2
      assert(numCells == TriangularArray.arraySize(sentences(i).length+1))
      val chart = new ParseChart(sentences(i).length, devCharts(::, cellTotals(i) until (cellTotals(i) + numCells)), devCharts(::, cellTotals(i) + numCells until cellTotals(i+1)))
      chart
    }

    lazy val outsideCharts = if(!needsOutside) None else Some{
      for(i <- 0 until numSentences) yield {

        val numCells = (cellTotals(i+1)-cellTotals(i))/2
        assert(numCells == TriangularArray.arraySize(sentences(i).length+1))
        val botBegin = cellTotals.last + cellTotals(i)
        val botEnd = botBegin + numCells
        val topBegin = botEnd
        val topEnd = topBegin + numCells
        val chart = new ParseChart(sentences(i).length, devCharts(::, botBegin until botEnd), devCharts(::, topBegin until topEnd))
        chart
      }
    }

    


    def initializeTagScores(events: CLEvent*) = {
      val dm = DenseMatrix.zeros[Float](totalLength, cellSize)
      dm := _zero
      var offset = 0
      for(i <- 0 until numSentences par) {
        tagScoresFor(dm, i)
        val parent = insideCharts(i).bot.spanRangeSlice(1)
        parent.copyToArray(pArray, _workArrayOffsetsForSpan(1)(i))
      }
      val ev = devParent(0 until totalLength, ::).writeFrom(dm, false, events:_*)
      transposeCopy.permuteTransposeCopyOut(devCharts, pArray, totalLengthForSpan(1), devParent(0 until totalLength, ::), ev:_*)
    }

    private def tagScoresFor(dm: DenseMatrix[Float], i: Int) {
      val sent = sentences(i)
      val tags = dm(workArrayOffsetsForSpan(i, 1), ::)
      val anch = grammar.tagScorer.anchor(sent)
      val lexAnch = grammar.lexicon.anchor(sent)
      for(pos <- 0 until sent.length; t <- lexAnch.allowedTags(pos); ref <- grammar.refinements.labels.refinementsOf(t)) {
        val index = ref
        val score = anch.scoreTag(pos, grammar.refinedGrammar.labelIndex.get(index))
        val gpuIndex = structure.labelIndexToTerminal(index)
        tags(pos, gpuIndex) = gen.IR.fromLogSpace(score.toFloat)
      }
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
      if(needsOutside)
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
    val cellTotals = sentences.scanLeft(0)((acc, sent) => acc + TriangularArray.arraySize(sent.length+1) * 2)
    Batch(lengthTotals.toArray, cellTotals.toArray, sentences)
  }

  def debugRaces = false
  def debugCharts = true

  def debugFinish() {
    if(debugRaces || debugCharts) queue.finish()
  }

  def debugCharts(batch: Batch) {
    if(debugCharts) {
      println("inside")
      println(batch.insideCharts(0).bot.toString(structure, _zero))
      println(batch.insideCharts(0).top.toString(structure, _zero))
      if(needsOutside) {

        println("outside")
        println(batch.outsideCharts.get(0).bot.toString(structure, _zero))
        println(batch.outsideCharts.get(0).top.toString(structure, _zero))
      }
    }
  }

  println(structure.nontermIndex.zipWithIndex)
  println(structure.termIndex.zipWithIndex)

  private def inside(batch: Batch):CLEvent = synchronized {
    allProfilers.foreach(_.clear())
    allProfilers.foreach(_.tick())

    var eZC = zmk.fillMemory(devCharts.data, _zero)
    val init = batch.initializeTagScores(eZC)
    hdTransferEvents += init
    memFillEvents += eZC

    var ev = doUnaryUpdates(batch, 1, init)

    debugFinish()
    //debugCharts(batch)

    for(span <- 2 to batch.maxLength) {
      println(span)
      ev = insideBinaryPass(batch, span, ev)
      ev = doUnaryUpdates(batch, span, ev)
      debugFinish()
    }
    debugCharts(batch)

    debugFinish()

    if(profile) {
      queue.finish()
      allProfilers.foreach(_.tock())
      allProfilers.foreach(p => println(s"Inside $p"))
    }

    ev
  }

  private def outside(batch: Batch, event: CLEvent):CLEvent = synchronized {
    allProfilers.foreach(_.clear())
    allProfilers.foreach(_.tick())

    var ev = event

    ev = data.util.setRootScores(devCharts, batch.outsideCharts.get.map(_.top.rootIndex).toArray, structure.root, _one, ev)
    ev = doOutsideUnaryUpdates(batch, batch.maxLength, ev)

    for(span <- (batch.maxLength-1) to 1 by -1) {
      println(span)
      ev = outsideBinaryPass(batch, span, ev)
      ev = doOutsideUnaryUpdates(batch, span, ev)
      debugFinish()
    }
    debugCharts(batch)

    debugFinish()

    if(profile) {
      queue.finish()
      allProfilers.foreach(_.tock())
      allProfilers.foreach(p => println(s"Outside $p"))
    }

    ev
  }



  private def insideBinaryPass(batch: Batch, span: Int, events: CLEvent*) = {
    var ev = events
    if(span == 2) {
      ev = insideTT.doUpdates(batch, span, ev :_*)
    }

    ev = insideNT.doUpdates(batch, span, ev :_*)
    ev = insideTN.doUpdates(batch, span, ev :_*)
    ev = insideNN.doUpdates(batch, span, ev :_*)
    assert(ev.length == 1)
    ev.head
  }

  private val insideBot = {(b: Batch, s: Int) =>  b.insideCharts(s).bot}
  private val insideTop = {(b: Batch, s: Int) =>  b.insideCharts(s).top}
  //private val outsideBot = {(b: Batch, s: Int) =>  b.insideCharts(s).bot}
  //private val outsideTop = {(b: Batch, s: Int) =>  b.insideCharts(s).top}
  private val outsideBot = {(b: Batch, s: Int) =>  b.outsideCharts.get(s).bot}
  private val outsideTop = {(b: Batch, s: Int) =>  b.outsideCharts.get(s).top}

  private val insideTT = new BinaryUpdateManager(data.inside.insideTTKernels, insideBot, insideBot, insideBot, (b, e, l) => (b+1 to b+1))
  private val insideNT = new BinaryUpdateManager(data.inside.insideNTKernels, insideBot, insideTop, insideBot, (b, e, l) => (e-1 to e-1))
  private val insideTN = new BinaryUpdateManager(data.inside.insideTNKernels, insideBot, insideBot, insideTop, (b, e, l) => (b+1 to b+1))
  private val insideNN = new BinaryUpdateManager(data.inside.insideNNKernels, insideBot, insideTop, insideTop, (b, e, l) => (b+1 to e-1))

  private def outsideBinaryPass(batch: Batch, span: Int, events: CLEvent*) = {
    var ev = events
    if(span == 1) {
      //ev = outsideTT_L.doUpdates(batch, span, ev :_*)
      //ev = outsideTT_R.doUpdates(batch, span, ev :_*)
      //ev = outsideNT_R.doUpdates(batch, span, ev :_*)
      //ev = outsideTN_L.doUpdates(batch, span, ev :_*)
    }

    ev = outsideTN_R.doUpdates(batch, span, ev :_*)
    ev = outsideNT_L.doUpdates(batch, span, ev :_*)
    ev = outsideNN_L.doUpdates(batch, span, ev :_*)
    ev = outsideNN_R.doUpdates(batch, span, ev :_*)
    assert(ev.length == 1)
    ev.head
  }



  // here, "parentChart" is actually the left child, left is the parent, right is the right completion
  private lazy val outsideTT_L = new BinaryUpdateManager(data.outside.get.outside_L_TTKernels, outsideBot, outsideBot, insideBot, (b, e, l) => (e+1 to e+1))
  private lazy val outsideNT_L = new BinaryUpdateManager(data.outside.get.outside_L_NTKernels, outsideTop, outsideBot, insideBot, (b, e, l) => (e+1 to e+1))
  private lazy val outsideTN_L = new BinaryUpdateManager(data.outside.get.outside_L_TNKernels, outsideBot, outsideBot, insideTop, (b, e, l) => (e+1 to l))
  private lazy val outsideNN_L = new BinaryUpdateManager(data.outside.get.outside_L_NNKernels, outsideTop, outsideBot, insideTop, (b, e, l) => (e+1 to l))

  // here, "parentChart" is actually the right child, right is the parent, left is the left completion
  private lazy val outsideTT_R = new BinaryUpdateManager(data.outside.get.outside_R_TTKernels, outsideBot, insideBot, outsideBot, (b, e, l) => (b-1 to b-1))
  private lazy val outsideNT_R = new BinaryUpdateManager(data.outside.get.outside_R_NTKernels, outsideBot, insideTop, outsideBot, (b, e, l) => (0 to b-1))
  private lazy val outsideTN_R = new BinaryUpdateManager(data.outside.get.outside_R_TNKernels, outsideTop, insideBot, outsideBot, (b, e, l) => (b-1 to b-1))
  private lazy val outsideNN_R = new BinaryUpdateManager(data.outside.get.outside_R_NNKernels, outsideTop, insideTop, outsideBot, (b, e, l) => (0 to b-1))



  private class BinaryUpdateManager(kernels: IndexedSeq[CLKernel],
    parentChart: (Batch,Int)=>ChartHalf,
    leftChart: (Batch,Int)=>ChartHalf,
    rightChart: (Batch,Int)=>ChartHalf,
    ranger: (Int, Int, Int)=>Range) {

    private var parentOffset = 0 // number of unique parent spans used so far
    private var offset = 0 // number of cells used so far.

    // TODO: ugh, state
    private var ev:Seq[CLEvent] = IndexedSeq.empty[CLEvent]
    private var span = 1

    def doUpdates(batch: Batch, span: Int, events: CLEvent*) = {
      ev = events
      this.span = span

      for{
        sent <- 0 until batch.numSentences
        start <- 0 to batch.sentences(sent).length - span
        split <- ranger(start, start + span, batch.sentences(sent).length) if split >= 0 && split <= batch.sentences(sent).length
      } {
        val end = start + span
        val parentTi = parentChart(batch, sent).treeIndex(start,end)
        val leftChild = if(split < start) leftChart(batch, sent).treeIndex(split,start) else leftChart(batch, sent).treeIndex(start, split)
        val rightChild = if(split < end) rightChart(batch, sent).treeIndex(split,end) else rightChart(batch, sent).treeIndex(end, split)
        enqueue(parentTi, leftChild, rightChild)
      }

      if(offset > 0) {
        flushQueue()
      } 

      ev
    }

    def enqueue(parent: Int, left: Int, right: Int) {
      if(parentOffset == 0 || pArray(parentOffset-1) != parent) {
        splitPointOffsets(parentOffset) = offset
        pArray(parentOffset) = parent
        parentOffset += 1
      }
      lArray(offset) = left
      rArray(offset) = right
      offset += 1
      if(offset >= numGPUCells)  {
        println(s"flush!")
        flushQueue()
      }
    }

    def flushQueue() {
      if(offset != 0) {
        splitPointOffsets(parentOffset) = offset
        val wl = transposeCopy.permuteTransposeCopy(devLeft(0 until offset, ::), devCharts, java.util.Arrays.copyOf(lArray, offset), ev:_*)
        val wr = transposeCopy.permuteTransposeCopy(devRight(0 until offset, ::), devCharts, java.util.Arrays.copyOf(rArray, offset), ev:_*)
        transferEvents += wl
        transferEvents += wr
        val zz = zmk.shapedFill(devParent(0 until offset, ::), _zero, ev:_*)
        memFillEvents += zz
        ev = kernels.map{ kernel =>
          kernel.setArgs(devParent.data.safeBuffer, devLeft.data.safeBuffer, devRight.data.safeBuffer, devRules, Integer.valueOf(numGPUCells), Integer.valueOf(offset))
          kernel.enqueueNDRange(queue, Array(offset), wl, wr, zz)
        } 

        binaryEvents ++= ev
        val sumEv = data.util.sumSplitPoints(devParent,
          devCharts,
          java.util.Arrays.copyOf(pArray, parentOffset),
          java.util.Arrays.copyOf(splitPointOffsets, parentOffset + 1),
          32 / span max 1, ev:_*)
        sumEvents += sumEv
        ev = IndexedSeq(sumEv)
        offset = 0
        parentOffset = 0
      }
    }
  }

  def doUnaryUpdates(batch: Batch, span: Int, events: CLEvent*) = {
    import batch._
    var offset = 0
    def enqueue(parent: IndexedSeq[Int], left: IndexedSeq[Int]) {
      parent.copyToArray(pArray, offset)
      left.copyToArray(lArray, offset)
      offset += parent.length
    }
    for(sent <- 0 until batch.numSentences if batch.sentences(sent).length >= span) {
      enqueue(batch.insideCharts(sent).top.spanRangeSlice(span),
              batch.insideCharts(sent).bot.spanRangeSlice(span))
    }

    val zz = zmk.shapedFill(devParent(0 until offset, ::), _zero, events:_*)
    memFillEvents += zz

    val wl = transposeCopy.permuteTransposeCopy(devLeft(0 until offset, ::), devCharts, java.util.Arrays.copyOf(lArray, offset), events:_*)
    transferEvents += wl
    debugFinish()

    val kernels = if(span == 1) data.inside.insideTUKernels else data.inside.insideNUKernels
    val endEvents = kernels.map{(kernel) =>
      kernel.setArgs(devParent.data.safeBuffer, devLeft.data.safeBuffer, devRules, Integer.valueOf(numGPUCells), Integer.valueOf(numGPUCells), Integer.valueOf(totalLengthForSpan(span)))
      kernel.enqueueNDRange(queue, Array(totalLengthForSpan(span)), wl, zz)
    }
    debugFinish()
    unaryEvents ++= endEvents
    
    val ev = transposeCopy.permuteTransposeCopyOut(devCharts, pArray, offset, devParent(0 until offset, ::), endEvents:_*)
    sumToChartsEvents += ev
    ev
  }

  def doOutsideUnaryUpdates(batch: Batch, span: Int, events: CLEvent*) = {
    import batch._
    var offset = 0
    def enqueue(parent: IndexedSeq[Int], left: IndexedSeq[Int]) {
      parent.copyToArray(pArray, offset)
      left.copyToArray(lArray, offset)
      offset += parent.length
    }
    for(sent <- 0 until batch.numSentences if batch.sentences(sent).length >= span) {
      enqueue(batch.outsideCharts.get(sent).bot.spanRangeSlice(span),
              batch.outsideCharts.get(sent).top.spanRangeSlice(span))
    }

    val zz = zmk.shapedFill(devParent(0 until offset, ::), _zero, events:_*)
    memFillEvents += zz

    val wl = transposeCopy.permuteTransposeCopy(devLeft(0 until offset, ::), devCharts, java.util.Arrays.copyOf(lArray, offset), events:_*)
    transferEvents += wl
    debugFinish()

    val kernels = if(span == 1) data.outside.get.outsideTUKernels else data.outside.get.outsideNUKernels
    val endEvents = kernels.map{(kernel) =>
      kernel.setArgs(devParent.data.safeBuffer, devLeft.data.safeBuffer, devRules, Integer.valueOf(numGPUCells), Integer.valueOf(numGPUCells), Integer.valueOf(totalLengthForSpan(span)))
      kernel.enqueueNDRange(queue, Array(totalLengthForSpan(span)), wl, zz)
    }
    debugFinish()
    unaryEvents ++= endEvents
    
    val ev = transposeCopy.permuteTransposeCopyOut(devCharts, pArray, offset, devParent(0 until offset, ::), endEvents:_*)
    sumToChartsEvents += ev
    ev
  }


}

object CLParser extends Logging {

  case class Params(annotator: TreeAnnotator[AnnotatedLabel, String, AnnotatedLabel] = FilterAnnotations(),
                    useGPU: Boolean = true, profile: Boolean = false, numToParse: Int = 1000, jvmParse: Boolean = false, parseTwice: Boolean = false)

  def main(args: Array[String]) = {
    import ParserParams.JointParams

    val params = CommandLineParser.readIn[JointParams[Params]](args)
    import params.trainer._
    println("Training Parser...")
    println(params)
    val transformed = params.treebank.trainTrees.par.map { ti => annotator(ti) }.seq.toIndexedSeq
    val grammar: SimpleRefinedGrammar[AnnotatedLabel, AnnotatedLabel, String] = GenerativeParser.extractGrammar(AnnotatedLabel.TOP, transformed)


    implicit val context = if(useGPU) {
      val gpu = JavaCL.listPlatforms.flatMap(_.listGPUDevices(true)).head
      JavaCL.createContext(new java.util.HashMap(), gpu)
    } else {
      val cpuPlatform:CLPlatform = JavaCL.listPlatforms().filter(_.listCPUDevices(true).nonEmpty).head
      cpuPlatform.createContext(new java.util.HashMap(), cpuPlatform.listCPUDevices(true):_*)
    }
    println(context)

    val kern = fromSimpleGrammar[AnnotatedLabel, AnnotatedLabel, String](grammar, profile)
    val train = transformed.slice(0, numToParse).map(_.words)

    var timeIn = System.currentTimeMillis()
    val parts = kern.partitions(train)
    var timeOut = System.currentTimeMillis()
    println(parts)
    println(s"CL Parsing took: ${(timeOut-timeIn)/1000.0}")
    if(parseTwice) {
      timeIn = timeOut
      val parts2 = kern.partitions(train)
      timeOut = System.currentTimeMillis()
      println(parts2)
      println(s"CL Parsing took x2: ${(timeOut-timeIn)/1000.0}")
    }
    println("isViterbi?!?!?")
    if(jvmParse) {
      timeIn = timeOut
      val margs = train.map { w => 
        val m = ChartMarginal(AugmentedGrammar.fromRefined(grammar).anchor(w), w, maxMarginal= kern.isViterbi)
        printChart(m, true, true)
        printChart(m, false, true)
        m.logPartition.toFloat
      }
      timeOut = System.currentTimeMillis()
      println(s"Scala Parsing took: ${(timeOut-timeIn)/1000.0}")
      println(margs)
    }

    kern.release()
    context.release()

  }

  def fromSimpleGrammar[L, L2, W](grammar: SimpleRefinedGrammar[L, L2, W], profile: Boolean = false)(implicit context: CLContext) = {
    val data = CLParserData.make(grammar, new CLParserKernelGenerator(grammar))
    val kern = new CLParser(data, profile = profile)
    kern
  }

  private def printChart[L, W](chart: ChartMarginal[L, W], isBot: Boolean, isOutside: Boolean) = {
    val cc1 = if(isOutside) chart.outside else chart.inside
    val cc = if (isBot) cc1.bot else cc1.top
    val m = chart
    if(isOutside) println("outside")
    for(span <- 1 to m.length; begin <- 0 to m.length-span)
      println(cc.enteredLabelScores(begin,begin+span).map{ case (k,v) => (k,v.map(_.toFloat).mkString("{",",","}"))}.mkString(s"!!($begin,${begin+span}) ${if(isBot) "bot" else "top"} {",", ", "}"))
  }
}


case class CLParserData[C, L, W](grammar: SimpleRefinedGrammar[C, L, W],
                                 structure: RuleStructure[C, L],
                                 inside: CLInsideKernels,
                                 outside: Option[CLOutsideKernels],
                                 util: CLParserUtilKernels,
                                 isViterbi: Boolean) {


  def write(out: OutputStream) {
    val zout = new ZipOutputStream(out)
    ZipUtil.serializedEntry(zout, "grammar", grammar)
    ZipUtil.serializedEntry(zout, "structure", structure)
    inside.write(zout)
    outside.foreach(_.write(zout))
    util.write(zout)
    zout.close()
  }
}

object CLParserData {
  def make[C, L, W](grammar: SimpleRefinedGrammar[C, L, W], gen: CLParserKernelGenerator[C, L])(implicit context: CLContext) = {
    val inside = CLInsideKernels.make(gen)
    val outside =  Some(CLOutsideKernels.make(gen)) //if(!gen.isViterbi) Some(CLOutsideKernels.make(gen)) else None
    val util = CLParserUtilKernels.make(gen)
    new CLParserData(grammar, gen.structure, inside, outside, util, gen.isViterbi)
  }

  def read[C, L, W](file: ZipFile)(implicit context: CLContext) = {
    val gr = ZipUtil.deserializeEntry[SimpleRefinedGrammar[C, L, W]](file.getInputStream(file.getEntry("grammar")))
    val structure = ZipUtil.deserializeEntry[RuleStructure[C, L]](file.getInputStream(file.getEntry("structure")))
    val inside = CLInsideKernels.read(file)
    val outside = CLOutsideKernels.tryRead(file)
    val util = CLParserUtilKernels.read(file)

    CLParserData(gr, structure, inside, outside, util, ???)
  }
}


