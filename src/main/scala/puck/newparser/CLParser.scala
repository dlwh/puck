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


  def partitions(sentences: IndexedSeq[IndexedSeq[W]]):IndexedSeq[Float] = synchronized {
    {for {
      batch <- getBatches(sentences).iterator
      _ = inside(batch).foreach(_.waitFor())
      i <- 0 until batch.numSentences
    } yield {
      batch.gpuCharts(i).top(0,batch.gpuCharts(i).length, structure.root)
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
  val allProfilers =  IndexedSeq(transferEvents, binaryEvents, unaryEvents, sumToChartsEvents, sumEvents, memFillEvents)

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

  // also the rules
  private val devRules = context.createFloatBuffer(CLMem.Usage.Input, FloatBuffer.wrap(ruleScores), false)

  def _zero: Float = gen.IR._zero

  // other stuff
  private val zmk = ZeroMemoryKernel()
  private val transposeCopy = CLMatrixTransposeCopy()

  private case class Batch(lengthTotals: Array[Int],
                           cellTotals: Array[Int],
                           sentences: IndexedSeq[IndexedSeq[W]]) {
    def totalLength = lengthTotals.last
    def numSentences = sentences.length
    val maxLength = sentences.map(_.length).max


    val _workArrayOffsetsForSpan = Array.tabulate(maxLength+1)(span => sentences.scanLeft(0)((off, sent) => off + math.max(0,sent.length-span+1)))
    def workArrayOffsetsForSpan(sent: Int, span: Int) = Range(_workArrayOffsetsForSpan(span)(sent), _workArrayOffsetsForSpan(span)(sent+1)) 

    def totalLengthForSpan(span: Int) = _workArrayOffsetsForSpan(span).last

    lazy val gpuCharts = for(i <- 0 until numSentences) yield {
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
      for(i <- 0 until numSentences par) {
        tagScoresFor(dm, i)
      }
      val ev = devParent(0 until totalLength, ::).writeFrom(dm, false, events:_*)
      val offsets = gpuCharts.map(_.bot.spanRangeSlice(1)).reduceLeft[IndexedSeq[Int]](_ ++ _).toArray
      IndexedSeq(transposeCopy.permuteTransposeCopyOut(devCharts, offsets, offsets.length, devParent(0 until totalLength, ::), ev:_*))
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

  def debugFinish() = if(debugRaces) queue.finish()


  private def inside(batch: Batch):Seq[CLEvent] = synchronized {
    val prof = new CLProfiler("...")
    prof.clear()
    prof.tick()
    hdTransferEvents.clear()
    hdTransferEvents.tick()

    var eZC = zmk.fillMemory(devCharts.data, _zero)
    val init = batch.initializeTagScores(eZC)
    hdTransferEvents ++= init
    var eZp = zmk.fillMemory(devParent.data, _zero)
    memFillEvents += eZC
    memFillEvents += eZp
    allProfilers.foreach(_.clear())
    allProfilers.foreach(_.tick())



    var events:Seq[CLEvent] = doUnaryUpdates(batch, 1, eZp +: init :_*)
    events = copyBackToCharts(batch, _.top, 1, events :_*)
    events = IndexedSeq(zmk.fillMemory(devParent.data, _zero, events:_*))
    memFillEvents ++= events
    debugFinish()

    for(span <- 2 to batch.maxLength) {
      println(span)
      // TODO: there's got to be a better way. implicits?
      events = Seq[Seq[CLEvent]=>Seq[CLEvent]](
        binaryPass(batch, span, _:_*),
        {(x: Seq[CLEvent]) => val ee = Seq(zmk.fillMemory(devParent.data, _zero, x:_*)); memFillEvents ++= ee; ee},
        doUnaryUpdates(batch, span, _ : _*),
        copyBackToCharts(batch, _.top, span, _ :_*),
        {(x: Seq[CLEvent]) => val ee = Seq(zmk.fillMemory(devParent.data, _zero, x:_*)); memFillEvents ++= ee; ee}
      ).foldLeft(events)((a,b) => b apply a)
      debugFinish()
    }

    debugFinish()

    if(profile) {
      prof.tock()
      println(prof)
      queue.finish()
      allProfilers.foreach(_.tock())
      hdTransferEvents.tock()
      allProfilers.foreach(p => println(s"Inside $p"))
      println("Inside " + hdTransferEvents)
    }

    events
  }

  private def binaryPass(batch: Batch, span: Int, events: CLEvent*) = {
    var ev = events
    if(span == 2) {
      ev = doTTUpdates(batch, ev:_*)
    }

    ev = doNTUpdates(batch, span, ev :_*)
    ev = doTNUpdates(batch, span, ev :_*)
    ev = doNNUpdates(batch, span, batch.maxLength, ev :_*)
    ev = copyBackToCharts(batch, _.bot, span, ev :_*)
    ev
  }

  // (dest, leftSource, rightSource) (right Source if applicable)
  val pArray, lArray, rArray = new Array[Int](numGPUCells)

  private def doBinaryUpdates(batch: Batch,
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
    assert(splitRange.last < span, splitRange + " " + span)

    def flushQueue() {
      if(offset != 0) {
        val wl = transposeCopy.permuteTransposeCopy(devLeft(0 until offset, ::), devCharts, lArray.take(offset), ev:_*)
        val wr = transposeCopy.permuteTransposeCopy(devRight(0 until offset, ::), devCharts, rArray.take(offset), ev:_*)
        transferEvents += wl
        transferEvents += wr
        ev = kernels.map{ kernel =>
          kernel.setArgs(devParent.data.safeBuffer, devLeft.data.safeBuffer, devRight.data.safeBuffer, devRules, Integer.valueOf(numGPUCells), Integer.valueOf(offset))
          kernel.enqueueNDRange(queue, Array(offset), wl, wr)
        } 
        binaryEvents ++= ev
        maxOffset = maxOffset max offset
        offset = 0
      }
    }

    def enqueue(parent: IndexedSeq[Int], left: IndexedSeq[Int], right: IndexedSeq[Int]) {
      parent.copyToArray(pArray, offset)
      left.copyToArray(lArray, offset)
      right.copyToArray(rArray, offset)
      offset += parent.length
    }

    for(leftChildLength <- splitRange) {
      //println(span,leftChildLength)
      // if we fill up the buffer, run a pass.
      if(offset + usedPerSplit >= numGPUCells)  {
        println(s"flush! used $offset of $numGPUCells. Another split needs $usedPerSplit.")
        flushQueue()
      }

      // add next split point
      for(sent <- 0 until batch.numSentences) {
        if(span <= batch.sentences(sent).length) {
          val lslice:Range = leftChart(batch.gpuCharts(sent)).spanRangeSlice(leftChildLength, 0, batch.gpuCharts(sent).length-span+1)
          val rslice:Range = rightChart(batch.gpuCharts(sent)).spanRangeSlice(span-leftChildLength, leftChildLength)
          val offsets = batch.gpuCharts(sent).bot.spanRangeSlice(span) 
          assert(lslice.length == rslice.length, lslice.length + " " + rslice.length + " " + offsets.length)
          assert(offsets.length == lslice.length)
          enqueue(offsets, lslice, rslice)
        }
      }

    }



    if(offset > 0) {
      flushQueue()
    } 

    if(maxOffset > usedPerSplit)
      ev = sumEvents.adding(sumSplitBlocks(usedPerSplit, maxOffset, ev:_*))

    ev
  }

  def doTTUpdates(batch: Batch, events: CLEvent*) = {
    // layout:
    // Parents for each span of length 2 == n -1
    // [Sent0Span2Pos0, Sent0Span2Pos1, ..., Sent0Span2PosN-2, Sent1Span2Pos0, ...]
    // [Sent0Term0    , Sent0Term1    , ..., Sent0TermN-1    , Sent1Term0    , ...]
    // [Sent0Term1    , Sent0Term2    , ..., Sent0TermN      , Sent1Term1    , ...]
    doBinaryUpdates(batch, data.inside.insideTTKernels, 2, 1 to 1, _.bot, _.bot, events:_*)
  }


  def doTNUpdates(batch: Batch, span: Int, events: CLEvent*) = {
    doBinaryUpdates(batch, data.inside.insideTNKernels, span, 1 to 1, _.bot, _.top, events:_*)
  }

  def doNTUpdates(batch: Batch, span: Int, events: CLEvent*) = {
    doBinaryUpdates(batch, data.inside.insideNTKernels, span, (span-1) to (span-1), _.top, _.bot, events:_*)
  }

  def doUnaryUpdates(batch: Batch, span: Int, events: CLEvent*): IndexedSeq[CLEvent] = {
    import batch._
    var offset = 0
    def enqueue(parent: IndexedSeq[Int], left: IndexedSeq[Int]) {
      parent.copyToArray(pArray, offset)
      left.copyToArray(lArray, offset)
      offset += parent.length
    }
    for(sent <- 0 until batch.numSentences if batch.sentences(sent).length >= span) {
      enqueue(batch.gpuCharts(sent).top.spanRangeSlice(span),
              batch.gpuCharts(sent).bot.spanRangeSlice(span))
    }

    val wl = transposeCopy.permuteTransposeCopy(devLeft(0 until offset, ::), devCharts, lArray.take(offset), events:_*)
    transferEvents += wl
    debugFinish()


    val kernels = if(span == 1) data.inside.insideTUKernels else data.inside.insideNUKernels
    val endEvents = kernels.map{(kernel) =>
      kernel.setArgs(devParent.data.safeBuffer, devLeft.data.safeBuffer, devRules, Integer.valueOf(numGPUCells), Integer.valueOf(numGPUCells), Integer.valueOf(totalLengthForSpan(span)))
      kernel.enqueueNDRange(queue, Array(totalLengthForSpan(span)), wl)
    }
    debugFinish()
    unaryEvents ++= endEvents
    endEvents
  }

  def doNNUpdates(batch: Batch, span: Int, maxLength: Int, events: CLEvent*) = {
    doBinaryUpdates(batch, data.inside.insideNNKernels, span, (1 to span-1), _.top, _.top, events:_*)
  }

  private def sumSplitBlocks(targetSize: Int, currentSize: Int, events: CLEvent*):IndexedSeq[CLEvent] = {
    assert(currentSize % targetSize == 0)
    val multiple: Int = currentSize / targetSize
    val log2 = BitHacks.log2(multiple)
    if(log2 == 0) return events.toIndexedSeq

    var currentMultiple = 1 << log2

    var ev:IndexedSeq[CLEvent] = events.toIndexedSeq
    if(currentMultiple != multiple) {
      val difference = multiple - currentMultiple
      ev = IndexedSeq(sumGrammarCells(devParent(0 until difference * targetSize, ::), devParent(currentSize - difference*targetSize until currentSize, ::), ev:_*))
    }

    while (currentMultiple > 1) {
      currentMultiple /= 2
      ev = IndexedSeq(sumGrammarCells(devParent(0 until currentMultiple * targetSize, ::), devParent(currentMultiple * targetSize until 2 * currentMultiple * targetSize, ::), ev:_*))
    }
    assert(currentMultiple == 1, currentMultiple + " " + targetSize + " " + currentSize)
    
    ev
  }

  private def sumGrammarCells(dest: CLMatrix[Float], src: CLMatrix[Float], events: CLEvent*) = data.util.sumGrammarKernel.synchronized {
    assert(dest.rows == src.rows)
    assert(dest.size == src.size)
    data.util.sumGrammarKernel.setArgs(dest.data.safeBuffer, Integer.valueOf(dest.offset), Integer.valueOf(dest.majorStride),
                                             src.data.safeBuffer, Integer.valueOf(src.offset),  Integer.valueOf(src.majorStride),
                                             Integer.valueOf(dest.cols), Integer.valueOf(dest.size))
    data.util.sumGrammarKernel.enqueueNDRange(queue, Array(dest.rows, dest.cols), /*Array(32, 1),*/ events:_*)
  }

  private def copyBackToCharts(batch: Batch, level: ParseChart=>ChartHalf, span: Int, events: CLEvent*):Seq[CLEvent] = {
    val totalLength = batch.totalLengthForSpan(span)
    val ev = if(span == 1) {
      val treeOffsets = batch.gpuCharts.filter(_.length >= span).map(c => level(c).spanRangeSlice(span)).reduceLeft[IndexedSeq[Int]](_ ++ _).toArray
      transposeCopy.permuteTransposeCopyOut(devCharts, treeOffsets, totalLength, devParent(0 until treeOffsets.length, ::), events:_*)
    } else {
      transposeCopy.permuteTransposeCopyOut(devCharts, pArray, totalLength, devParent(0 until totalLength, ::), events:_*)
    }
    sumToChartsEvents += ev
    IndexedSeq(ev)
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
    val train = transformed.slice(0,numToParse).map(_.words)
    

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
    println("Needs outside?!?!?" + kern.needsOutside)
    if(jvmParse) {
      timeIn = timeOut
      val margs = train.map { w => 
        val m = ChartMarginal(AugmentedGrammar.fromRefined(grammar).anchor(w), w, maxMarginal= !kern.needsOutside)
        m.logPartition
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

  private def printChart[L, W](chart: ChartMarginal[L, W], isBot: Boolean) = {
    val cc = if (isBot) chart.inside.bot else chart.inside.top
    val m = chart
    for(span <- 1 to m.length; begin <- 0 to m.length-span)
      println(cc.enteredLabelScores(begin,begin+span).map{ case (k,v) => (k,v.mkString("{",",","}"))}.mkString(s"($begin,${begin+span}) ${if(isBot) "bot" else "top"} {",", ", "}"))
  }
}


case class CLParserData[C, L, W](grammar: SimpleRefinedGrammar[C, L, W],
                                 structure: RuleStructure[C, L],
                                 inside: CLInsideKernels,
                                 outside: Option[CLOutsideKernels],
                                 util: CLParserUtilKernels) {


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
    val outside = if(!gen.isViterbi) Some(CLOutsideKernels.make(gen)) else None
    val util = CLParserUtilKernels.make(gen)
    new CLParserData(grammar, gen.structure, inside, outside, util)
  }

  def read[C, L, W](file: ZipFile)(implicit context: CLContext) = {
    val gr = ZipUtil.deserializeEntry[SimpleRefinedGrammar[C, L, W]](file.getInputStream(file.getEntry("grammar")))
    val structure = ZipUtil.deserializeEntry[RuleStructure[C, L]](file.getInputStream(file.getEntry("structure")))
    val inside = CLInsideKernels.read(file)
    val outside = CLOutsideKernels.tryRead(file)
    val util = CLParserUtilKernels.read(file)

    CLParserData(gr, structure, inside, outside, util)
  }
}


