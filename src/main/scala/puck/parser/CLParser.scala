package puck
package parser

import gen._
import com.nativelibs4java.opencl._
import com.typesafe.scalalogging.log4j.Logging
import puck.util.{ZipUtil, BitHacks, ZeroMemoryKernel, CLProfiler}
import puck.linalg.CLMatrix
import java.nio.{ByteOrder, ByteBuffer, FloatBuffer}
import puck.linalg.kernels.CLMatrixTransposeCopy
import breeze.collection.mutable.TriangularArray
import breeze.linalg.{Counter, DenseVector, DenseMatrix}
import epic.trees.annotations.{Xbarize, TreeAnnotator, FilterAnnotations}
import epic.parser._
import breeze.config.CommandLineParser
import epic.trees._
import java.io.{BufferedOutputStream, FileOutputStream, File, OutputStream}
import java.util.zip.{ZipInputStream, ZipFile, ZipOutputStream}
import scala.collection.mutable.ArrayBuffer
import java.util
import BitHacks._
import org.bridj.Pointer
import scala.Array

/**
 * TODO
 *
 * @author dlwh
 **/
class CLParser[C, L, W](data: IndexedSeq[CLParserData[C, L, W]],
                        maxAllocSize: Long = 1<<30, // 1 gig
                        maxSentencesPerBatch: Long = 400,
                        profile: Boolean = true)(implicit val context: CLContext) extends Logging {

  def grammar = data.head.grammar


  def parse(sentences: IndexedSeq[IndexedSeq[W]]):IndexedSeq[BinarizedTree[L]] = synchronized {
    getBatches(sentences).iterator.flatMap{ batch =>
      val finalBatch = parsers.take(data.length-1).foldLeft(batch){(b, parser) =>
        var ev = parser.inside(batch)
        ev = if(needsOutside) parser.outside(batch, ev) else ev
        parser.addMasksToBatches(b)
      }

      var ev = parsers.last.inside(finalBatch)
      ev = if(needsOutside) parsers.last.outside(finalBatch, ev) else ev
      val (vimasks, ev3) = parsers.last.computeViterbiMasks(finalBatch, ev)
      parsers.last.extractParses(batch, vimasks, ev3)
    }.toIndexedSeq
  }

  def partitions(sentences: IndexedSeq[IndexedSeq[W]]):IndexedSeq[Float] = synchronized {
    {for {
      batch <- getBatches(sentences).iterator
      evi = parsers.last.inside(batch)
      i <- 0 until batch.numSentences
    } yield {
      batch.insideCharts(i).top(0,batch.insideCharts(i).length, data.head.structure.root)
    }}.toIndexedSeq
  }

  private implicit val queue = if(profile) context.createDefaultProfilingQueue() else context.createDefaultQueue()

  def needsOutside = true
  def isViterbi = data.last.isViterbi

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
    parsers.foreach(_.release())
    queue.release()
  }

  val cellSize = data.map(d => ((d.structure.numNonTerms max d.structure.numTerms)+31)/32 * 32).max

  val (numGPUCells:Int, numGPUChartCells: Int) = {
    val sizeOfFloat = 4
    val fractionOfMemoryToUse = 0.8 // slack!
    val amountOfMemory = ((context.getMaxMemAllocSize min maxAllocSize) * fractionOfMemoryToUse).toInt - data.map(_.ruleScores.length).sum * sizeOfFloat - maxSentencesPerBatch * 3 * 4;
    val maxPossibleNumberOfCells = (amountOfMemory/sizeOfFloat) / (cellSize + 3) toInt // + 1 for offsets
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
  private val devParentPointers, devLeftPointers, devRightPointers = context.createIntBuffer(CLMem.Usage.Input, numGPUCells)
  // transposed
  private val devCharts = new CLMatrix[Float](cellSize, numGPUChartCells)

  // (dest, leftSource, rightSource) (right Source if applicable)
  val pArray, lArray, rArray = new Array[Int](numGPUCells)
  val splitPointOffsets = new Array[Int](numGPUCells+1)

  // other stuff
  private val zmk = ZeroMemoryKernel()
  private val transposeCopy = CLMatrixTransposeCopy()


  private val parsers = data.map(new ActualParser(_))


  private class ActualParser(val data: CLParserData[C, L, W]) {
    import data._
    val devRules = context.createFloatBuffer(CLMem.Usage.Input, FloatBuffer.wrap(ruleScores), false)

    def release() { devRules.release() }

    def _zero: Float = data.semiring.zero
    def _one: Float = data.semiring.one


    def inside(batch: Batch, events: CLEvent*):CLEvent = synchronized {
      allProfilers.foreach(_.clear())
      allProfilers.foreach(_.tick())

      var eZC = zmk.fillMemory(devCharts.data, _zero, events:_*)
      val init = initializeTagScores(batch, eZC)
      hdTransferEvents += init
      memFillEvents += eZC

      var ev = insideTU.doUpdates(batch, 1, init)

      for(span <- 2 to batch.maxLength) {
        println(span)
        ev = insideBinaryPass(batch, span, ev)
        ev = insideNU.doUpdates(batch, span, ev)
      }

      if(profile) {
        queue.finish()
        allProfilers.foreach(_.tock())
        allProfilers.foreach(p => println(s"Inside $p"))
      }

      ev
    }

    def outside(batch: Batch, event: CLEvent):CLEvent = synchronized {
      var ev = event
      allProfilers.foreach(_.clear())
      allProfilers.foreach(_.tick())

      ev = data.util.setRootScores(devCharts, batch.outsideCharts.get.map(_.top.rootIndex).toArray, structure.root, _one, ev)
      ev = outsideNU.doUpdates(batch, batch.maxLength, ev)

      for(span <- (batch.maxLength-1) to 1 by -1) {
        println(span)
        ev = outsideBinaryPass(batch, span, ev)
        if(span == 1) {
          ev = outsideTU.doUpdates(batch, span, ev)
        } else {
          ev = outsideNU.doUpdates(batch, span, ev)
        }

      }

      ev = outsideTT_L.doUpdates(batch, 1, ev)
      ev = outsideNT_R.doUpdates(batch, 1, ev)
      ev = outsideTN_L.doUpdates(batch, 1, ev)
      ev = outsideTT_R.doUpdates(batch, 1, ev)


      if(profile) {
        queue.finish()
        allProfilers.foreach(_.tock())
        Thread.sleep(15)
        allProfilers.foreach(p => println(s"Outside $p"))
      }

      ev
    }

    def computeViterbiMasks(batch: Batch, events: CLEvent*):(CLMatrix[Int], CLEvent) = synchronized {
      computeMasks(batch, -1E-3f, "viterbi", events:_*)
    }


    def initializeTagScores(batch: Batch, events: CLEvent*) = {
      import batch._
      val dm = DenseMatrix.zeros[Float](totalLength, cellSize)
      dm := _zero
      for(i <- 0 until numSentences par) {
        tagScoresFor(batch, dm, i)
        val parent = insideCharts(i).bot.spanRangeSlice(1)
        parent.copyToArray(pArray, _workArrayOffsetsForSpan(1)(i) )
      }
      val ev2 = devParentPointers.write(queue, Pointer.pointerToArray[Integer](pArray), false, events:_*)
      val ev = devParent(0 until totalLength, ::).writeFrom(dm, false, ev2)
      val evr = transposeCopy.permuteTransposeCopyOut(devCharts,  devParentPointers, totalLengthForSpan(1), devParent(0 until totalLength, ::), (ev2 +: ev):_*)

      evr
    }


    private def tagScoresFor(batch: Batch, dm: DenseMatrix[Float], i: Int) {
      import batch._
      val sent = sentences(i)
      val tags = dm(workArrayOffsetsForSpan(i, 1), ::)
      val anch = data.grammar.tagScorer.anchor(sent)
      val lexAnch = data.grammar.lexicon.anchor(sent)
      for(pos <- 0 until sent.length; t <- lexAnch.allowedTags(pos); ref <- data.grammar.refinements.labels.refinementsOf(t)) {
        val index = ref
        val score = anch.scoreTag(pos, data.grammar.refinedGrammar.labelIndex.get(index))
        val gpuIndex = data.structure.labelIndexToTerminal(index)
        tags(pos, gpuIndex) = data.semiring.fromLogSpace(score.toFloat)
      }
    }

    private def computeMasks(batch: Batch, threshold: Float, name: String, events: CLEvent*):(CLMatrix[Int], CLEvent) = synchronized {
      val masks = new CLMatrix[Int](cellSize/32, devParent.size / (cellSize/32), devParent.data.asCLIntBuffer)
      val prof = new CLProfiler(name)
      prof.tick()
      val ev = data.util.getMasks(masks(::, 0 until batch.insideCellOffsets.last),
        devCharts(::, 0 until batch.insideCellOffsets.last),
        devCharts(::, batch.insideCellOffsets.last until batch.insideCellOffsets.last * 2),
        batch.outsideCharts.get.head.bot.globalRowOffset, batch.insideCellOffsets, structure.root, threshold, events:_*)
      prof += ev
      if(profile) {
        queue.finish()
        prof.tock()
        println(s"Masks $prof")
      }
      masks -> ev
    }

    private def insideBinaryPass(batch: Batch, span: Int, events: CLEvent*) = {
      var ev = events
      if(span == 2) {
        ev = Seq(insideTT.doUpdates(batch, span, ev :_*))
      }

      ev = Seq(insideNT.doUpdates(batch, span, ev :_*))
      ev = Seq(insideTN.doUpdates(batch, span, ev :_*))
      ev = Seq(insideNN.doUpdates(batch, span, ev :_*))
      ev.head
    }

    private def outsideBinaryPass(batch: Batch, span: Int, events: CLEvent) = {
      var ev = events


      ev = outsideTN_R.doUpdates(batch, span, ev)
      ev = outsideNT_L.doUpdates(batch, span, ev)
      ev = outsideNN_L.doUpdates(batch, span, ev)
      ev = outsideNN_R.doUpdates(batch, span, ev)

      ev
    }

    private val insideBot = {(b: Batch, s: Int) =>  b.insideCharts(s).bot}
    private val insideTop = {(b: Batch, s: Int) =>  b.insideCharts(s).top}
    private val outsideBot = {(b: Batch, s: Int) =>  b.outsideCharts.get(s).bot}
    private val outsideTop = {(b: Batch, s: Int) =>  b.outsideCharts.get(s).top}

    private def insideTU = new UnaryUpdateManager(this, data.inside.insideTUKernels, insideTop, insideBot)
    private def insideNU = new UnaryUpdateManager(this, data.inside.insideNUKernels, insideTop, insideBot)

    private def insideTT = new BinaryUpdateManager(this, data.inside.insideTTKernels, insideBot, insideBot, insideBot, (b, e, l) => (b+1 to b+1))
    private def insideNT = new BinaryUpdateManager(this, data.inside.insideNTKernels, insideBot, insideTop, insideBot, (b, e, l) => (e-1 to e-1))
    private def insideTN = new BinaryUpdateManager(this, data.inside.insideTNKernels, insideBot, insideBot, insideTop, (b, e, l) => (b+1 to b+1))
    private def insideNN = new BinaryUpdateManager(this, data.inside.insideNNKernels, insideBot, insideTop, insideTop, (b, e, l) => (b+1 to e-1))

    // here, "parentChart" is actually the left child, left is the parent, right is the right completion
    private def outsideTT_L = new BinaryUpdateManager(this, data.outside.get.outside_L_TTKernels, outsideBot, outsideBot, insideBot, (b, e, l) => (e+1 to e+1))
    private def outsideNT_L = new BinaryUpdateManager(this, data.outside.get.outside_L_NTKernels, outsideTop, outsideBot, insideBot, (b, e, l) => (e+1 to e+1))
    private def outsideTN_L = new BinaryUpdateManager(this, data.outside.get.outside_L_TNKernels, outsideBot, outsideBot, insideTop, (b, e, l) => (e+1 to l))
    private def outsideNN_L = new BinaryUpdateManager(this, data.outside.get.outside_L_NNKernels, outsideTop, outsideBot, insideTop, (b, e, l) => (e+1 to l))

    // here, "parentChart" is actually the right child, right is the parent, left is the left completion
    private def outsideTT_R = new BinaryUpdateManager(this, data.outside.get.outside_R_TTKernels, outsideBot, insideBot, outsideBot, (b, e, l) => (b-1 to b-1))
    private def outsideNT_R = new BinaryUpdateManager(this, data.outside.get.outside_R_NTKernels, outsideBot, insideTop, outsideBot, (b, e, l) => (0 to b-1))
    private def outsideTN_R = new BinaryUpdateManager(this, data.outside.get.outside_R_TNKernels, outsideTop, insideBot, outsideBot, (b, e, l) => (b-1 to b-1))
    private def outsideNN_R = new BinaryUpdateManager(this, data.outside.get.outside_R_NNKernels, outsideTop, insideTop, outsideBot, (b, e, l) => (0 to b-1))

    private def outsideTU = new UnaryUpdateManager(this, data.outside.get.outsideTUKernels, outsideBot, outsideTop)
    private def outsideNU = new UnaryUpdateManager(this, data.outside.get.outsideNUKernels, outsideBot, outsideTop)


    def addMasksToBatches(batch: Batch, ev: CLEvent*): Batch = {
      val insideEvents = inside(batch, ev:_*)
      val outsideEvents = outside(batch, insideEvents)
      val (masks, ev2) = computeMasks(batch, -7, "masks", outsideEvents)
      val denseMasks = masks.toDense
      ev2.waitFor()
      batch.copy(allowedSpan = (sent, begin, end) => denseMasks(::, batch.insideCellOffsets(sent) + ChartHalf.chartIndex(begin, end, batch.sentences(sent).length)).any)
    }

    def extractParses(batch: Batch, masks: CLMatrix[Int], events: CLEvent*) = {
      events.foreach(_.waitFor())
      val in = if(profile) System.currentTimeMillis() else 0L
      val dmMasks:DenseMatrix[Int] = masks.toDense
      val trees = for(s <- 0 until batch.numSentences par) yield {
        import batch.insideCellOffsets
        val length = batch.sentences(s).length
        val numCells = (insideCellOffsets(s+1)-insideCellOffsets(s))/2
        assert(numCells == TriangularArray.arraySize(length))
        val botBegin = insideCellOffsets(s)
        val botEnd = botBegin + numCells
        val topBegin = botEnd
        val topEnd = topBegin + numCells
        val bot = dmMasks(::, botBegin until botEnd)
        val top = dmMasks(::, topBegin until topEnd)

        def recTop(begin: Int, end: Int):BinarizedTree[L] = {
          val column:DenseVector[Int] = top(::, ChartHalf.chartIndex(begin, end, length))
          val x = firstSetBit(column:DenseVector[Int])
          if(x == -1) {
            assert(begin == end - 1, column.toString + " " + x + " " + (begin,end) + " " + s + " " + batch.numSentences)
            recBot(begin, end)
          } else {
            assert(column.valuesIterator.exists(_ != 0), (begin, end))
            val label = structure.nontermIndex.get(x)
            val t = recBot(begin, end)
            new UnaryTree[L](label, t, IndexedSeq.empty, Span(begin, end))
          }
        }

        def recBot(begin: Int, end: Int):BinarizedTree[L] = {
          val column:DenseVector[Int] = bot(::, ChartHalf.chartIndex(begin, end, length))
          val x = firstSetBit(column:DenseVector[Int])
          if(begin == end-1) {
            val label = structure.termIndex.get(x)
            NullaryTree(label, Span(begin, end))
          } else {
            val label = structure.nontermIndex.get(x)
            for(split <- (begin+1) until end) {
              val left = (if(begin == split - 1) bot else top)(::, ChartHalf.chartIndex(begin, split, length))
              val right = (if(end == split + 1) bot else top)(::, ChartHalf.chartIndex(split, end, length))
              if(left.any && right.any) {
                return BinaryTree[L](label, recTop(begin, split), recTop(split, end), Span(begin, end))
              }
            }
            error("nothing here!" + " "+ (begin, end) +
              {for(split <- (begin+1) until end) yield {
                val left = (if(begin == split - 1) bot else top)(::, ChartHalf.chartIndex(begin, split, length))
                val right = (if(end == split + 1) bot else top)(::, ChartHalf.chartIndex(split, end, length))
                (ChartHalf.chartIndex(begin, split, length), ChartHalf.chartIndex(split, end, length), split, left,right)

              }})
          }
        }

        recTop(0, length)

      }
      val out = if(profile) System.currentTimeMillis() else 0L
      if(profile) {
        println(s"Parse extraction took:  ${(out - in)/1000.0}s")

      }
      trees
    }

  }

  private case class Batch(lengthTotals: Array[Int],
                           insideCellOffsets: Array[Int],
                           sentences: IndexedSeq[IndexedSeq[W]],
                           allowedSpan: (Int, Int, Int)=>Boolean = (_, _, _) => true) {
    def totalLength = lengthTotals.last
    def numSentences = sentences.length
    val maxLength = sentences.map(_.length).max
    if(needsOutside)
      assert(insideCellOffsets.last * 2 <= devCharts.cols)
    else
      assert(insideCellOffsets.last <= devCharts.cols)


    val _workArrayOffsetsForSpan = Array.tabulate(maxLength+1)(span => sentences.scanLeft(0)((off, sent) => off + math.max(0,sent.length-span+1)).toArray)
    def workArrayOffsetsForSpan(sent: Int, span: Int) = Range(_workArrayOffsetsForSpan(span)(sent), _workArrayOffsetsForSpan(span)(sent+1))

    def totalLengthForSpan(span: Int) = _workArrayOffsetsForSpan(span).last

    lazy val insideCharts = for(i <- 0 until numSentences) yield {
      val numCells = (insideCellOffsets(i+1)-insideCellOffsets(i))/2
      assert(numCells == TriangularArray.arraySize(sentences(i).length))
      val chart = new ParseChart(sentences(i).length, devCharts(::, insideCellOffsets(i) until (insideCellOffsets(i) + numCells)), devCharts(::, insideCellOffsets(i) + numCells until insideCellOffsets(i+1)))
      chart
    }

    lazy val outsideCharts = if(!needsOutside) None else Some{
      for(i <- 0 until numSentences) yield {

        val numCells = (insideCellOffsets(i+1)-insideCellOffsets(i))/2
        assert(numCells == TriangularArray.arraySize(sentences(i).length))
        val botBegin = insideCellOffsets.last + insideCellOffsets(i)
        val botEnd = botBegin + numCells
        val topBegin = botEnd
        val topEnd = topBegin + numCells
        assert(topEnd < devCharts.cols)
        val chart = new ParseChart(sentences(i).length, devCharts(::, botBegin until botEnd), devCharts(::, topBegin until topEnd))
        chart
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
      currentCellTotal += TriangularArray.arraySize(s.length) * 2
      if(needsOutside)
        currentCellTotal += TriangularArray.arraySize(s.length) * 2
      if(currentLengthTotal > numGPUCells || currentCellTotal > numGPUChartCells) {
        assert(current.nonEmpty)
        result += createBatch(current)
        currentLengthTotal = s.length
        currentCellTotal = TriangularArray.arraySize(s.length) * 2
        if(needsOutside)
          currentCellTotal += TriangularArray.arraySize(s.length) * 2
        current = ArrayBuffer()
      }
      current += s
    }

    if(current.nonEmpty) result += createBatch(current)
    result
  }

  private def createBatch(sentences: IndexedSeq[IndexedSeq[W]]): Batch = {
    val lengthTotals = sentences.scanLeft(0)((acc, sent) => acc + sent.length)
    val cellTotals = sentences.scanLeft(0)((acc, sent) => acc + TriangularArray.arraySize(sent.length) * 2)
    if(needsOutside)
      assert(cellTotals.last * 2 <= devCharts.cols, cellTotals.last * 2 + " " +  devCharts.cols)
    Batch(lengthTotals.toArray, cellTotals.toArray, sentences)
  }

  private class BinaryUpdateManager(parser: ActualParser,
                                    kernels: IndexedSeq[CLKernel],
                                    parentChart: (Batch,Int)=>ChartHalf,
                                    leftChart: (Batch,Int)=>ChartHalf,
                                    rightChart: (Batch,Int)=>ChartHalf,
                                    ranger: (Int, Int, Int)=>Range) {

    var parentOffset = 0 // number of unique parent spans used so far
    var offset = 0 // number of cells used so far.

    // TODO: ugh, state
    var lastParent = -1

    var ev = Seq.empty[CLEvent]

    private def enqueue(span: Int, parent: Int, left: Int, right: Int) {
      if(parentOffset == 0 || lastParent != parent) {
        splitPointOffsets(parentOffset) = offset
        pArray(parentOffset) = parent
        parentOffset += 1
      }
      lastParent = parent
      lArray(offset) = left
      rArray(offset) = right
      offset += 1
      if(offset >= numGPUCells)  {
        println(s"flush!")
        flushQueue(span)
      }
    }

    private def flushQueue(span: Int) {
      if(offset != 0) {
        splitPointOffsets(parentOffset) = offset
        val wevl = devLeftPointers.write(queue, Pointer.pointerToArray[Integer](lArray), false, ev:_*)
        val wevr = devRightPointers.write(queue, Pointer.pointerToArray[Integer](rArray), false, ev:_*)
        val wl = transposeCopy.permuteTransposeCopy(devLeft(0 until offset, ::), devCharts, devLeftPointers, offset, wevl)
        val wr = transposeCopy.permuteTransposeCopy(devRight(0 until offset, ::), devCharts, devRightPointers, offset, wevr)

        transferEvents += wl
        transferEvents += wr
        transferEvents += wevl
        transferEvents += wevr
        val zz = zmk.shapedFill(devParent(0 until offset, ::), parser._zero, ev:_*)
        memFillEvents += zz
        ev = kernels.map{ kernel =>
          kernel.setArgs(devParent.data.safeBuffer, devLeft.data.safeBuffer, devRight.data.safeBuffer, parser.devRules, Integer.valueOf(numGPUCells), Integer.valueOf(offset))
          kernel.enqueueNDRange(queue, Array(offset), wl, wr, zz)
        }


        binaryEvents ++= ev
        val sumEv = parser.data.util.sumSplitPoints(devParent,
          devCharts,
          java.util.Arrays.copyOf(pArray, parentOffset),
          java.util.Arrays.copyOf(splitPointOffsets, parentOffset + 1),
          32 / span max 1, ev:_*)


        sumEvents += sumEv
        ev = IndexedSeq(sumEv)
        queue.finish()
        offset = 0
        parentOffset = 0
      }
    }



    def doUpdates(batch: Batch, span: Int, events: CLEvent*) = {

      ev = events
      parentOffset = 0
      lastParent = -1

      for {
        sent <- 0 until batch.numSentences
        start <- 0 to batch.sentences(sent).length - span
        if batch.allowedSpan(sent, start, start + span)
        split <- ranger(start, start + span, batch.sentences(sent).length)
        if split >= 0 && split <= batch.sentences(sent).length
      } {
        val end = start + span
        val parentTi = parentChart(batch, sent).treeIndex(start,end)
        val leftChildAllowed = if(split < start) batch.allowedSpan(sent,split, start) else batch.allowedSpan(sent,start, split)
        val rightChildAllowed = if(split < end) batch.allowedSpan(sent,split,end) else batch.allowedSpan(sent,end, split)

        if(leftChildAllowed && rightChildAllowed) {
          val leftChild = if(split < start) leftChart(batch, sent).treeIndex(split,start) else leftChart(batch, sent).treeIndex(start, split)
          val rightChild = if(split < end) rightChart(batch, sent).treeIndex(split,end) else rightChart(batch, sent).treeIndex(end, split)
          enqueue(span, parentTi, leftChild, rightChild)
        }

      }

      if(offset > 0) {
        flushQueue(span)
      }

      assert(ev.length == 1)
      ev.head
    }

  }

  private class UnaryUpdateManager(parser: ActualParser,
                                    kernels: IndexedSeq[CLKernel],
                                    parentChart: (Batch,Int)=>ChartHalf,
                                    childChart: (Batch,Int)=>ChartHalf) {

    var offset = 0 // number of cells used so far.

    var ev = Seq.empty[CLEvent]

    private def enqueue(span: Int, parent: Int, left: Int) {
      lArray(offset) = left
      pArray(offset) = parent
      offset += 1
      if(offset >= numGPUCells)  {
        println(s"flush!")
        flushQueue(span)
      }
    }

    private def flushQueue(span: Int) = {
      if(offset != 0) {
        val zz = zmk.shapedFill(devParent(0 until offset, ::), parser._zero, ev:_*)
        memFillEvents += zz

        val wevl = devLeftPointers.write(queue, Pointer.pointerToArray[Integer](lArray), false, ev:_*)

        val wl = transposeCopy.permuteTransposeCopy(devLeft(0 until offset, ::), devCharts, devLeftPointers, offset, wevl)
        transferEvents += wl
        transferEvents += wevl

        val endEvents = kernels.map{ kernel  =>
          kernel.setArgs(devParent.data.safeBuffer, devLeft.data.safeBuffer, parser.devRules, Integer.valueOf(numGPUCells), Integer.valueOf(offset))
          kernel.enqueueNDRange(queue, Array(offset), wl, zz)
        }

        unaryEvents ++= endEvents

        val ev2 = devParentPointers.write(queue, Pointer.pointerToArray[Integer](pArray), false, endEvents:_*)
        val _ev = transposeCopy.permuteTransposeCopyOut(devCharts, devParentPointers, offset, devParent(0 until offset, ::), (ev2 +: endEvents):_*)

        queue.finish()
        sumToChartsEvents += _ev
        offset = 0
        this.ev = IndexedSeq(_ev)
      }
    }

    def doUpdates(batch: Batch, span: Int, events: CLEvent*) = {
      ev = events

      for {
        sent <- 0 until batch.numSentences
        start <- 0 to batch.sentences(sent).length - span
        if batch.allowedSpan(sent, start, start + span)
      } {
        val end = start + span
        val parentTi = parentChart(batch, sent).treeIndex(start,end)
        val child = childChart(batch, sent).treeIndex(start,end)

        enqueue(span, parentTi, child)
      }

      if(offset > 0) {
        flushQueue(span)
      }

      assert(ev.length == 1)
      ev.head
    }

  }
}

object CLParser extends Logging {

  case class Params(annotator: TreeAnnotator[AnnotatedLabel, String, AnnotatedLabel] = Xbarize(),
                    useGPU: Boolean = true, profile: Boolean = false,
                    numToParse: Int = 1000, codeCache: File = new File("grammar.grz"),
                    jvmParse: Boolean = false, parseTwice: Boolean = false,
                    textGrammarPrefix: String = null, checkPartitions: Boolean = false)

  def main(args: Array[String]) = {
    import ParserParams.JointParams

    val params = CommandLineParser.readIn[JointParams[Params]](args)
    val myParams:Params = params.trainer
    import myParams._
    println("Training Parser...")
    println(params)
    val transformed = params.treebank.trainTrees.par.map { ti => annotator(ti) }.seq.toIndexedSeq
    val grammar: SimpleRefinedGrammar[AnnotatedLabel, AnnotatedLabel, String] = if(textGrammarPrefix == null) {
      GenerativeParser.extractGrammar(AnnotatedLabel.TOP, transformed)
    } else {
      SimpleRefinedGrammar.parseBerkeleyText(textGrammarPrefix, -10)
    }

    implicit val context = if(useGPU) {
      val gpu = JavaCL.listPlatforms.flatMap(_.listGPUDevices(true)).head
      JavaCL.createContext(new java.util.HashMap(), gpu)
    } else {
//      val gpu = JavaCL.listPlatforms.flatMap(_.listGPUDevices(true)).last
//      JavaCL.createContext(new java.util.HashMap(), gpu)
      val cpuPlatform:CLPlatform = JavaCL.listPlatforms().filter(_.listCPUDevices(true).nonEmpty).head
      cpuPlatform.createContext(new java.util.HashMap(), cpuPlatform.listCPUDevices(true):_*)
    }
    println(context)

    var parserData:CLParserData[AnnotatedLabel, AnnotatedLabel, String] = if(codeCache != null && codeCache.exists()) {
      CLParserData.read(new ZipFile(codeCache))
    } else {
      null
    }

    if(parserData == null || parserData.grammar.signature != grammar.signature) {
      println("Regenerating parser data")
      parserData = CLParserData.make(grammar)
      if(codeCache != null) {
        parserData.write(new BufferedOutputStream(new FileOutputStream(codeCache)))
      }
    }

    val kern = fromParserData[AnnotatedLabel, AnnotatedLabel, String](parserData, profile)
    val train = transformed.slice(1, 1+numToParse).map(_.words)


    if(checkPartitions) {
      val partsX = kern.partitions(train)
      println(partsX)
      val parser = SimpleChartParser(AugmentedGrammar.fromRefined(grammar), if(kern.isViterbi) new ViterbiDecoder[AnnotatedLabel, String] else new MaxConstituentDecoder[AnnotatedLabel, String])
      val parts2 = train.par.map(parser.charts(_).logPartition)
      println(parts2)
      println("max difference: " + (DenseVector(partsX.map(_.toDouble):_*) - DenseVector(parts2.seq:_*)).norm(Double.PositiveInfinity))
      System.exit(0)
    }
    var timeIn = System.currentTimeMillis()
    val parts = kern.parse(train)
    var timeOut = System.currentTimeMillis()
    println(parts zip train map {case (k,v) => k render v})
    println(s"CL Parsing took: ${(timeOut-timeIn)/1000.0}")
    if(parseTwice) {
      timeIn = System.currentTimeMillis()
      val parts2 = kern.parse(train)
      timeOut = System.currentTimeMillis()
      println(parts2 zip train map {case (k,v) => k render v})
      println(s"CL Parsing took x2: ${(timeOut-timeIn)/1000.0}")
    }
    if(jvmParse) {
      val parser = SimpleChartParser(AugmentedGrammar.fromRefined(grammar), if(kern.isViterbi) new ViterbiDecoder[AnnotatedLabel, String] else new MaxConstituentDecoder[AnnotatedLabel, String])
      timeIn = System.currentTimeMillis()
      val margs = train.map { w =>
        val m = parser.apply(w)
        /*
        printChart(m, true, false)
        printChart(m, false, false)
        printChart(m, true, true)
        printChart(m, false, true)
        */
        m -> w
      }
      timeOut = System.currentTimeMillis()
      println(s"Scala Parsing took: ${(timeOut-timeIn)/1000.0}")
      println(margs.map{case (m ,w) => m render w})
    }

    kern.release()
    context.release()

  }

  def fromSimpleGrammar[L, L2, W](grammar: SimpleRefinedGrammar[L, L2, W], profile: Boolean = false)(implicit context: CLContext) = {
    val data: CLParserData[L, L2, W] = CLParserData.make(grammar)
    fromParserData(data, profile)
  }


  def fromParserData[L, L2, W](data: CLParserData[L, L2, W], profile: Boolean)(implicit context: CLContext): CLParser[L, L2, W] = {
    val kern = new CLParser[L, L2, W](IndexedSeq(data), profile = profile)
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
                                 semiring: RuleSemiring,
                                 ruleScores: Array[Float],
                                 inside: CLInsideKernels,
                                 outside: Option[CLOutsideKernels],
                                 util: CLParserUtils,
                                 isViterbi: Boolean) {


  def write(out: OutputStream) {
    val zout = new ZipOutputStream(out)
    ZipUtil.serializedEntry(zout, "grammar", grammar)
    ZipUtil.serializedEntry(zout, "structure", structure)
    ZipUtil.serializedEntry(zout, "semiring", semiring)
    inside.write(zout)
    outside.foreach(_.write(zout))
    util.write(zout)
    ZipUtil.serializedEntry(zout, "scores", ruleScores)
    zout.close()
  }
}

object CLParserData {
  def make[C, L, W](grammar: SimpleRefinedGrammar[C, L, W])(implicit context: CLContext) = {
    implicit val viterbi = ViterbiRuleSemiring
    val structure = new RuleStructure(grammar.refinements, grammar.refinedGrammar)
    val inside = CLInsideKernels.make(structure)
    val outside =  Some(CLOutsideKernels.make(structure)) //if(!gen.isViterbi) Some(CLOutsideKernels.make(gen)) else None
    val util = CLParserUtils.make(structure)

    val ruleScores: Array[Float] = Array.tabulate(grammar.refinedGrammar.index.size){r =>
      val score = grammar.ruleScoreArray(grammar.refinements.rules.project(r))(grammar.refinements.rules.localize(r))
      viterbi.fromLogSpace(score.toFloat)
    }
    new CLParserData(grammar, structure, viterbi, ruleScores, inside, outside, util, viterbi.plusIsIdempotent)
  }

  def read[C, L, W](file: ZipFile)(implicit context: CLContext) = {
    val gr = ZipUtil.deserializeEntry[SimpleRefinedGrammar[C, L, W]](file.getInputStream(file.getEntry("grammar")))
    val structure = ZipUtil.deserializeEntry[RuleStructure[C, L]](file.getInputStream(file.getEntry("structure")))
    val semiring = ZipUtil.deserializeEntry[RuleSemiring](file.getInputStream(file.getEntry("semiring")))
    val inside = CLInsideKernels.read(file)
    val outside = CLOutsideKernels.tryRead(file)
    val util = CLParserUtils.read(file)
    val scores = ZipUtil.deserializeEntry[Array[Float]](file.getInputStream(file.getEntry("scores")))

    CLParserData(gr, structure, semiring, scores, inside, outside, util, semiring.plusIsIdempotent)
  }
}


