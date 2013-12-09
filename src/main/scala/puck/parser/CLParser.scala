package puck
package parser

import gen._
import com.nativelibs4java.opencl._
import com.typesafe.scalalogging.slf4j.Logging
import puck.util.{ZipUtil, BitHacks, ZeroMemoryKernel, CLProfiler}
import puck.linalg.CLMatrix
import java.nio.FloatBuffer
import puck.linalg.kernels.{CLMatrixSliceCopy, CLMatrixTransposeCopy}
import breeze.collection.mutable.TriangularArray
import breeze.linalg.{Counter2, DenseVector, DenseMatrix}
import epic.trees.annotations.{Xbarize, TreeAnnotator}
import epic.parser._
import breeze.config.CommandLineParser
import epic.trees._
import java.io._
import java.util.zip.{ZipFile, ZipOutputStream}
import scala.collection.mutable.ArrayBuffer
import BitHacks._
import epic.trees.UnaryTree
import scala.Some
import epic.parser.ViterbiDecoder
import epic.trees.NullaryTree
import epic.trees.BinaryTree
import epic.trees.annotations.Xbarize
import epic.trees.Span
import epic.parser.SimpleRefinedGrammar.CloseUnaries
import epic.parser.projections.{ParserChartConstraintsFactory, ConstraintCoreGrammarAdaptor}
import epic.lexicon.SimpleLexicon
import scala.collection.parallel.immutable.ParSeq

/**
 * TODO
 *
 * @author dlwh
 **/
class CLParser[C, L, W](data: IndexedSeq[CLParserData[C, L, W]],
                        maxAllocSize: Long = 1<<30, // 1 gig
                        maxSentencesPerBatch: Long = 400,
                        profile: Boolean = true)(implicit val context: CLContext) extends Logging {

  def parse(sentences: IndexedSeq[IndexedSeq[W]]):IndexedSeq[BinarizedTree[C]] = synchronized {
    getBatches(sentences).iterator.flatMap{ batch =>
      var ev =  maskParent.assignAsync(-1)
      ev = maskCharts.assignAsync(-1, ev)
      val finalBatch = parsers.take(data.length - 1).foldLeft(batch){(b, parser) =>
        parser.addMasksToBatches(b, ev)
      }

      ev = parsers.last.inside(finalBatch, ev)
      ev = parsers.last.outside(finalBatch, ev)
      val ev3 = parsers.last.computeViterbiMasks(finalBatch, ev)
      parsers.last.extractParses(batch, maskCharts, ev3)
    }.toIndexedSeq
  }

  def partitions(sentences: IndexedSeq[IndexedSeq[W]]):IndexedSeq[Float] = synchronized {
    getBatches(sentences).iterator.flatMap{ batch =>
      var ev =  maskParent.assignAsync(-1)
      ev = maskCharts.assignAsync(-1, ev)
      // allow all spans
      val finalBatch = parsers.take(data.length - 1).foldLeft(batch){(b, parser) =>
        parser.addMasksToBatches(b, ev)
      }

      ev = parsers.last.inside(finalBatch, ev)
      ev = parsers.last.outside(finalBatch, ev)
      ev.waitFor()
      for ( i <- 0 until batch.numSentences) yield {
        (batch.insideCharts(i).top(0, batch.sentences(i).length, data.last.structure.root))
//        (batch.insideCharts(i).bot(2, 3)
//        + batch.outsideCharts(i).bot(2,3)).max
      }
    }.toIndexedSeq
  }

  private implicit val queue = if (profile) context.createDefaultProfilingQueue() else context.createDefaultQueue()

  def isViterbi = data.last.isViterbi

  private val initMemFillEvents  = new CLProfiler("initMemfill")
  private val memFillEvents  = new CLProfiler("memfill")
  private val hdTransferEvents  = new CLProfiler("Host2Dev Transfer")
  private val transferEvents  = new CLProfiler("Transfer")
  private val binaryEvents  = new CLProfiler("Binary")
  private val unaryEvents  = new CLProfiler("Unary")
  private val sumToChartsEvents  = new CLProfiler("SumToCharts")
  private val sumEvents  = new CLProfiler("Sum")
  private val masksEvents  = new CLProfiler("Masks")
  val allProfilers =  IndexedSeq(transferEvents, binaryEvents, unaryEvents, sumToChartsEvents, sumEvents, initMemFillEvents, memFillEvents, hdTransferEvents, masksEvents)

  def release() {
    devParent.release()
    devInside.release()
    devOutside.release()
    devLeft.release()
    devRight.release()
    queue.release()
    devParentPtrs.release()
    devLeftPtrs.release()
    devRightPtrs.release()

    maskParent.release()
    maskCharts.release()
    maskLeft.release()
    maskRight.release()
  }

  // size in floats, just the number of symbols
  val cellSize:Int = roundUpToMultipleOf(data.map(_.numSyms).max, 32)
  val maskSize:Int = {
    assert(data.forall(_.maskSize == data.head.maskSize))
    data.head.maskSize
  }

  val (numWorkCells:Int, numChartCells: Int) = {
    val sizeOfFloat = 4
    val fractionOfMemoryToUse = 0.7 // slack!
    val amountOfMemory = ((context.getDevices.head.getGlobalMemSize min maxAllocSize) * fractionOfMemoryToUse).toInt  - maxSentencesPerBatch * 3 * 4;
    val maxPossibleNumberOfCells = ((amountOfMemory / sizeOfFloat) / (cellSize + 4 + maskSize)).toInt // + 4 for each kind of offset
    // We want numGPUCells and numGPUChartCells to be divisible by 16, so that we get aligned strided access:
    //       On devices of compute capability 1.0 or 1.1, the k-th thread in a half warp must access the
    //       k-th word in a segment aligned to 16 times the size of the elements being accessed; however,
    //       not all threads need to participate... If sequential threads in a half warp access memory that is
    //       sequential but not aligned with the segments, then a separate transaction results for each element
    //       requested on a device with compute capability 1.1 or lower.
    val numberOfUnitsOf16 = maxPossibleNumberOfCells / 16
    // average sentence length of sentence, let's say n.
    // for the gpu charts, we'll need (n choose 2) * 2 * 2 =
    // for the "P/L/R" parts, the maximum number of relaxations (P = L * R * rules) for a fixed span
    // in a fixed sentence is (n/2)^2= n^2/4.
    // Take n = 32, then we want our P/L/R arrays to be of the ratio (3 * 256):992 \approx 3/4 (3/4 exaclty if we exclude the - n term)
    // doesn't quite work the way we want (outside), so we'll bump the number to 4/5
    val relativeSizeOfChartsToP = 4
    val baseSize = numberOfUnitsOf16 / (3 + relativeSizeOfChartsToP)
    val extra = numberOfUnitsOf16 % (3 + relativeSizeOfChartsToP)
    val plrSize = baseSize
    // TODO, can probably do a better job of these calculations?
    (plrSize * 16, (baseSize * relativeSizeOfChartsToP + extra) * 16)
  }

  println(numWorkCells, numChartCells, (3 * numWorkCells + numChartCells) * cellSize * 4, context.getMaxMemAllocSize)


  // On the Device side we have 4 Matrices:
  // One is where we calculate P = L * R * rules, for fixed spans and split points (the "bot")
  // One is the L part of the above
  // Another is the R part.
  // finally, we have the array of parse charts
  private val devParent, devLeft, devRight = new CLMatrix[Float](numWorkCells, cellSize)
  private val maskParent, maskLeft, maskRight = new CLMatrix[Int](maskSize, numWorkCells)
  private val devParentPtrs, devLeftPtrs, devRightPtrs = context.createIntBuffer(CLMem.Usage.Input, numWorkCells)
  private val devSplitPointOffsets = context.createIntBuffer(CLMem.Usage.Input, numWorkCells + 1)
  // transposed
  private val devInside, devOutside = new CLMatrix[Float](cellSize, numChartCells/2)
  private val maskCharts = new CLMatrix[Int](maskSize, numChartCells)

  // (dest, leftSource, rightSource) (right Source if binary rules)
  val pArray, lArray, rArray = new Array[Int](numWorkCells)
  val splitPointOffsets = new Array[Int](numWorkCells+1)

  // other stuff
  private val zmk = ZeroMemoryKernel()
  private val transposeCopy = CLMatrixTransposeCopy()
  private val sliceCopy = CLMatrixSliceCopy()

  private val parsers = data.map(new ActualParser(_))
  var pruned = 0
  var total = 0

  private class ActualParser(val data: CLParserData[C, L, W]) {
    import data._


    def _zero: Float = data.semiring.zero

    def inside(batch: Batch, events: CLEvent*):CLEvent = synchronized {
      allProfilers.foreach(_.clear())
      allProfilers.foreach(_.tick())
      pruned = 0
      total = 0

      val evZeroCharts = zmk.fillMemory(devInside.data, _zero, events:_*) profileIn initMemFillEvents
      val evZeroOutside = zmk.fillMemory(devOutside.data, _zero, events:_*) profileIn initMemFillEvents
      val init = initializeTagScores(batch, evZeroCharts, evZeroOutside)

//      maskCharts.assignAsync(-1, evZeroCharts).waitFor
      var ev = insideTU.doUpdates(batch, 1, init)

      for (span <- 2 to batch.maxLength) {
        print(s"$span ")
        ev = insideBinaryPass(batch, span, ev)
        ev = insideNU.doUpdates(batch, span, ev)
      }

      if (profile) {
        queue.finish()
        allProfilers.foreach(_.tock())
        allProfilers.foreach(p => println(s"Inside $p"))
        println(f"Time accounted for: ${allProfilers.map(_.processingTime).sum}%.3f")
      }
      println(s"Pruned $pruned/$total")

      ev
    }

    def outside(batch: Batch, event: CLEvent):CLEvent = synchronized {
      var ev = event
      allProfilers.foreach(_.clear())
      allProfilers.foreach(_.tick())
      pruned  = 0
      total = 0

      ev = devParentPtrs.writeArray(queue, batch.outsideCharts.map(_.top.rootIndex).toArray, batch.outsideCharts.length, ev) profileIn hdTransferEvents
      ev = data.util.setRootScores(devOutside, devParentPtrs, batch.numSentences, structure.root, data.semiring.one, ev) profileIn memFillEvents

      ev = outsideNU.doUpdates(batch, batch.maxLength, ev)

      for (span <- (batch.maxLength - 1) to 1 by -1) {
        print(s"$span ")
        ev = outsideBinaryPass(batch, span, ev)
        if (span == 1) {
          ev = outsideTU.doUpdates(batch, span, ev)
        } else {
          ev = outsideNU.doUpdates(batch, span, ev)
        }

      }

      ev = outsideTT_L.doUpdates(batch, 1, ev)
      ev = outsideNT_R.doUpdates(batch, 1, ev)
      ev = outsideTN_L.doUpdates(batch, 1, ev)
      ev = outsideTT_R.doUpdates(batch, 1, ev)

      if (profile) {
        queue.finish()
        allProfilers.foreach(_.tock())
        Thread.sleep(15)
        allProfilers.foreach(p => println(s"Outside $p"))
        println(f"Time accounted for: ${allProfilers.map(_.processingTime).sum}%.3f")
      }
      println(s"Pruned $pruned/$total")

      ev
    }

    def computeViterbiMasks(batch: Batch, events: CLEvent*):CLEvent = synchronized {
      computeMasks(batch, -4E-3f, events:_*)
    }

    def initializeTagScores(batch: Batch, events: CLEvent*) = {
      import batch._
      val dm = DenseMatrix.zeros[Float](totalLength, CLParser.this.cellSize)
      dm := _zero
      for (i <- 0 until numSentences par) {
        tagScoresFor(batch, dm, i)
        val parent = insideCharts(i).bot.spanRangeSlice(1)
        parent.copyToArray(pArray, _workArrayOffsetsForSpan(1)(i) )
      }

      val ev2 = devParentPtrs.writeArray(queue, pArray, totalLengthForSpan(1), events:_*) profileIn hdTransferEvents
      val ev = devParent(0 until totalLength, ::).writeFrom(dm, false, ev2) map (_ profileIn  hdTransferEvents)
      transposeCopy.permuteTransposeCopyOut(devInside,  devParentPtrs, totalLengthForSpan(1), devParent(0 until totalLength, ::), (ev2 +: ev):_*) profileIn sumToChartsEvents
    }


    private def tagScoresFor(batch: Batch, dm: DenseMatrix[Float], i: Int) {
      import batch._
      val sent = sentences(i)
      val tags = dm(workArrayOffsetsForSpan(i, 1), ::)
      val anch = data.grammar.tagScorer.anchor(sent)
      val lexAnch = data.grammar.lexicon.anchor(sent)
      for (pos <- 0 until sent.length; t <- lexAnch.allowedTags(pos); ref <- data.grammar.refinements.labels.refinementsOf(t)) {
        val index = ref
        val score = anch.scoreTag(pos, data.grammar.refinedGrammar.labelIndex.get(index))
        val gpuIndex = data.structure.labelIndexToTerminal(index)
//        if(pos == 0) println(pos,t,ref,data.grammar.refinements.labels.fineIndex.get(ref), gpuIndex,score)
        tags(pos, gpuIndex) = data.semiring.fromLogSpace(score.toFloat)
      }
    }

    private def markTerminalsInMasks(batch: Batch, events: CLEvent*) = {
      import batch._
      val set0 = maskCharts.assignAsync(0, events:_*)
      set0.waitFor()
      for (i <- 0 until numSentences par) {
        val parent = insideCharts(i).bot.spanRangeSlice(1)
        parent.copyToArray(pArray, _workArrayOffsetsForSpan(1)(i) )
      }

      val ev2 = devParentPtrs.writeArray(queue, pArray, totalLengthForSpan(1), events:_*) profileIn hdTransferEvents
      val ev = maskParent(::, 0 until totalLength).assignAsync(-1, set0) profileIn memFillEvents
      sliceCopy.sliceCopyOut(maskCharts.asInstanceOf[CLMatrix[Float]],  devParentPtrs, totalLengthForSpan(1), maskParent(::, 0 until totalLength).asInstanceOf[CLMatrix[Float]], ev, ev2, set0) profileIn sumToChartsEvents
    }

    private def computeMasks(batch: Batch, threshold: Float, events: CLEvent*):CLEvent = synchronized {
      if(profile) {
        allProfilers.foreach(_.clear())
        allProfilers.foreach(_.tick())
      }

      val ev =  markTerminalsInMasks(batch, events:_*)

      val evr = data.masks.getMasks(maskCharts(::, 0 until batch.cellOffsets.last),
        devInside(::, 0 until batch.cellOffsets.last),
        devOutside(::, 0 until batch.cellOffsets.last),
        0, batch.cellOffsets, structure.root, threshold, ev) profileIn masksEvents
      if (profile) {
        queue.finish()
        allProfilers.foreach(_.tock())
        allProfilers.foreach(p => println(s"Masks $p"))
      }

      evr
    }

    private def insideBinaryPass(batch: Batch, span: Int, events: CLEvent*) = {
      var ev = events
      if (span == 2) {
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
    private val outsideBot = {(b: Batch, s: Int) =>  b.outsideCharts(s).bot}
    private val outsideTop = {(b: Batch, s: Int) =>  b.outsideCharts(s).top}

    private def insideTU = new UnaryUpdateManager(this, data.inside.insideTUKernels, devInside, insideTop, insideBot)
    private def insideNU = new UnaryUpdateManager(this, data.inside.insideNUKernels, devInside, insideTop, insideBot)

    private def insideTT = new BinaryUpdateManager(this, data.inside.insideTTKernels, devInside, devInside, devInside, insideBot, insideBot, insideBot, (b, e, l) => (b+1 to b+1))
    private def insideNT = new BinaryUpdateManager(this, data.inside.insideNTKernels, devInside, devInside, devInside, insideBot, insideTop, insideBot, (b, e, l) => (e-1 to e-1))
    private def insideTN = new BinaryUpdateManager(this, data.inside.insideTNKernels, devInside, devInside, devInside, insideBot, insideBot, insideTop, (b, e, l) => (b+1 to b+1))
    private def insideNN = new BinaryUpdateManager(this, data.inside.insideNNKernels, devInside, devInside, devInside, insideBot, insideTop, insideTop, (b, e, l) => (b+1 to e-1))

    // here, "parentChart" is actually the left child, left is the parent, right is the right completion
    private def outsideTT_L = new BinaryUpdateManager(this, data.outside.outside_L_TTKernels, devOutside, devOutside, devInside, outsideBot, outsideBot, insideBot, (b, e, l) => (e+1 to e+1))
    private def outsideNT_L = new BinaryUpdateManager(this, data.outside.outside_L_NTKernels, devOutside, devOutside, devInside, outsideTop, outsideBot, insideBot, (b, e, l) => (e+1 to e+1))
    private def outsideTN_L = new BinaryUpdateManager(this, data.outside.outside_L_TNKernels, devOutside, devOutside, devInside, outsideBot, outsideBot, insideTop, (b, e, l) => (e+1 to l))
    private def outsideNN_L = new BinaryUpdateManager(this, data.outside.outside_L_NNKernels, devOutside, devOutside, devInside, outsideTop, outsideBot, insideTop, (b, e, l) => (e+1 to l))

    // here, "parentChart" is actually the right child, right is the parent, left is the left completion
    private def outsideTT_R = new BinaryUpdateManager(this, data.outside.outside_R_TTKernels, devOutside, devInside, devOutside, outsideBot, insideBot, outsideBot, (b, e, l) => (b-1 to b-1))
    private def outsideNT_R = new BinaryUpdateManager(this, data.outside.outside_R_NTKernels, devOutside, devInside, devOutside, outsideBot, insideTop, outsideBot, (b, e, l) => (0 to b-1))
    private def outsideTN_R = new BinaryUpdateManager(this, data.outside.outside_R_TNKernels, devOutside, devInside, devOutside, outsideTop, insideBot, outsideBot, (b, e, l) => (b-1 to b-1))
    private def outsideNN_R = new BinaryUpdateManager(this, data.outside.outside_R_NNKernels, devOutside, devInside, devOutside, outsideTop, insideTop, outsideBot, (b, e, l) => (0 to b-1))

    private def outsideTU = new UnaryUpdateManager(this, data.outside.outsideTUKernels, devOutside, outsideBot, outsideTop)
    private def outsideNU = new UnaryUpdateManager(this, data.outside.outsideNUKernels, devOutside, outsideBot, outsideTop)


    def addMasksToBatches(batch: Batch, ev: CLEvent*): Batch = {
      val insideEvents = inside(batch, ev:_*)
      val outsideEvents = outside(batch, insideEvents)
      val ev2 = computeMasks(batch, -9, outsideEvents)
      ev2.waitFor()
      val denseMasks = maskCharts.toDense
      maskCharts.data.waitUnmap()
      batch.copy(masks = Some(denseMasks))
    }

    def extractParses(batch: Batch, masks: CLMatrix[Int], events: CLEvent*): ParSeq[BinarizedTree[C]] = {
      events.foreach(_.waitFor())
      val in = if (profile) System.currentTimeMillis() else 0L
      val dmMasks:DenseMatrix[Int] = masks.toDense
      val trees = for (s <- 0 until batch.numSentences par) yield try {
        import batch.cellOffsets
        val length = batch.sentences(s).length
        val numCells = (cellOffsets(s+1)-cellOffsets(s))/2
        assert(numCells == TriangularArray.arraySize(length))
        val botBegin = cellOffsets(s)
        val botEnd = botBegin + numCells
        val topBegin = botEnd
        val topEnd = topBegin + numCells
        val bot = dmMasks(::, botBegin until botEnd)
        val top = dmMasks(::, topBegin until topEnd)

        def recTop(begin: Int, end: Int):BinarizedTree[C] = {
          val column:DenseVector[Int] = top(::, ChartHalf.chartIndex(begin, end, length))
          val x = firstSetBit(column:DenseVector[Int])
          if (x == -1) {
            assert(begin == end - 1, s"$column ($begin, $end) $length $s ${batch.numSentences}")
            recBot(begin, end)
          } else {
            assert(column.valuesIterator.exists(_ != 0), (begin, end))
//            val label = structure.nontermIndex.get(x)
            val label = structure.refinements.labels.coarseIndex.get(x)
            val t = recBot(begin, end)
            new UnaryTree[C](label, t, IndexedSeq.empty, Span(begin, end))
          }
        }

        def recBot(begin: Int, end: Int):BinarizedTree[C] = {
          val column:DenseVector[Int] = bot(::, ChartHalf.chartIndex(begin, end, length))
          val x = firstSetBit(column:DenseVector[Int])
          if (begin == end - 1) {
//            val label = structure.termIndex.get(x)
            val label = structure.refinements.labels.coarseIndex.get(x)
            NullaryTree(label, Span(begin, end))
          } else {
//            val label = structure.nontermIndex.get(x)
            val label = structure.refinements.labels.coarseIndex.get(x)
            for (split <- (begin+1) until end) {
              val left = (if (begin == split - 1) bot else top)(::, ChartHalf.chartIndex(begin, split, length))
              val right = (if (end == split + 1) bot else top)(::, ChartHalf.chartIndex(split, end, length))
              if (left.any && right.any) {
                return BinaryTree[C](label, recTop(begin, split), recTop(split, end), Span(begin, end))
              }
            }

            error(s"nothing here $length!" + " "+ (begin, end) +
              {for (split <- (begin+1) until end) yield {
                val left = (if (begin == split - 1) bot else top)(::, ChartHalf.chartIndex(begin, split, length))
                val right = (if (end == split + 1) bot else top)(::, ChartHalf.chartIndex(split, end, length))
                (ChartHalf.chartIndex(begin, split, length), ChartHalf.chartIndex(split, end, length), split, left,right)

              }} + " " +batch.sentences(s))
          }
        }

        recTop(0, length)

      } catch {
        case ex: Throwable => ex.printStackTrace(); null
      }
      val out = if (profile) System.currentTimeMillis() else 0L
      masks.data.waitUnmap()
      if (profile) {
        println(s"Parse extraction took:  ${(out - in)/1000.0}s")
      }
      trees
    }

  }

  private case class Batch(lengthTotals: Array[Int],
                           cellOffsets: Array[Int],
                           sentences: IndexedSeq[IndexedSeq[W]],
                           masks: Option[DenseMatrix[Int]]) {
    def totalLength = lengthTotals.last
    def numSentences = sentences.length
    val maxLength = sentences.map(_.length).max
    assert(cellOffsets.last <= devInside.cols)

    def isAllowedSpan(sent: Int, begin: Int, end: Int) = maskFor(sent, begin, end).forall(_.any)

    def maskFor(sent: Int, begin: Int, end: Int) = masks.map(m =>  m(::, insideCharts(sent).bot.cellOffset(begin, end)))

    def hasMasks = masks.nonEmpty

    val _workArrayOffsetsForSpan = Array.tabulate(maxLength+1)(span => sentences.scanLeft(0)((off, sent) => off + math.max(0,sent.length-span+1)).toArray)
    def workArrayOffsetsForSpan(sent: Int, span: Int) = Range(_workArrayOffsetsForSpan(span)(sent), _workArrayOffsetsForSpan(span)(sent+1))

    def totalLengthForSpan(span: Int) = _workArrayOffsetsForSpan(span).last

    lazy val insideCharts = for (i <- 0 until numSentences) yield {
      val numCells = (cellOffsets(i+1)-cellOffsets(i))/2
      assert(numCells == TriangularArray.arraySize(sentences(i).length))
      val chart = new ParseChart(sentences(i).length, devInside(::, cellOffsets(i) until (cellOffsets(i) + numCells)), devInside(::, cellOffsets(i) + numCells until cellOffsets(i+1)))
      chart
    }

    lazy val outsideCharts = {
      for (i <- 0 until numSentences) yield {

        val numCells = (cellOffsets(i+1)-cellOffsets(i))/2
        assert(numCells == TriangularArray.arraySize(sentences(i).length))
        val botBegin = cellOffsets(i)
        val botEnd = botBegin + numCells
        val topBegin = botEnd
        val topEnd = topBegin + numCells
        assert(topEnd < devOutside.cols)
        val chart = new ParseChart(sentences(i).length, devOutside(::, botBegin until botEnd), devOutside(::, topBegin until topEnd))
        chart
      }
    }

  }

  private def getBatches(sentences: IndexedSeq[IndexedSeq[W]]): IndexedSeq[Batch] = {
    val result = ArrayBuffer[Batch]()
    var current = ArrayBuffer[IndexedSeq[W]]()
    var currentLengthTotal = 0
    var currentCellTotal = 0
    for ( (s, i) <- sentences.zipWithIndex) {
      currentLengthTotal += s.length
      currentCellTotal += TriangularArray.arraySize(s.length) * 4
      if (currentLengthTotal > numWorkCells || currentCellTotal > numChartCells) {
        assert(current.nonEmpty)
        result += createBatch(current)
        currentLengthTotal = s.length
        currentCellTotal = TriangularArray.arraySize(s.length) * 4
        current = ArrayBuffer()
      }
      current += s
    }

    if (current.nonEmpty) result += createBatch(current)
    result
  }

  private def createBatch(sentences: IndexedSeq[IndexedSeq[W]]): Batch = {
    val lengthTotals = sentences.scanLeft(0)((acc, sent) => acc + sent.length)
    println(s"Batch size of ${sentences.length}, total length of ${lengthTotals.last}")
    val cellTotals = sentences.scanLeft(0)((acc, sent) => acc + TriangularArray.arraySize(sent.length) * 2)
    assert(cellTotals.last <= devInside.cols, cellTotals.last + " " +  devInside.cols)
    Batch(lengthTotals.toArray, cellTotals.toArray, sentences, None)
  }

  private class BinaryUpdateManager(parser: ActualParser,
                                    updater: CLBinaryRuleUpdater,
                                    parentChartMatrix: CLMatrix[Float],
                                    leftChartMatrix: CLMatrix[Float],
                                    rightChartMatrix: CLMatrix[Float],
                                    parentChart: (Batch,Int)=>ChartHalf,
                                    leftChart: (Batch,Int)=>ChartHalf,
                                    rightChart: (Batch,Int)=>ChartHalf,
                                    ranger: (Int, Int, Int)=>Range) {



    case class WorkItem(sent: Int, begin: Int, end: Int, masks: DenseVector[Double])

    var splitPointOffset = 0 // number of unique parent spans used so far
    var offset = 0 // number of work cells used so far.

    // TODO: ugh, state
    var lastParent = -1

    private def enqueue(batch: Batch, span: Int, parent: Int, left: Int, right: Int, events: Seq[CLEvent]): Seq[CLEvent] = {
      if (splitPointOffset == 0 || lastParent != parent) {
        splitPointOffsets(splitPointOffset) = offset
        splitPointOffset += 1
      }
      lastParent = parent
      pArray(offset) = parent
      lArray(offset) = left
      rArray(offset) = right
      offset += 1
      if (offset >= numWorkCells)  {
        flushQueue(batch, span, events)
      } else {
        events
      }
    }

    private def flushQueue(batch: Batch, span: Int, ev: Seq[CLEvent]): Seq[CLEvent] = {
      if (offset != 0) {
        splitPointOffsets(splitPointOffset) = offset

        // copy ptrs to opencl
        val evTLeftPtrs  =  devLeftPtrs.writeArray(queue, lArray, offset, ev:_*) profileIn hdTransferEvents
        val evTRightPtrs = devRightPtrs.writeArray(queue, rArray, offset, ev:_*) profileIn hdTransferEvents

        // do transpose based on ptrs
        val evTransLeft  = transposeCopy.permuteTransposeCopy(devLeft(0 until offset, ::), leftChartMatrix, devLeftPtrs, offset, evTLeftPtrs) profileIn transferEvents
        val evTransRight = transposeCopy.permuteTransposeCopy(devRight(0 until offset, ::), rightChartMatrix, devRightPtrs, offset, evTRightPtrs) profileIn transferEvents

        // copy parent pointers
        val evWriteDevParent = devParentPtrs.writeArray(queue, pArray, offset, ev:_*) profileIn hdTransferEvents
        // corresponding splits
        val evWriteDevSplitPoint = devSplitPointOffsets.writeArray(queue, splitPointOffsets, splitPointOffset + 1, ev:_*) profileIn hdTransferEvents

        val zeroParent = zmk.shapedFill(devParent(0 until offset, ::), parser._zero, ev:_*) profileIn memFillEvents
        val kEvents: IndexedSeq[CLEvent] = updater.update(devParent(0 until offset, ::), devParentPtrs,
          devLeft(0 until offset, ::), devRight(0 until offset, ::),
          maskCharts, evTransLeft, evTransRight, zeroParent, evWriteDevParent) map  (_ profileIn binaryEvents)

        val sumEv = parser.data.util.sumSplitPoints(devParent,
          parentChartMatrix,
          devParentPtrs, splitPointOffset,
          devSplitPointOffsets,
          32 / span max 1, parser.data.numSyms, Seq(evWriteDevParent, evWriteDevSplitPoint) ++ kEvents:_*) profileIn sumEvents

        offset = 0
        splitPointOffset = 0
        IndexedSeq(sumEv)
      } else {
        ev
      }
    }

    def doUpdates(batch: Batch, span: Int, events: CLEvent*) = {

      var ev = events
      splitPointOffset = 0
      lastParent = -1

      val allSpans = if (batch.hasMasks) {
        import BitHacks.OrderBitVectors.OrderingBitVectors
        val allSpans = for {
          sent <- 0 until batch.numSentences
          start <- 0 to batch.sentences(sent).length - span
          _ = total += 1
          mask <-  batch.maskFor(sent, start, start + span)
          if {val x = mask.any; if(!x) pruned += 1; x }
        } yield (sent, start, start+span, mask)
        val ordered = allSpans.sortBy(_._4)
//        println(s"Sorting $span took " + (out - in)/1000.0)
//        println(ordered.mkString("{\t","\n\t","\n}"))
        ordered
//        allSpans
      } else {
         for {
          sent <- 0 until batch.numSentences
          start <- 0 to batch.sentences(sent).length - span
        } yield (sent, start, start+span, null)
      }

      for {
        (sent, start, end, _) <- allSpans
      //  if batch.isAllowedSpan(sent,start,start+span)
        split <- ranger(start, start + span, batch.sentences(sent).length)
        if split >= 0 && split <= batch.sentences(sent).length
      } {
        val end = start + span
        val parentTi = parentChart(batch, sent).cellOffset(start,end)
        val leftChildAllowed = if (split < start) batch.isAllowedSpan(sent,split, start) else batch.isAllowedSpan(sent, start, split)
        val rightChildAllowed = if (split < end) batch.isAllowedSpan(sent,split,end) else batch.isAllowedSpan(sent, end, split)

        if (leftChildAllowed && rightChildAllowed) {
          val leftChild = if (split < start) leftChart(batch, sent).cellOffset(split,start) else leftChart(batch, sent).cellOffset(start, split)
          val rightChild = if (split < end) rightChart(batch, sent).cellOffset(split,end) else rightChart(batch, sent).cellOffset(end, split)
          ev = enqueue(batch, span, parentTi, leftChild, rightChild, ev)
        }

      }

      if (offset > 0) {
        ev = flushQueue(batch, span, ev)
      }

      assert(ev.length == 1)
      ev.head
    }

  }

  private class UnaryUpdateManager(parser: ActualParser,
                                   kernels: CLUnaryRuleUpdater,
                                   scoreMatrix: CLMatrix[Float],
                                   parentChart: (Batch,Int)=>ChartHalf,
                                   childChart: (Batch,Int)=>ChartHalf) {

    var offset = 0 // number of cells used so far.

    private def enqueue(span: Int, parent: Int, left: Int, events: Seq[CLEvent]) = {
      lArray(offset) = left
      pArray(offset) = parent
      offset += 1
      if (offset >= numWorkCells)  {
        logger.debug(s"flush unaries!")
        flushQueue(span, events)
      } else {
        events
      }
    }

    private def flushQueue(span: Int, ev: Seq[CLEvent]) = {
      if (offset != 0) {
        val zz = zmk.shapedFill(devParent(0 until offset, ::), parser._zero, ev:_*) profileIn memFillEvents

        val wevl = devLeftPtrs.writeArray(queue, lArray, offset, ev:_*) profileIn hdTransferEvents

        val wl = transposeCopy.permuteTransposeCopy(devLeft(0 until offset, ::), scoreMatrix, devLeftPtrs, offset, wevl) profileIn transferEvents

        val endEvents = kernels.update(devParent(0 until offset, ::),
          devLeft(0 until offset, ::),  wl, zz) map  (_ profileIn unaryEvents)


        val ev2 = devParentPtrs.writeArray(queue, pArray, offset, ev:_*) profileIn hdTransferEvents

        val _ev = transposeCopy.permuteTransposeCopyOut(scoreMatrix, devParentPtrs, offset, devParent(0 until offset, ::), (ev2 +: endEvents):_*) profileIn sumToChartsEvents

        offset = 0
        IndexedSeq(_ev)
      } else {
        ev
      }
    }

    def doUpdates(batch: Batch, span: Int, events: CLEvent*) = {
      var ev = events

      for {
        sent <- 0 until batch.numSentences
        start <- 0 to batch.sentences(sent).length - span
        if batch.isAllowedSpan(sent, start, start + span)
      } {
        val end = start + span
        val parentTi = parentChart(batch, sent).cellOffset(start,end)
        val child = childChart(batch, sent).cellOffset(start,end)

        ev = enqueue(span, parentTi, child, ev)
      }

      if (offset > 0) {
        flushQueue(span, ev)
      }

      assert(ev.length == 1)
      ev.head
    }

  }
}

case class StripUnaries() extends TreeAnnotator[AnnotatedLabel, String, AnnotatedLabel] {
  def apply(tree: BinarizedTree[AnnotatedLabel], words: Seq[String]):BinarizedTree[AnnotatedLabel] = tree match {
    case UnaryTree(a, b, _, span) => UnaryTree(a, apply(b, words), IndexedSeq.empty, span)
    case BinaryTree(a, b, c, span) => BinaryTree(a, apply(b, words), apply(c, words), span)
    case x => x
  }


}

object CLParser extends Logging {

  case class Params(annotator: TreeAnnotator[AnnotatedLabel, String, AnnotatedLabel] = Xbarize(),
                    prune: Boolean = false,
                    useGPU: Boolean = true, profile: Boolean = false,
                    numToParse: Int = 1000, codeCache: File = new File("grammar.grz"),
                    maxParseLength: Int = 10000,
                    jvmParse: Boolean = false, parseTwice: Boolean = false,
                    textGrammarPrefix: String = null, checkPartitions: Boolean = false)

  def main(args: Array[String]) = {
    import ParserParams.JointParams

    val params = CommandLineParser.readIn[JointParams[Params]](args)
    val myParams:Params = params.trainer
    import myParams._
    println("Training Parser...")
    println(params)
    val transformed = params.treebank.copy(binarization="left", keepUnaryChainsFromTrain = false).trainTrees
    val toParse =  params.treebank.trainTrees.filter(_.words.length <= maxParseLength).map(_.words).take(numToParse)


    val grammar: SimpleRefinedGrammar[AnnotatedLabel, AnnotatedLabel, String] = if (textGrammarPrefix == null) {
      GenerativeParser.annotated(annotator, transformed)
    } else {
      SimpleRefinedGrammar.parseBerkeleyText(textGrammarPrefix, -12, CloseUnaries.None)
    }

    lazy val xbarGrammar = GenerativeParser.annotated(grammar.grammar, grammar.lexicon, Xbarize(), transformed)


    grammar.prettyPrint(new FileWriter("grammar.out"))
    val out = new PrintStream(new FileOutputStream("refs.txt"))
    out.println(xbarGrammar.refinements.rules.toString())
    out.close()

    implicit val context = if (useGPU) {
      val gpu = JavaCL.listPlatforms.flatMap(_.listGPUDevices(true)).sortBy(d => d.toString.contains("GeForce") || d.toString.toLowerCase.contains("nvidia")).last
      JavaCL.createContext(new java.util.HashMap(), gpu)
    } else {
      //      val gpu = JavaCL.listPlatforms.flatMap(_.listGPUDevices(true)).last
      //      JavaCL.createContext(new java.util.HashMap(), gpu)
      val cpuPlatform:CLPlatform = JavaCL.listPlatforms().filter(_.listCPUDevices(true).nonEmpty).head
      cpuPlatform.createContext(new java.util.HashMap(), cpuPlatform.listCPUDevices(true):_*)
    }
    println(context)

    var parserData:CLParserData[AnnotatedLabel, AnnotatedLabel, String] = if (codeCache != null && codeCache.exists()) {
      CLParserData.read(new ZipFile(codeCache))
    } else {
      null
    }

    if (parserData == null || parserData.grammar.signature != grammar.signature) {
      println("Regenerating parser data")
      parserData = CLParserData.make(grammar)
      if (codeCache != null) {
        parserData.write(new BufferedOutputStream(new FileOutputStream(codeCache)))
      }
    }

    val kern = if (prune) {
      val genData = CLParserData.make(xbarGrammar)

      fromParserDatas[AnnotatedLabel, AnnotatedLabel, String](IndexedSeq(genData, parserData), profile)
    } else {
      fromParserData[AnnotatedLabel, AnnotatedLabel, String](parserData, profile)
    }


    if (checkPartitions) {
      val partsX = kern.partitions(toParse)
      println(partsX)
      val parser = Parser(grammar, if (kern.isViterbi) new ViterbiDecoder[AnnotatedLabel, String] else new MaxConstituentDecoder[AnnotatedLabel, String])
      val parts2 = toParse.par.map(parser.marginal(_).logPartition)
      println(parts2)
      println("max difference: " + (DenseVector(partsX.map(_.toDouble):_*) - DenseVector(parts2.seq:_*)).norm(Double.PositiveInfinity))
      System.exit(0)
    }
    var timeIn = System.currentTimeMillis()
    val trees = kern.parse(toParse)
    var timeOut = System.currentTimeMillis()
    println(trees zip toParse map {case (k,v) if k != null => k render v; case (k,v) => ":("})
    println(s"CL Parsing took: ${(timeOut-timeIn)/1000.0}")
    if (parseTwice) {
      timeIn = System.currentTimeMillis()
      val parts2 = kern.parse(toParse)
      timeOut = System.currentTimeMillis()
      //println(parts2 zip train map {case (k,v) => k render v})
      println(s"CL Parsing took x2: ${(timeOut-timeIn)/1000.0}")
    }
    if (jvmParse) {
      val parser = if(prune) {
        Parser(new ConstraintCoreGrammarAdaptor(xbarGrammar.grammar, xbarGrammar.lexicon, new ParserChartConstraintsFactory(Parser(xbarGrammar, new ViterbiDecoder[AnnotatedLabel, String]), (_:AnnotatedLabel).isIntermediate)),
          grammar,
          if (kern.isViterbi) new ViterbiDecoder[AnnotatedLabel, String] else new MaxConstituentDecoder[AnnotatedLabel, String])
      } else {
        Parser(grammar, if (kern.isViterbi) new ViterbiDecoder[AnnotatedLabel, String] else new MaxConstituentDecoder[AnnotatedLabel, String])
      }
      timeIn = System.currentTimeMillis()
      val margs = toParse.map { w =>
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

  def fromParserDatas[L, L2, W](data: IndexedSeq[CLParserData[L, L2, W]], profile: Boolean)(implicit context: CLContext): CLParser[L, L2, W] = {
    val kern = new CLParser[L, L2, W](data, profile = profile)
    kern
  }


}


case class CLParserData[C, L, W](grammar: SimpleRefinedGrammar[C, L, W],
                                 structure: RuleStructure[C, L],
                                 semiring: RuleSemiring,
                                 inside: CLInsideKernels,
                                 outside: CLOutsideKernels,
                                 masks: CLMaskKernels,
                                 util: CLParserUtils,
                                 isViterbi: Boolean) {

  def numSyms = structure.nontermIndex.size max structure.termIndex.size
  def maskSize = masks.maskSize

  def write(out: OutputStream) {
    val zout = new ZipOutputStream(out)
    ZipUtil.serializedEntry(zout, "grammar", grammar)
    ZipUtil.serializedEntry(zout, "structure", structure)
    ZipUtil.serializedEntry(zout, "semiring", semiring)
    inside.write(zout)
    outside.write(zout)
    util.write(zout)
    masks.write(zout)
    zout.close()
  }
}

object CLParserData {
  def make[C, L, W](grammar: SimpleRefinedGrammar[C, L, W])(implicit context: CLContext) = {
    implicit val viterbi = ViterbiRuleSemiring
    val ruleScores: Array[Float] = Array.tabulate(grammar.refinedGrammar.index.size){r =>
      val score = grammar.ruleScoreArray(grammar.refinements.rules.project(r))(grammar.refinements.rules.localize(r))
      viterbi.fromLogSpace(score.toFloat)
    }
    val structure = new RuleStructure(grammar.refinements, grammar.refinedGrammar, ruleScores)
    val inside = CLInsideKernels.make(structure)
    val outside =  CLOutsideKernels.make(structure)
    val util = CLParserUtils.make(structure)
    val masks = CLMaskKernels.make(structure)

    new CLParserData(grammar, structure, viterbi, inside, outside, masks, util, viterbi.plusIsIdempotent)
  }

  def read[C, L, W](file: ZipFile)(implicit context: CLContext) = {
    val gr = ZipUtil.deserializeEntry[SimpleRefinedGrammar[C, L, W]](file.getInputStream(file.getEntry("grammar")))
    val structure = ZipUtil.deserializeEntry[RuleStructure[C, L]](file.getInputStream(file.getEntry("structure")))
    val semiring = ZipUtil.deserializeEntry[RuleSemiring](file.getInputStream(file.getEntry("semiring")))
    val inside = CLInsideKernels.read(file)
    val outside = CLOutsideKernels.read(file)
    val util = CLParserUtils.read(file)
    val masks = CLMaskKernels.read(file)

    CLParserData(gr, structure, semiring, inside, outside, masks, util, semiring.plusIsIdempotent)
  }
}
