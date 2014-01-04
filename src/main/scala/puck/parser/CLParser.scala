package puck
package parser

import epic.AwesomeBitSet
import gen._
import com.nativelibs4java.opencl._
import com.typesafe.scalalogging.slf4j.Logging
import puck.util._
import puck.linalg.CLMatrix
import puck.linalg.kernels.{CLMatrixSliceCopy, CLMatrixTransposeCopy}
import breeze.collection.mutable.TriangularArray
import breeze.linalg.{DenseVector, DenseMatrix}
import epic.trees.annotations.TreeAnnotator
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
import scala.collection.parallel.immutable.ParSeq
import java.util.Collections
import epic.trees.UnaryTree
import scala.Some
import epic.parser.ViterbiDecoder
import epic.trees.NullaryTree
import epic.trees.BinaryTree
import epic.trees.annotations.Xbarize
import epic.trees.Span
import puck.parser.RuleStructure

/**
 * TODO
 *
 * @author dlwh
 **/
class CLParser[C, L, W](data: IndexedSeq[CLParserData[C, L, W]],
//                        maxAllocSize: Long = 1L<<30, // 1 gig
//                        maxAllocSize: Long = 1L<<32, // 4 gig
                        maxAllocSize: Long = 1<<30, //1.25 G
                        maxSentencesPerBatch: Long = 400,
                        doEmptySpans: Boolean = false,
                        profile: Boolean = true,
                        var oldPruning: Boolean = false,
                        trackRules: Boolean = false)(implicit val context: CLContext) extends Logging {
  val skipFineWork = false

  def parse(sentences: IndexedSeq[IndexedSeq[W]]):IndexedSeq[BinarizedTree[C]] = synchronized {
    val mask: Option[DenseMatrix[Int]] = computeMasks(sentences)
    parsers.last.parse(sentences, mask)
  }


  def partitions(sentences: IndexedSeq[IndexedSeq[W]]):IndexedSeq[Float] = synchronized {
    val mask: Option[DenseMatrix[Int]] = computeMasks(sentences)
    parsers.last.partitions(sentences, mask)
  }


  private def computeMasks(sentences: IndexedSeq[IndexedSeq[W]]): Option[DenseMatrix[Int]] = {
    val ev = maskCharts.assignAsync(-1)
    ev.waitFor()
    val mask: Option[DenseMatrix[Int]] = None
    parsers.dropRight(1).foldLeft(mask)((mask, p) =>  Some(p.updateMasks(sentences, mask) ))
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
    devParentRaw.release()
    devInsideRaw.release()
    devOutsideRaw.release()
    devLeftRaw.release()
    devRightRaw.release()
    queue.release()
    devParentPtrs.release()

    maskParent.release()
    maskCharts.release()
  }

  // size in floats, just the number of symbols
  val cellSize:Int = roundUpToMultipleOf(data.map(_.numSyms).max, 32)
  val maskSize:Int = {
    assert(data.forall(_.maskSize == data.head.maskSize))
    data.head.maskSize
  }

  val (numDefaultWorkCells:Int, numDefaultChartCells: Int) = {
    val sizeOfFloat = 4
    val fractionOfMemoryToUse = 0.7 // slack!
//    val fractionOfMemoryToUse = 0.9 // slack!
    val amountOfMemory = ((context.getDevices.head.getGlobalMemSize min maxAllocSize) * fractionOfMemoryToUse).toInt  - maxSentencesPerBatch * 3 * 4;
    val maxPossibleNumberOfCells = ((amountOfMemory / sizeOfFloat) / (cellSize + 4 + maskSize)).toInt // + 4 for each kind of offset
    // We want numGPUCells and numGPUChartCells to be divisible by 16, so that we get aligned strided access:
    //       On devices of compute capability 1.0 or 1.1, the k-th thread in a half warp must access the
    //       k-th word in a segment aligned to 16 times the size of the elements being accessed; however,
    //       not all threads need to participate... If sequential threads in a half warp access memory that is
    //       sequential but not aligned with the segments, then a separate transaction results for each element
    //       requested on a device with compute capability 1.1 or lower.
    val numberOfUnitsOf32 = maxPossibleNumberOfCells / 32
    // average sentence length of sentence, let's say n.
    // for the gpu charts, we'll need (n choose 2) * 2 * 2 =
    // for the "P/L/R" parts, the maximum number of relaxations (P = L * R * rules) for a fixed span
    // in a fixed sentence is (n/2)^2= n^2/4.
    // Take n = 32, then we want our P/L/R arrays to be of the ratio (3 * 256):992 \approx 3/4 (3/4 exaclty if we exclude the - n term)
    // doesn't quite work the way we want (outside), so we'll bump the number to 4/5
    val relativeSizeOfChartsToP = 7
    val baseSize = numberOfUnitsOf32 / (3 + relativeSizeOfChartsToP)
    val extra = numberOfUnitsOf32 % (3 + relativeSizeOfChartsToP)
    val plrSize = baseSize
    // TODO, can probably do a better job of these calculations?
    (plrSize * 32, (baseSize * relativeSizeOfChartsToP + extra) * 32)
  }

  def maxNumWorkCells = parsers.map(_.numWorkCells).max
  def maxNumChartCells = parsers.map(_.numChartCells).max

  // On the Device side we have 4 Matrices:
  // One is where we calculate P = L * R * rules, for fixed spans and split points (the "bot")
  // One is the L part of the above
  // Another is the R part.
  // finally, we have the array of parse charts
  private val devParentRaw, devLeftRaw, devRightRaw = context.createFloatBuffer(CLMem.Usage.InputOutput, numDefaultWorkCells.toLong * cellSize)
  private lazy val maskParent = new CLMatrix[Int](maskSize, maxNumWorkCells)
  private lazy val devParentPtrs = context.createIntBuffer(CLMem.Usage.Input, maxNumWorkCells)
  private lazy val offsetBuffer = new CLBufferPointerPair[Integer](context.createIntBuffer(CLMem.Usage.Input, maxNumWorkCells * 3))
  private lazy val devSplitPointOffsets = context.createIntBuffer(CLMem.Usage.Input, maxNumWorkCells + 1)
  // transposed
  private val devInsideRaw, devOutsideRaw = context.createFloatBuffer(CLMem.Usage.InputOutput, numDefaultChartCells/2 * cellSize)
  //  private val maskCharts = new CLMatrix[Int](maskSize, numDefaultChartCells)
  private lazy val maskCharts = new CLMatrix[Int](maskSize, maxNumChartCells)

  // other stuff
  private val zmk = ZeroMemoryKernel()
  private val transposeCopy = CLMatrixTransposeCopy()
  private val sliceCopy = CLMatrixSliceCopy()

  private val parsers = data.map(new ActualParser(_))
  var pruned = 0
  var total = 0
  var rulesEvaled = 0L
  var theoreticalRules = 0L
  var rulesTotal = 0L
  var sortTime = 0L

  private class ActualParser(val data: CLParserData[C, L, W]) {
    import data._


    def parse(sentences: IndexedSeq[IndexedSeq[W]], mask: Option[DenseMatrix[Int]]) = synchronized {
      logTime("parse", sentences.length) {
        getBatches(sentences, mask).iterator.flatMap { batch =>
          var ev = inside(batch)
          ev = outside(batch, ev)
          val ev3 = computeViterbiMasks(batch, ev)
          Option(ev3).foreach(_.waitFor())
          val dmMasks:DenseMatrix[Int] = maskCharts(::, 0 until batch.numCellsUsed).toDense
          val parses = extractParses(batch, dmMasks, ev3)
          maskCharts.data.waitUnmap()
          parses
        }.toIndexedSeq
      }
    }

    def partitions(sentences: IndexedSeq[IndexedSeq[W]], mask: Option[DenseMatrix[Int]]) = synchronized {
      logTime("partitions", sentences.length) {
        getBatches(sentences, mask).iterator.flatMap { batch =>
          var ev = inside(batch)
          val dest = context.createFloatBuffer(CLMem.Usage.Output, sentences.length)
          ev = devParentPtrs.writeArray(queue, batch.insideCharts.map(_.top.rootIndex).toArray, batch.numSentences, ev) profileIn hdTransferEvents
          ev = data.util.getRootScores(dest, devInside, devParentPtrs, batch.numSentences, structure.root, ev)
          dest.read(queue, ev).getFloats(batch.numSentences)
        }.toIndexedSeq
      }
    }

    def updateMasks(sentences: IndexedSeq[IndexedSeq[W]], mask: Option[DenseMatrix[Int]]): DenseMatrix[Int] = synchronized {
      logTime("masks", sentences.length) {
        val masks = getBatches(sentences, mask).iterator.map { computePruningMasks(_) }.toIndexedSeq
        DenseMatrix.horzcat(masks:_*)
      }
    }

    def computePruningMasks(batch: Batch[W], ev: CLEvent*): DenseMatrix[Int] = {
      val insideEvents = inside(batch, ev:_*)
      val outsideEvents = outside(batch, insideEvents)
      val ev2 = computeMasks(batch, -7, outsideEvents)
      ev2.waitFor()
      val denseMasks:DenseMatrix[Int] = maskCharts(::, 0 until batch.numCellsUsed).toDense
      maskCharts.data.waitUnmap()
      denseMasks
    }

    def myCellSize:Int = roundUpToMultipleOf(data.numSyms, 32)
    def numWorkCells = ((devParentRaw.getElementCount) / myCellSize ).toInt
    def numChartCells = ((devInsideRaw.getElementCount) / myCellSize ).toInt

    // (dest, leftSource, rightSource) (right Source if binary rules)
    val pArray, lArray, rArray = new Array[Int](numWorkCells)
    val splitPointOffsets = new Array[Int](numWorkCells+1)


    val devParent = new CLMatrix[Float]( numWorkCells, myCellSize, devParentRaw)
    val devLeft = new CLMatrix[Float]( numWorkCells, myCellSize, devLeftRaw)
    val devRight = new CLMatrix[Float]( numWorkCells, myCellSize, devRightRaw)

    val devInside = new CLMatrix[Float](myCellSize, numChartCells, devInsideRaw)
    val devOutside = new CLMatrix[Float](myCellSize, numChartCells, devOutsideRaw)

    def _zero: Float = data.semiring.zero

    def inside(batch: Batch[W], events: CLEvent*):CLEvent = synchronized {
      allProfilers.foreach(_.clear())
      allProfilers.foreach(_.tick())
      pruned = 0
      total = 0
      sortTime = 0
      rulesEvaled = 0
      theoreticalRules = 0
      rulesTotal = 0

      val evZeroCharts = zmk.fillMemory(devInside.data, _zero, events:_*) profileIn initMemFillEvents
      val evZeroOutside = zmk.fillMemory(devOutside.data, _zero, events:_*) profileIn initMemFillEvents
      val init = initializeTagScores(batch, evZeroCharts, evZeroOutside)

//      maskCharts.assignAsync(-1, evZeroCharts).waitFor
      var ev = insideTU.doUpdates(batch, 1, init)

      for (span <- 2 to batch.maxLength) {
        print(s"$span ")
        ev = insideBinaryPass(batch, span, ev)
//        queue.finish()
//        println(batch.insideCharts.head.bot.toString(structure, _zero))
        ev = insideNU.doUpdates(batch, span, ev)

      }

      if (profile) {
        queue.finish()
        allProfilers.foreach(_.tock())
        allProfilers.foreach(p => println(s"Inside $p"))
        println(f"Time accounted for: ${allProfilers.map(_.processingTime).sum}%.3f")
        println("Sorting took: " + sortTime/1000.0)
        println(s"Pruned $pruned/$total")
        println(s"Rules evaled: $rulesEvaled/$rulesTotal ${(rulesEvaled.toDouble/rulesTotal)}")
        println(s"Rules Theoretical (by rulesTotal): $theoreticalRules/$rulesTotal ${(theoreticalRules.toDouble/rulesTotal)}")
        println(s"Rules Theoretical (by rulesEvaled): $theoreticalRules/$rulesEvaled ${(theoreticalRules.toDouble/rulesEvaled)}")
      }

      ev
    }

    def outside(batch: Batch[W], event: CLEvent):CLEvent = synchronized {
      var ev = event
      allProfilers.foreach(_.clear())
      allProfilers.foreach(_.tick())
      rulesEvaled = 0
      rulesTotal = 0
      theoreticalRules = 0
      pruned  = 0
      total = 0

     ev = offsetBuffer.writeInts(0, batch.outsideCharts.map(_.top.rootIndex).toArray, 0, batch.outsideCharts.length, ev) profileIn hdTransferEvents
      ev = data.util.setRootScores(devOutside, offsetBuffer.buffer, batch.numSentences, structure.root, data.semiring.one, ev) profileIn memFillEvents
//      ev = data.util.setRootScores(devOutside, devParentPtrs, batch.numSentences, structure.root, data.semiring.one, ev) profileIn memFillEvents

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
        println("Sorting took: " + sortTime/1000.0)
        println(s"Pruned $pruned/$total")
        println(s"Rules evaled: $rulesEvaled/$rulesTotal ${(rulesEvaled.toDouble/rulesTotal)}")
      }

      ev
    }

    def computeViterbiMasks(batch: Batch[W], events: CLEvent*):CLEvent = synchronized {
      computeMasks(batch, -4E-3f, events:_*)
    }

    def initializeTagScores(batch: Batch[W], events: CLEvent*) = {
      val totalLength = batch.totalLength
      val tagScores = DenseMatrix.zeros[Float](totalLength, myCellSize)
      tagScores := _zero
      var offset = 0
      for (i <- 0 until batch.numSentences) {
        val sent = batch.sentences(i)
        val anch = data.grammar.tagScorer.anchor(sent)
        val lexAnch = data.grammar.lexicon.anchor(sent)
        val tags = tagScores(offset, ::)
        for (pos <- 0 until sent.length) {
          for(t <- lexAnch.allowedTags(pos); ref <- data.grammar.refinements.labels.refinementsOf(t)) {
            val index = ref
            val score = anch.scoreTag(pos, data.grammar.refinedGrammar.labelIndex.get(index))
            val gpuIndex = data.structure.labelIndexToTerminal(index)
            //        if(pos == 0) println(pos,t,ref,data.grammar.refinements.labels.fineIndex.get(ref), gpuIndex,score)
            pArray(offset) = batch.insideBotCell(i, pos, pos + 1)
            tagScores(offset, gpuIndex) = data.semiring.fromLogSpace(score.toFloat)
          }
          offset += 1
        }
      }

      val ev2 = devParentPtrs.writeArray(queue, pArray, offset, events:_*) profileIn hdTransferEvents
      val ev = devParent(0 until offset, ::).writeFrom(tagScores, false, ev2) map (_ profileIn  hdTransferEvents)
      transposeCopy.permuteTransposeCopyOut(devInside,  devParentPtrs, offset, devParent(0 until offset, ::), (ev2 +: ev):_*) profileIn sumToChartsEvents
    }

    private def markTerminalsInMasks(batch: Batch[W], events: CLEvent*) = {
      import batch._
      val set0 = maskCharts.assignAsync(0, events:_*)
      set0.waitFor()
      var offset = 0
      for (i <- 0 until numSentences) {
        for(pos <- 0 until sentences(i).length) {
          pArray(offset) = batch.insideBotCell(i, pos, pos + 1)
          offset += 1
        }
      }

      val ev2 = devParentPtrs.writeArray(queue, pArray, offset, events:_*) profileIn hdTransferEvents
      val ev = maskParent(::, 0 until totalLength).assignAsync(-1, set0) profileIn memFillEvents
      sliceCopy.sliceCopyOut(maskCharts.asInstanceOf[CLMatrix[Float]],  devParentPtrs, offset, maskParent(::, 0 until totalLength).asInstanceOf[CLMatrix[Float]], ev, ev2, set0) profileIn sumToChartsEvents
    }

    private def computeMasks(batch: Batch[W], threshold: Float, events: CLEvent*):CLEvent = synchronized {
      if(profile) {
        allProfilers.foreach(_.clear())
        allProfilers.foreach(_.tick())
      }

      val ev =  markTerminalsInMasks(batch, events:_*)

      val evr = data.masks.getMasks(maskCharts(::, 0 until batch.numCellsUsed),
        devInside(::, 0 until batch.numCellsUsed),
        devOutside(::, 0 until batch.numCellsUsed),
        0, batch.cellOffsets, structure.root, threshold, ev) profileIn masksEvents
      if (profile) {
        queue.finish()
        allProfilers.foreach(_.tock())
        allProfilers.foreach(p => println(s"Masks $p"))
      }

      evr
    }

    private def insideBinaryPass(batch: Batch[W], span: Int, events: CLEvent*) = {
      var ev = events
      if (span == 2) {
        ev = Seq(insideTT.doUpdates(batch, span, ev :_*))
      }

      ev = Seq(insideNT.doUpdates(batch, span, ev :_*))
      ev = Seq(insideTN.doUpdates(batch, span, ev :_*))
      ev = Seq(insideNN.doUpdates(batch, span, ev :_*))
      ev.head
    }

    private def outsideBinaryPass(batch: Batch[W], span: Int, events: CLEvent) = {
      var ev = events

      ev = outsideTN_R.doUpdates(batch, span, ev)
      ev = outsideNT_L.doUpdates(batch, span, ev)
      ev = outsideNN_L.doUpdates(batch, span, ev)
      ev = outsideNN_R.doUpdates(batch, span, ev)

      ev
    }

    private val insideBot = {(b: Batch[W], s: Int) =>  b.insideCharts(s).bot}
    private val insideTop = {(b: Batch[W], s: Int) =>  b.insideCharts(s).top}
    private val outsideBot = {(b: Batch[W], s: Int) =>  b.outsideCharts(s).bot}
    private val outsideTop = {(b: Batch[W], s: Int) =>  b.outsideCharts(s).top}

    private def insideTU = new UnaryUpdateManager(data.inside.insideTUKernels, devInside, insideTop, insideBot)
    private def insideNU = new UnaryUpdateManager(data.inside.insideNUKernels, devInside, insideTop, insideBot)

    private def insideTT = new BinaryUpdateManager(data.inside.insideTTKernels, true, devInside, devInside, devInside, insideBot, insideBot, insideBot, (b, e, l) => (b+1 to b+1))
    private def insideNT = new BinaryUpdateManager(data.inside.insideNTKernels, true, devInside, devInside, devInside, insideBot, insideTop, insideBot, (b, e, l) => (e-1 to e-1))
    private def insideTN = new BinaryUpdateManager(data.inside.insideTNKernels, true, devInside, devInside, devInside, insideBot, insideBot, insideTop, (b, e, l) => (b+1 to b+1))
    private def insideNN = new BinaryUpdateManager(data.inside.insideNNKernels, true, devInside, devInside, devInside, insideBot, insideTop, insideTop, (b, e, l) => (b+1 to e-1), trackRules)

    // here, "parentChart" is actually the left child, left is the parent, right is the right completion
    private def outsideTT_L = new BinaryUpdateManager(data.outside.outside_L_TTKernels, true, devOutside, devOutside, devInside, outsideBot, outsideBot, insideBot, (b, e, l) => (e+1 to e+1))
    private def outsideNT_L = new BinaryUpdateManager(data.outside.outside_L_NTKernels, false, devOutside, devOutside, devInside, outsideTop, outsideBot, insideBot, (b, e, l) => (e+1 to e+1))
    private def outsideTN_L = new BinaryUpdateManager(data.outside.outside_L_TNKernels, true, devOutside, devOutside, devInside, outsideBot, outsideBot, insideTop, (b, e, l) => (e+1 to l))
    private def outsideNN_L = new BinaryUpdateManager(data.outside.outside_L_NNKernels, false, devOutside, devOutside, devInside, outsideTop, outsideBot, insideTop, (b, e, l) => (e+1 to l))

    // here, "parentChart" is actually the right child, right is the parent, left is the left completion
    private def outsideTT_R = new BinaryUpdateManager(data.outside.outside_R_TTKernels, true, devOutside, devInside, devOutside, outsideBot, insideBot, outsideBot, (b, e, l) => (b-1 to b-1))
    private def outsideNT_R = new BinaryUpdateManager(data.outside.outside_R_NTKernels, true, devOutside, devInside, devOutside, outsideBot, insideTop, outsideBot, (b, e, l) => (0 to b-1))
    private def outsideTN_R = new BinaryUpdateManager(data.outside.outside_R_TNKernels, false, devOutside, devInside, devOutside, outsideTop, insideBot, outsideBot, (b, e, l) => (b-1 to b-1))
    private def outsideNN_R = new BinaryUpdateManager(data.outside.outside_R_NNKernels, false, devOutside, devInside, devOutside, outsideTop, insideTop, outsideBot, (b, e, l) => (0 to b-1), trackRules)

    private def outsideTU = new UnaryUpdateManager(data.outside.outsideTUKernels, devOutside, outsideBot, outsideTop)
    private def outsideNU = new UnaryUpdateManager(data.outside.outsideNUKernels, devOutside, outsideBot, outsideTop)

    def extractParses(batch: Batch[W], masks: DenseMatrix[Int], events: CLEvent*): ParSeq[BinarizedTree[C]] = {
      events.foreach(_.waitFor())
      val in = if (profile) System.currentTimeMillis() else 0L
      val trees = for (s <- 0 until batch.numSentences par) yield try {
        import batch.cellOffsets
        val length = batch.sentences(s).length
        val numCells = (cellOffsets(s+1)-cellOffsets(s))/2
        assert(numCells == TriangularArray.arraySize(length))
        val botBegin = cellOffsets(s)
        val botEnd = botBegin + numCells
        val topBegin = botEnd
        val topEnd = topBegin + numCells
        val bot = masks(::, botBegin until botEnd)
        val top = masks(::, topBegin until topEnd)

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
      if (profile) {
        println(s"Parse extraction took:  ${(out - in)/1000.0}s")
      }
      trees
    }


    private[CLParser] def getBatches(sentences: IndexedSeq[IndexedSeq[W]], masks: Option[DenseMatrix[Int]]): IndexedSeq[Batch[W]] = {
      val result = ArrayBuffer[Batch[W]]()
      var current = ArrayBuffer[IndexedSeq[W]]()
      var currentLengthTotal = 0
      var currentCellTotal = 0
      var offsetIntoMasksArray = 0
      for ( (s, i) <- sentences.zipWithIndex) {
        currentLengthTotal += s.length
        currentCellTotal += TriangularArray.arraySize(s.length) * 2
        if (currentLengthTotal > numWorkCells || currentCellTotal > numChartCells) {
          currentCellTotal -= TriangularArray.arraySize(s.length) * 2
          assert(current.nonEmpty)
          result += createBatch(current, masks.map(m => m(::, offsetIntoMasksArray until (offsetIntoMasksArray + currentCellTotal))))
          offsetIntoMasksArray += currentCellTotal
          currentLengthTotal = s.length
          currentCellTotal = TriangularArray.arraySize(s.length) * 2
          current = ArrayBuffer()
        }
        current += s
      }


      if (current.nonEmpty) result += createBatch(current, masks.map(m => m(::, offsetIntoMasksArray until (offsetIntoMasksArray + currentCellTotal))))
      result
    }

<<<<<<< HEAD
    private[CLParser] def createBatch(sentences: IndexedSeq[IndexedSeq[W]], masks: Option[DenseMatrix[Int]]): Batch[W] = {
      val lengthTotals = sentences.scanLeft(0)((acc, sent) => acc + sent.length)
      val cellTotals = sentences.scanLeft(0)((acc, sent) => acc + TriangularArray.arraySize(sent.length) * 2)
      println(f"Batch[W] size of ${sentences.length}, total length of ${lengthTotals.last}, total (inside) cells: ${cellTotals.last}, total inside ${cellTotals.last * myCellSize * 4.0/1024/1024}%.2fM  ")
      assert(masks.forall(_.cols == cellTotals.last), masks.map(_.cols) -> cellTotals.last)
      assert(cellTotals.last <= devInside.cols, cellTotals.last + " " +  devInside.cols)
      Batch[W](lengthTotals.toArray, cellTotals.toArray, sentences, devInside, devOutside, masks)
=======
    private[CLParser] def createBatch(sentences: IndexedSeq[IndexedSeq[W]], masks: PruningMask): Batch[W] = {
      val batch = Batch[W](sentences, devInside, devOutside, masks)
      println(f"Batch size of ${sentences.length}, ${batch.numCellsUsed} cells used, total inside ${batch.numCellsUsed * myCellSize * 4.0/1024/1024}%.2fM  ")
      batch
>>>>>>> acb0b4e... fix to Batch refactor.
    }

    private class UnaryUpdateManager(kernels: CLUnaryRuleUpdater,
                                     scoreMatrix: CLMatrix[Float],
                                     parentChart: (Batch[W],Int)=>ChartHalf,
                                     childChart: (Batch[W],Int)=>ChartHalf) {

      var offset = 0 // number of cells used so far.


      private def enqueue(batch: Batch[W], span: Int, parent: Int, left: Int, events: Seq[CLEvent]) = {
        lArray(offset) = left
        pArray(offset) = parent
        offset += 1
        if (offset >= numWorkCells)  {
          logger.debug(s"flush unaries!")
          flushQueue(batch, span, events)
        } else {
          events
        }
      }

      private def flushQueue(batch: Batch[W], span: Int, ev: Seq[CLEvent]) = {
        if (offset != 0) {
          val zz = zmk.shapedFill(devParent(0 until offset, ::), _zero, ev:_*) profileIn memFillEvents

          val bufArray = new Array[Int](offset * 3)
          System.arraycopy(pArray, 0, bufArray, 0, offset)
          System.arraycopy(lArray, 0, bufArray, offset, offset)
          val evx = offsetBuffer.buffer.writeArray(queue, bufArray, offset * 3, ev:_*) profileIn hdTransferEvents

          val wl = transposeCopy.permuteTransposeCopy(devLeft(0 until offset, ::), scoreMatrix, offsetBuffer.buffer, offset, offset, evx) profileIn transferEvents

          val endEvents = kernels.update(unaryEvents, devParent(0 until offset, ::), devLeft(0 until offset, ::),  wl, zz)

          val _ev = transposeCopy.permuteTransposeCopyOut(scoreMatrix, offsetBuffer.buffer, offset, devParent(0 until offset, ::), (evx +: endEvents):_*) profileIn sumToChartsEvents

          offset = 0
          Seq(_ev)
        } else {
          ev
        }
      }

      def doUpdates(batch: Batch[W], span: Int, events: CLEvent*) = {
        var ev = events

        for {
          sent <- 0 until batch.numSentences
          start <- 0 to batch.sentences(sent).length - span
          if batch.isAllowedSpan(sent, start, start + span)
        } {
          val end = start + span
          val parentTi = parentChart(batch, sent).cellOffset(start,end)
          val child = childChart(batch, sent).cellOffset(start,end)

          ev = enqueue(batch, span, parentTi, child, ev)
        }

        if (offset > 0) {
          flushQueue(batch, span, ev)
        }

        assert(ev.length == 1)
        ev.head
      }

    }

    private class BinaryUpdateManager(updater: CLBinaryRuleUpdater,
                                      parentIsBot: Boolean,
                                      parentChartMatrix: CLMatrix[Float],
                                      leftChartMatrix: CLMatrix[Float],
                                      rightChartMatrix: CLMatrix[Float],
                                      parentChart: (Batch[W],Int)=>ChartHalf,
                                      leftChart: (Batch[W],Int)=>ChartHalf,
                                      rightChart: (Batch[W],Int)=>ChartHalf,
                                      ranger: (Int, Int, Int)=>Range,
                                      trackRulesForThisSetOfRules: Boolean = false) {
      lazy val totalRulesInKernels = (updater.kernels.map(_.rules.length).sum)


       var splitPointOffset = 0 // number of unique parent spans used so far
       var offset = 0 // number of work cells used so far.

       // TODO: ugh, state
       var lastParent = -1

       private def enqueue(block: IndexedSeq[Int], batch: Batch[W], span: Int, parent: Int, left: Int, right: Int, events: Seq[CLEvent]): Seq[CLEvent] = {
         if (splitPointOffset == 0 || lastParent != parent) {
           splitPointOffsets(splitPointOffset) = offset
           splitPointOffset += 1
         }
         lastParent = parent
         pArray(offset) = parent
         lArray(offset) = left
         rArray(offset) = right

         if(profile && trackRulesForThisSetOfRules) {
           rulesEvaled += block.map(updater.kernels(_).rules.length).sum.toLong
         }

         offset += 1
         if (offset >= numWorkCells)  {
           println("flush?")
           flushQueue(block, batch, span, events)
         } else {
           events
         }
       }


       private def flushQueue(block: IndexedSeq[Int], batch: Batch[W], span: Int, ev: Seq[CLEvent]): Seq[CLEvent] = {
         if (offset != 0) {
           splitPointOffsets(splitPointOffset) = offset

           // copy ptrs to opencl

           val bufArray = new Array[Int](offset * 3)
           System.arraycopy(pArray, 0, bufArray, 0, offset)
           System.arraycopy(lArray, 0, bufArray, offset, offset)
           System.arraycopy(rArray, 0, bufArray, offset * 2, offset)

           val evx = offsetBuffer.buffer.writeArray(queue, bufArray, offset * 3, ev:_*)

           // do transpose based on ptrs
           val evTransLeft  = if(skipFineWork && batch.hasMasks) null else transposeCopy.permuteTransposeCopy(devLeft(0 until offset, ::), leftChartMatrix, offsetBuffer.buffer, offset, offset, evx) profileIn transferEvents
           val evTransRight = if(skipFineWork && batch.hasMasks) null else transposeCopy.permuteTransposeCopy(devRight(0 until offset, ::), rightChartMatrix, offsetBuffer.buffer, offset * 2, offset, evx) profileIn transferEvents

           val updateDirectToChart = updater.directWriteToChart
           
           // copy parent pointers
           // corresponding splits
           val evWriteDevSplitPoint =  if ((skipFineWork && batch.hasMasks) || updateDirectToChart) null else devSplitPointOffsets.writeArray(queue, splitPointOffsets, splitPointOffset + 1, ev:_*) profileIn hdTransferEvents

           val zeroParent = if((skipFineWork && batch.hasMasks) || updateDirectToChart) null else zmk.shapedFill(devParent(0 until offset, ::), _zero, ev:_*) profileIn memFillEvents

           val targetChart = if(updateDirectToChart) parentChartMatrix else devParent(0 until offset, ::)
           val kEvents = updater.update(block, binaryEvents,
             targetChart, offsetBuffer.buffer,
             devLeft(0 until offset, ::),
             devRight(0 until offset, ::),
             maskCharts, evTransLeft, evTransRight, evx)

           val sumEv: CLEvent = if((skipFineWork && batch.hasMasks) || updateDirectToChart) null else sumSplitPoints(span, Seq(evx, evWriteDevSplitPoint) ++ kEvents: _*)


           offset = 0
           splitPointOffset = 0
           if(sumEv eq null) kEvents else IndexedSeq(sumEv)
         } else {
           ev
         }
       }


       def sumSplitPoints(span: Int, events: CLEvent*): CLEvent = {
         val sumEv = data.util.sumSplitPoints(devParent,
           parentChartMatrix,
           offsetBuffer.buffer, splitPointOffset,
           devSplitPointOffsets,
           32 / span max 1, data.numSyms, events:_*) profileIn sumEvents
         sumEv
       }

       def doUpdates(batch: Batch[W], span: Int, events: CLEvent*) = {

         var ev = events
         splitPointOffset = 0
         lastParent = -1


         val merge = !batch.hasMasks || oldPruning

         val allSpans = if (batch.hasMasks) {
           val allSpans = for {
             sent <- 0 until batch.numSentences
             start <- 0 to batch.sentences(sent).length - span
             _ = total += 1
             mask <- if(parentIsBot) batch.botMaskFor(sent, start, start + span) else batch.topMaskFor(sent, start, start + span)
             if {val x = any(mask); if(!x) pruned += 1; x || doEmptySpans || trackRulesForThisSetOfRules }
           } yield (sent, start, start+span, mask)

           val ordered = orderSpansBySimilarity(allSpans)
           ordered
         } else {
           for {
             sent <- 0 until batch.numSentences
             start <- 0 to batch.sentences(sent).length - span
           } yield (sent, start, start+span, null)
         }



         val numBlocks = updater.numKernelBlocks


         val blocks = if(merge) IndexedSeq(0 until numBlocks) else (0 until numBlocks).groupBy(updater.kernels(_).parents.data.toIndexedSeq).values.toIndexedSeq

         for(block <- blocks) {
           val parentCounts = DenseVector.zeros[Int](data.maskSize * 32)
           val blockParents = updater.kernels(block.head).parents
           //        if(allSpans.head._4 != null)
           //          println(BitHacks.asBitSet(blockParents).cardinality)
           for ( (sent, start, end, mask) <- allSpans ) {
             if(profile && trackRulesForThisSetOfRules) {
               val numRules = block.map(updater.kernels(_).rules.length).sum
               val numSplits = ranger(start, start + span, batch.sentences(sent).length).filter(split => split >= 0 && split <= batch.sentences(sent).length).length
               rulesTotal += numSplits.toLong * numRules

             }
             if(mask == null || oldPruning || intersects(blockParents, mask)) {


               val splitRange = ranger(start, start + span, batch.sentences(sent).length)
               var split =  splitRange.start
               val splitEnd = splitRange.terminalElement
               val step = splitRange.step
               while (split != splitEnd) {
                 if (split >= 0 && split <= batch.sentences(sent).length) {
                   val end = start + span
                   val parentTi = parentChart(batch, sent).cellOffset(start,end)
                   val leftChildAllowed = if (split < start) batch.isAllowedSpan(sent,split, start) else batch.isAllowedSpan(sent, start, split)
                   val rightChildAllowed = if (split < end) batch.isAllowedSpan(sent,split,end) else batch.isAllowedSpan(sent, end, split)

                   if (doEmptySpans || (leftChildAllowed && rightChildAllowed)) {
                     val leftChild = if (split < start) leftChart(batch, sent).cellOffset(split,start) else leftChart(batch, sent).cellOffset(start, split)
                     val rightChild = if (split < end) rightChart(batch, sent).cellOffset(split,end) else rightChart(batch, sent).cellOffset(end, split)
                     ev = enqueue(block, batch, span, parentTi, leftChild, rightChild, ev)
                     if(profile && trackRules && mask != null) {
                       val mask2 = BitHacks.asBitSet(mask)
                       for(m <- mask2.iterator) {
                        parentCounts(m) += 1
                       }
                     }
                   }
                 }
                 split += step
               }

             }

           }

           if(trackRulesForThisSetOfRules && profile)
            theoreticalRules += block.flatMap(updater.kernels(_).rules).groupBy(_.parent.coarse).map { case (k,v) => parentCounts(k) * v.length}.sum

           if (offset > 0) {
             ev = flushQueue(block, batch, span, ev)
           }
         }


//         if(profile  && trackRulesForThisSetOfRules) {
//           rulesTotal += {for {
//             sent <- 0 until batch.numSentences
//             start <- 0 to batch.sentences(sent).length - span
//             split <- ranger(start, start + span, batch.sentences(sent).length)
//             if (split >= 0 && split <= batch.sentences(sent).length)
//           } yield split}.sum.toLong * totalRulesInKernels
//         }

         assert(ev.length == 1)
         ev.head
       }

     }

  }



  def intersects(blockMask: DenseVector[Int], spanMask: DenseVector[Int]):Boolean = {
    var i = 0
    assert(blockMask.length == spanMask.length)
    while(i < blockMask.length) {
      if( (blockMask(i) & spanMask(i)) != 0) return true
      i += 1
    }

    false

  }






  // Sentence, Begin, End, BitMask
  def orderSpansBySimilarity(spans: IndexedSeq[(Int, Int, Int, DenseVector[Int])]): IndexedSeq[(Int, Int, Int, DenseVector[Int])] = {
//    val res = spans.groupBy(v =>v._4.toArray.toIndexedSeq).values.flatten.toIndexedSeq
//    res
    if(oldPruning) {
      import BitHacks.OrderBitVectors.OrderingBitVectors
      val in = System.currentTimeMillis()
      val res = spans.sortBy(_._4)
      val out = System.currentTimeMillis()
      sortTime += (out - in)
      res
    } else {
      spans

    }
  }


}

object CLParser extends Logging {

  case class Params(annotator: TreeAnnotator[AnnotatedLabel, String, AnnotatedLabel] = Xbarize(),
                    device: String = "nvidia",
                    profile: Boolean = false,
                    numToParse: Int = 1000, codeCache: File = new File("grammar.grz"), cache: Boolean = true,
                    maxParseLength: Int = 10000,
                    jvmParse: Boolean = false, parseTwice: Boolean = false,
                    textGrammarPrefix: String = null,
                    checkPartitions: Boolean = false,
                    justInsides: Boolean = false,
                    mem: String = "1g")

  def main(args: Array[String]) = {
    import ParserParams.JointParams

    val params = CommandLineParser.readIn[JointParams[Params]](args)
    val myParams:Params = params.trainer
    import myParams._


    implicit val context: CLContext = {
      val (good, bad) = JavaCL.listPlatforms().flatMap(_.listAllDevices(true)).partition(d => device.r.findFirstIn(d.toString.toLowerCase()).nonEmpty)
      println(good.toIndexedSeq)
      println(bad.toIndexedSeq)
      if(good.isEmpty) {
        JavaCL.createContext(Collections.emptyMap(), bad.sortBy(d => d.toString.toLowerCase().contains("geforce")).last)
      } else {
        JavaCL.createContext(Collections.emptyMap(), good.head)
      }

    }
    println(context)

    println("Training Parser...")
    println(params)
    val transformed = params.treebank.copy(binarization="left", keepUnaryChainsFromTrain = false).trainTrees
    val gold = params.treebank.trainTrees.filter(_.words.length <= maxParseLength).take(numToParse)
    val toParse =  gold.map(_.words)

    val grammars: IndexedSeq[SimpleRefinedGrammar[AnnotatedLabel, AnnotatedLabel, String]] = if (textGrammarPrefix == null) {
      IndexedSeq(GenerativeParser.annotated(annotator, transformed))
    } else {
      textGrammarPrefix.split(":").map(SimpleRefinedGrammar.parseBerkeleyText(_, -12, CloseUnaries.None))
    }

    val grammar = grammars.last


    var parserData:CLParserData[AnnotatedLabel, AnnotatedLabel, String] = if (cache && codeCache != null && codeCache.exists()) {
      CLParserData.read(new ZipFile(codeCache))
    } else {
      null
    }

    val defaultGenerator = GenType.VariableLength
    val prunedGenerator = GenType.VariableLength

    if (parserData == null || parserData.grammar.signature != grammar.signature) {
      println("Regenerating parser data")
      val gen = if(grammars.length > 1) prunedGenerator else defaultGenerator
      parserData =  CLParserData.make(grammar, gen, grammars.length > 1)
      if (cache && codeCache != null) {
        parserData.write(new BufferedOutputStream(new FileOutputStream(codeCache)))
      }
    }

    val allData = grammars.dropRight(1).map(CLParserData.make(_,  defaultGenerator, false)) :+ parserData

    val kern = {
      fromParserDatas[AnnotatedLabel, AnnotatedLabel, String](allData, profile, parseMemString(mem))
    }

    if (justInsides) {
      val partsX = logTime("CL Insides", toParse.length)( kern.partitions(toParse))
      println(partsX)
      System.exit(0)
    }

    if (checkPartitions) {
      val partsX = logTime("CL Insides", toParse.length)( kern.partitions(toParse))
      println(partsX)
      val parser = Parser(grammar, if (kern.isViterbi) new ViterbiDecoder[AnnotatedLabel, String] else new MaxConstituentDecoder[AnnotatedLabel, String])
      val parts2 = toParse.par.map(parser.marginal(_).logPartition)
      println(parts2)
      println("max difference: " + (DenseVector(partsX.map(_.toDouble):_*) - DenseVector(parts2.seq:_*)).norm(Double.PositiveInfinity))
      System.exit(0)
    }

    val trees = logTime("CL Parsing:", toParse.length)(kern.parse(toParse))
    println(eval(trees zip gold.map(_.tree)))
    //println(trees zip toParse map {case (k,v) if k != null => k render v; case (k,v) => ":("})
    if (parseTwice) {
      val trees = logTime("CL Parsing x2:", toParse.length)(kern.parse(toParse))
      println(eval(trees zip gold.map(_.tree)))
    }
    if (jvmParse) {
      val parser = if(grammars.length > 1) {
        Parser(new ConstraintCoreGrammarAdaptor(grammar.grammar, grammar.lexicon, new ParserChartConstraintsFactory(Parser(grammars.head, new ViterbiDecoder[AnnotatedLabel, String]), (_:AnnotatedLabel).isIntermediate)),
          grammar,
          if (kern.isViterbi) new ViterbiDecoder[AnnotatedLabel, String] else new MaxConstituentDecoder[AnnotatedLabel, String])
      } else {
        Parser(grammar, if (kern.isViterbi) new ViterbiDecoder[AnnotatedLabel, String] else new MaxConstituentDecoder[AnnotatedLabel, String])
      }
      val margs = logTime("JVM Parse", toParse.length) {
        toParse.par.map { w =>
          val m = parser.apply(w)
          m
        }.seq.toIndexedSeq
      }
      println(eval(margs zip gold.map(_.tree)))
    }

    kern.release()
    context.release()

  }


  def fromParserData[L, L2, W](data: CLParserData[L, L2, W], profile: Boolean, mem: Long)(implicit context: CLContext): CLParser[L, L2, W] = {
    fromParserDatas(IndexedSeq(data), profile, mem)
  }

  def fromParserDatas[L, L2, W](data: IndexedSeq[CLParserData[L, L2, W]], profile: Boolean, mem: Long)(implicit context: CLContext): CLParser[L, L2, W] = {
    val kern = new CLParser[L, L2, W](data, profile = profile, maxAllocSize = mem)
    kern
  }

  def eval(trees: IndexedSeq[(BinarizedTree[AnnotatedLabel], BinarizedTree[AnnotatedLabel])]) = {
    val chainReplacer = AnnotatedLabelChainReplacer
    val eval: ParseEval[String] = new ParseEval(Set("","''", "``", ".", ":", ",", "TOP"))
    trees filter (_._1 ne null) map { case (guess, gold) =>
      val tree: Tree[String] = chainReplacer.replaceUnaries(guess).map(_.label)
      val guessTree = Trees.debinarize(Trees.deannotate(tree))
      val deBgold: Tree[String] = Trees.debinarize(Trees.deannotate(chainReplacer.replaceUnaries(gold).map(_.label)))
      eval.apply(guessTree, deBgold)
    } reduceLeft (_ + _)
  }

  def parseMemString(x: String) = x.last.toLower match {
    case 'g' => Math.scalb(x.dropRight(1).toDouble, 30).toLong
    case 'm' => Math.scalb(x.dropRight(1).toDouble, 20).toLong
    case 'k' => Math.scalb(x.dropRight(1).toDouble, 10).toLong
    case y:Char if y.isLetterOrDigit => throw new RuntimeException(s"bad mem string: $x")
    case _ => x.toLong

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
  def make[C, L, W](grammar: SimpleRefinedGrammar[C, L, W], genType: GenType, directWrite: Boolean)(implicit context: CLContext) = {
    implicit val viterbi = ViterbiRuleSemiring
    val ruleScores: Array[Float] = Array.tabulate(grammar.refinedGrammar.index.size){r =>
      val score = grammar.ruleScoreArray(grammar.refinements.rules.project(r))(grammar.refinements.rules.localize(r))
      viterbi.fromLogSpace(score.toFloat)
    }
    val structure = new RuleStructure(grammar.refinements, grammar.refinedGrammar, ruleScores)
    val inside = CLInsideKernels.make(structure, directWrite, genType)
    val outside =  CLOutsideKernels.make(structure, directWrite, genType)
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
