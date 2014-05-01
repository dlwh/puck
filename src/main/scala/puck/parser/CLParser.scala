package puck
package parser

import gen._
import com.nativelibs4java.opencl._
import com.typesafe.scalalogging.slf4j.LazyLogging
import puck.util._
import puck.linalg.CLMatrix
import puck.linalg.kernels.CLMatrixTransposeCopy
import breeze.collection.mutable.TriangularArray
import breeze.linalg.{DenseVector, DenseMatrix}
import epic.trees.annotations.TreeAnnotator
import epic.parser._
import breeze.config.CommandLineParser
import epic.trees._
import java.io._
import java.util.zip.{ZipEntry, ZipFile, ZipOutputStream}
import scala.collection.mutable.ArrayBuffer
import epic.parser.SimpleRefinedGrammar.CloseUnaries
import epic.parser.projections.{ProjectionIndexer, GrammarRefinements, ParserChartConstraintsFactory, ConstraintCoreGrammarAdaptor}
import scala.collection.parallel.immutable.ParSeq
import java.util.Collections
import epic.trees.UnaryTree
import epic.parser.ViterbiDecoder
import epic.trees.NullaryTree
import epic.trees.BinaryTree
import epic.trees.annotations.Xbarize
import epic.trees.Span
import scala.io.{Codec, Source}
import epic.lexicon.Lexicon
import breeze.util.Index
import org.bridj.Pointer
import scala.reflect.ClassTag
import scala.concurrent.{ExecutionContext, Await, Future}
import scala.concurrent.duration.Duration
import scala.collection.mutable


/**
 * TODO
 *
 * @author dlwh
 **/
class CLParser[C, L, W](data: IndexedSeq[CLParserData[C, L, W]],
                        maxAllocSize: Long = 1<<30,
                        profile: Boolean = true)(implicit val context: CLContext) extends LazyLogging {

  def parse(sentences: IndexedSeq[IndexedSeq[W]]):IndexedSeq[BinarizedTree[C]] = synchronized {
    val mask = computeMasks(sentences)
    parsers.last.parse(sentences, mask)
  }

  def partitions(sentences: IndexedSeq[IndexedSeq[W]]):IndexedSeq[Float] = synchronized {
    val mask = computeMasks(sentences)
    parsers.last.partitions(sentences, mask)
  }

  def insideOutside(sentences: IndexedSeq[IndexedSeq[W]]):Unit = synchronized {
    val mask = computeMasks(sentences)
    parsers.last.insideOutside(sentences, mask)
  }

  private def computeMasks(sentences: IndexedSeq[IndexedSeq[W]]): PruningMask = {
    parsers.dropRight(1).foldLeft(NoPruningMask:PruningMask)( (a,b) => b.updateMasks(sentences, a))
  }

  private implicit val queue = if (profile) context.createDefaultProfilingQueue() else context.createDefaultQueue()

  def isViterbi = data.last.isViterbi

  private val profiler = new CLProfiler(profile)

  private val initMemFillEvents  = profiler.eventTimer("initMemfill")
  private val memFillEvents  = profiler.eventTimer("memfill")
  private val hdTransferEvents  = profiler.eventTimer("Host2Dev Transfer")
  private val transferEvents  = profiler.eventTimer("Transfer")
  private val binaryEvents  = profiler.eventTimer("Binary")
  private val unaryEvents  = profiler.eventTimer("Unary")
  private val unarySumEvents  = profiler.eventTimer("Unary Sum")
  private val posEvents  = profiler.eventTimer("POS")
  private val masksEvents  = profiler.eventTimer("Masks")

  // TODO:

  def release() {
    queue.release()
  }


  private val parsers = (0 until data.length).map(i => new ActualParser(data(i), i < data.length - 1 && data(i + 1).isScaling))

  // other stuff
  private val zmk = ZeroMemoryKernel()
  private val transposeCopy = CLMatrixTransposeCopy()

  var pruned = 0
  var total = 0
  var totalSyms = 0

  private class ActualParser(val data: CLParserData[C, L, W], nextParserNeedsScales: Boolean) {
    import data._

    type TagFuture =  Future[(DenseMatrix[Float], Array[Int])]

    def pretagger(b: Iterator[Batch[W]]):Iterator[(Batch[W],  TagFuture)] = new Iterator[(Batch[W], TagFuture)] {
      var n: Option[(Batch[W], TagFuture) ] = None

      hasNext


      override def next(): (Batch[W], TagFuture) = {
        hasNext
        val q = n.get
        n = None
        hasNext
        q
      }

      override def hasNext: Boolean = {
        n.nonEmpty || {
          if(b.hasNext) {
            val bb = b.next
            n = Some(bb -> pretag(bb))
            true
          } else {
            false
          }

        }



      }
    }

    def pretag(b: Batch[W]): TagFuture = {
      import ExecutionContext.Implicits.global
      Future {
        computeTagScores(b, b.devInside.rows)
      }
    }

    def parse(sentences: IndexedSeq[IndexedSeq[W]], mask: PruningMask) = synchronized {
      logTime("parse", sentences.length) {
        withWorkSpace(mask.hasMasks) {workspace =>
          pretagger(workspace.getBatches(sentences, mask).iterator).flatMap { case (batch, tagFuture) =>
            var ev = inside(workspace, batch, tagFuture)

            if(data.isViterbi) {
              val ev3 = computeViterbiParts(workspace, batch, ev)
              Option(ev3).foreach(_.waitFor())
              val parseMask: DenseMatrixMask = extractMasks(workspace, batch, false)
              val parses = extractParses(batch, parseMask.matrix, ev3)
              parses
            } else {
              ev = outside(workspace, batch, ev)
              val ev3 = computeMBRParts(workspace, batch, ev)
              Option(ev3).foreach(_.waitFor())
              val parseMask = extractMasks(workspace, batch, false)
              val parses = extractMBRParses(workspace, batch, parseMask, ev3)
              parses
            }
          }.toIndexedSeq
        }
      }

    }

    def insideOutside(sentences: IndexedSeq[IndexedSeq[W]], mask: PruningMask) = synchronized {
      logTime("parse", sentences.length) {
        withWorkSpace(mask.hasMasks) { workspace =>
          pretagger(workspace.getBatches(sentences, mask).iterator).foreach { case (batch, tf) =>
            var ev = inside(workspace, batch, tf)
            ev = outside(workspace, batch, ev)
            val ev3 = if(data.isViterbi) computeViterbiParts(workspace, batch, ev) else computeMBRParts(workspace, batch, ev)
            Option(ev3).foreach(_.waitFor())
          }
        }
      }
    }

    private def withWorkSpace[B](hasMasks: Boolean)(w: WorkSpace=>B) = {
      val myCellSize:Int = roundUpToMultipleOf(data.numSyms, 32)
      resource.managed(WorkSpace.allocate(myCellSize, data.maskSize, maxAllocSize, if(hasMasks) 8 else 7)).acquireAndGet(w)
    }

    def partitions(sentences: IndexedSeq[IndexedSeq[W]], mask: PruningMask) = synchronized {
      logTime("partitions", sentences.length) {
         withWorkSpace(mask.hasMasks) {workspace =>
           pretagger(workspace.getBatches(sentences, mask).iterator).flatMap { case (batch, tf) =>
            var ev = inside(workspace, batch, tf)
            val dest = context.createFloatBuffer(CLMem.Usage.Output, sentences.length)
            ev = workspace.pPtrBuffer.writeArray(queue, batch.rootIndices.toArray, batch.numSentences, ev) profileIn hdTransferEvents
            ev = data.util.getRootScores(dest, workspace.devInside, workspace.pPtrBuffer, batch.numSentences, structure.root, ev)
            if(semiring.needsScaling) {
              val scaledScores = dest.read(queue, ev).getFloats(batch.numSentences)
              for(i <- 0 until batch.numSentences) yield semiring.toLogSpace(scaledScores(i), batch.masks.insideTopScaleFor(i, 0, batch.lengths(i))) + batch.partitionScales(i).toFloat
            } else {
              dest.read(queue, ev).getFloats(batch.numSentences)
            }
          }.toIndexedSeq
        }
      }
    }

    def updateMasks(sentences: IndexedSeq[IndexedSeq[W]], mask: PruningMask): PruningMask = synchronized {
      logTime("masks", sentences.length) {
        withWorkSpace(mask.hasMasks) {workspace =>
           val masks = pretagger(workspace.getBatches(sentences, mask).iterator).map {  case (batch, tf) =>
            workspace.maskCharts.assignAsync(-1).waitFor()
            val mask = computePruningMasks(workspace, batch, tf):PruningMask
            mask
          }.toIndexedSeq
          val newMasks = masks.reduceLeft(_ ++ _)

          newMasks
        }
      }
    }

    def computePruningMasks(workspace: WorkSpace, batch: Batch[W], tf: TagFuture, ev: CLEvent*): DenseMatrixMask = {
      val insideEvents = inside(workspace, batch, tf, ev:_*)
      val outsideEvents = outside(workspace, batch, insideEvents)
      val ev2 = computeMasks(workspace, batch, -9, outsideEvents)
      ev2.waitFor()
      extractMasks(workspace, batch, true)
   }


    def extractMasks(workspace: WorkSpace, batch: Batch[W], updateScales: Boolean): DenseMatrixMask = {
      import workspace._
      val denseMasks = maskCharts(::, 0 until batch.numCellsUsed).toDense
      maskCharts.data.waitUnmap()
      if(nextParserNeedsScales) {
        val ptrScale = Pointer.allocateFloats(devInsideScale.getElementCount)
        val insideEv = if(updateScales) data.scaling.getScaling(devInsideScale, devInside(::, 0 until batch.numCellsUsed)) else null
        devInsideScale.read(queue, 0, batch.numCellsUsed, ptrScale, true, insideEv)
        val inside = new DenseVector(ptrScale.getFloats(batch.numCellsUsed))
        val outsideEv = if(updateScales) data.scaling.getScaling(devOutsideScale, devOutside(::, 0 until batch.numCellsUsed)) else null
        devOutsideScale.read(queue, 0, batch.numCellsUsed, ptrScale, true, outsideEv)
        val outside = new DenseVector(ptrScale.getFloats(batch.numCellsUsed))
        ptrScale.release()
        new DenseMatrixMask(denseMasks, inside, outside, batch.lengths, batch.cellOffsets)
      } else {
        new DenseMatrixMask(denseMasks, DenseVector.zeros[Float](batch.numCellsUsed), DenseVector.zeros[Float](batch.numCellsUsed), batch.lengths, batch.cellOffsets)

      }
    }



    private def zero: Float = data.semiring.zero

    def inside(workspace: WorkSpace, batch: Batch[W], tagFuture: TagFuture, events: CLEvent*):CLEvent = synchronized {
      import workspace._
      profiler.clear()
      profiler.tick()
      pruned = 0
      total = 0
      totalSyms = 0

      val evZeroCharts = zmk.fillMemory(devInside.data, zero, events:_*) profileIn initMemFillEvents
      val evZeroOutside = zmk.fillMemory(devOutside.data, zero, events:_*) profileIn initMemFillEvents

      var ev = evZeroCharts

      if(batch.hasMasks && semiring.needsScaling) {
        ev = devInsideScale.write(queue, batch.masks.getIScales, false, ev) profileIn hdTransferEvents
        ev = devOutsideScale.write(queue, batch.masks.getOScales, false, ev) profileIn hdTransferEvents
        queue.finish()
      }


        val (tagScores, pArray) =  Await.result ( tagFuture, Duration.Inf )

        ev = writeTagScores(workspace, pArray, tagScores, ev, evZeroOutside)

      val insideTU = new UnaryUpdateManager(data.inside.insideTUKernels, devInside, devInsideScale, devInsideScale, insideTop, insideBot)

      ev = insideTU.doUpdates(workspace, batch, 1, ev)

      for (span <- 2 to batch.maxLength) {
        print(s"$span ")
        ev = insideBinaryPass(workspace, batch, span, ev)

        val insideNU = new UnaryUpdateManager(data.inside.insideNUKernels, devInside, devInsideScale, devInsideScale, insideTop, insideBot)
        ev = insideNU.doUpdates(workspace, batch, span, ev)

      }

//      if(CLParser.this.data.last eq this.data) {
//        queue.finish()
//        println("=======")
//        println(batch.insideCharts(2).bot.cellString(4, 5, structure, data.semiring, batch.masks.insideScaleFor(2, _, _)))
//        println("-------")
//        println(batch.insideCharts(1).top.toString(structure, _zero))
//        println("=======")
//      }

      if (profile) {
        queue.finish()
        profiler.tock()
        println(profiler.report("inside"))
        println(s"Enqueuing writes took ${writeTimer.clear()}s")
        println(s"Queuing took ${queueTimer.clear()}s")
        println(s"Spin up for writes took ${allTimer.clear()}s")
        println(s"Pruned $pruned/$total")
        println(s"Syms per cell: ${totalSyms * 1.0/(total - pruned)}")
      }

      ev
    }

    def outside(workspace: WorkSpace, batch: Batch[W], event: CLEvent):CLEvent = synchronized {
      import workspace._
      var ev = event
      profiler.clear()
      profiler.tick()
      pruned  = 0
      total = 0
      totalSyms = 0

     ev = pPtrBuffer.writeArray(queue, batch.outsideRootIndices, batch.numSentences, ev) profileIn hdTransferEvents
      ev = data.util.setRootScores(devOutside, pPtrBuffer, batch.numSentences, structure.root, data.semiring.one, ev) profileIn memFillEvents

      val outsideNU = new UnaryUpdateManager(data.outside.outsideNUKernels, devOutside, devOutsideScale, devOutsideScale, outsideBot, outsideTop)
      ev = outsideNU.doUpdates(workspace: WorkSpace, batch, batch.maxLength, ev)

      for (span <- (batch.maxLength - 1) to 1 by -1) {
        print(s"$span ")
        ev = outsideBinaryPass(workspace, batch, span, ev)
        if (span == 1) {
          val outsideTU = new UnaryUpdateManager(data.outside.outsideTUKernels, devOutside, devOutsideScale, devOutsideScale, outsideBot, outsideTop)
          ev = outsideTU.doUpdates(workspace, batch, span, ev)
        } else {
          val outsideNU = new UnaryUpdateManager(data.outside.outsideNUKernels, devOutside, devOutsideScale, devOutsideScale, outsideBot, outsideTop)
          ev = outsideNU.doUpdates(workspace, batch, span, ev)
        }

      }

      // here, "parentChart" is actually the left child, left is the parent, right is the right completion
      val outsideTT_L = new BinaryUpdateManager(data.outside.outside_L_TTKernels, true, devOutside, devOutsideScale, devOutside, devOutsideScale, devInside, devInsideScale, outsideBot, outsideBot, insideBot, (b, e, l) => (e+1 to e+1))
      val outsideTN_L = new BinaryUpdateManager(data.outside.outside_L_TNKernels, true, devOutside, devOutsideScale, devOutside, devOutsideScale, devInside, devInsideScale, outsideBot, outsideBot, insideTop, (b, e, l) => (e+1 to l))

      // here, "parentChart" is actually the right child, right is the parent, left is the left completion
      val outsideTT_R = new BinaryUpdateManager(data.outside.outside_R_TTKernels, true, devOutside, devOutsideScale, devInside, devInsideScale, devOutside, devOutsideScale, outsideBot, insideBot, outsideBot, (b, e, l) => (b-1 to b-1))
      val outsideNT_R = new BinaryUpdateManager(data.outside.outside_R_NTKernels, true, devOutside, devOutsideScale, devInside, devInsideScale, devOutside, devOutsideScale, outsideBot, insideTop, outsideBot, (b, e, l) => (0 to b-1))

      ev = outsideTT_L.doUpdates(workspace, batch, 1, ev)
      ev = outsideNT_R.doUpdates(workspace, batch, 1, ev)
      ev = outsideTN_L.doUpdates(workspace, batch, 1, ev)
      ev = outsideTT_R.doUpdates(workspace, batch, 1, ev)
      profiler.tock()

      if (profile) {
        queue.finish()
        Thread.sleep(15)
        println(profiler.report("outside"))
        println(s"Enqueuing writes took ${writeTimer.clear()}s")
        println(s"Spin up for writes took ${allTimer.clear()}s")
        println(s"Pruned $pruned/$total")
        println(s"Syms per cell: ${totalSyms * 1.0/(total - pruned)}")
      }

      ev
    }


    private def computeMBRParts(workspace: WorkSpace, batch: Batch[W], events: CLEvent*):CLEvent = synchronized {
      import workspace._
      if(profile) {
        profiler.clear()
        profiler.tick()
      }

      val evr = data.mbr.getMasks(maskCharts(::, 0 until batch.numCellsUsed),
        devInside(::, 0 until batch.numCellsUsed),
        devOutside(::, 0 until batch.numCellsUsed),
        batch.cellOffsets, batch.lengths, structure.root, events:_*) profileIn masksEvents
      if (profile) {
        profiler.tock()
        queue.finish()
        println(profiler.report("mbr"))
      }

      evr
    }

    private def computeViterbiParts(workspace: WorkSpace, batch: Batch[W], events: CLEvent*):CLEvent = synchronized {
      import workspace._
      if(profile) {
        profiler.clear()
        profiler.tick()
      }

      val evr = data.viterbi.viterbi(structure, maskCharts(::, 0 until batch.numCellsUsed),
        devInside(::, 0 until batch.numCellsUsed),
        batch.cellOffsets, batch.lengths, structure.root, events:_*) profileIn masksEvents
      if (profile) {
        queue.finish()
        profiler.tock()
        println(profiler.report("viterbi"))
      }

      evr
    }

    def initializeTagScores(workspace: WorkSpace, batch: Batch[W], events: CLEvent*) = {
      val cellSize = batch.devInside.rows
      val totalLength = batch.totalLength

      val (tagScores, pArray) = computeTagScores(batch, cellSize)

      writeTagScores(workspace, pArray, tagScores, events:_*)
    }


    def writeTagScores(workspace: WorkSpace, pArray: Array[Int], tagScores: DenseMatrix[Float], events: CLEvent*): CLEvent = {
      val totalLength = pArray.length
      import workspace._
      val ev2 = pPtrBuffer.writeArray(queue, pArray, totalLength, events: _*) profileIn posEvents
      var offset = 0
      var ev = ev2
      while (offset < totalLength) {
        val toDoThisTime = math.min(devLeft.rows, totalLength - offset)
        val evW = devLeft(0 until toDoThisTime, ::).writeFrom(tagScores(offset until offset + toDoThisTime, ::), false, ev) map (_ profileIn posEvents)
        ev = transposeCopy.permuteTransposeCopyOut(devInside, pPtrBuffer, offset, toDoThisTime, devLeft(0 until toDoThisTime, ::), evW: _*) profileIn posEvents
        offset += toDoThisTime
      }
      ev
    }

    def computeTagScores(batch: Batch[W], cellSize: Int): (DenseMatrix[Float], Array[Int]) = {
      val totalLength = batch.totalLength
      val pArray = new Array[Int](totalLength)
      val tagScores = DenseMatrix.zeros[Float](totalLength, cellSize)
      tagScores := zero
      for (i <- (0 until batch.numSentences).par) {
        val sent = batch.sentences(i)
        val anch = data.grammar.tagScorer.anchor(sent)
        val lexAnch = data.grammar.lexicon.anchor(sent)
        var scoreScale = 0.0
        var offset = batch.lengthOffsets(i)
        for (pos <- 0 until sent.length) {
          val myScale = if (semiring.needsScaling) math.exp(-batch.masks.insideScaleFor(i, pos, pos + 1)) else 1.0
          var maxScore = zero
          for (t <- lexAnch.allowedTags(pos); ref <- data.grammar.refinements.labels.refinementsOf(t)) {
            val index = ref
            val score = anch.scoreTag(pos, data.grammar.refinedGrammar.labelIndex.get(index))
            val gpuIndex = data.structure.labelIndexToTerminal(index)
            //        if(pos == 0) println(pos,t,ref,data.grammar.refinements.labels.fineIndex.get(ref), gpuIndex,score)
            pArray(offset) = batch.insideBotCell(i, pos, pos + 1)
            tagScores(offset, gpuIndex) = data.semiring.fromLogSpace(score.toFloat) * myScale.toFloat
            maxScore = maxScore + tagScores(offset, gpuIndex)
          }

          if (semiring.needsScaling) {
            scoreScale += math.log(maxScore / 2)

            for (t <- lexAnch.allowedTags(pos); ref <- data.grammar.refinements.labels.refinementsOf(t)) {
              val index = ref
              val gpuIndex = data.structure.labelIndexToTerminal(index)
              tagScores(offset, gpuIndex) /= (maxScore / 2)
            }

          }


          offset += 1
        }
        batch.partitionScales(i) = scoreScale
      }
      (tagScores, pArray)
    }

    private def computeMasks(workspace: WorkSpace, batch: Batch[W], threshold: Float, events: CLEvent*):CLEvent = synchronized {
      import workspace._
      if(profile) {
        profiler.clear()
        profiler.tick()
      }

      val evr = data.masks.getMasks(maskCharts(::, 0 until batch.numCellsUsed),
        devInside(::, 0 until batch.numCellsUsed),
        devOutside(::, 0 until batch.numCellsUsed),
        batch.cellOffsets, batch.lengths, structure.root, threshold, events:_*) profileIn masksEvents
      if (profile) {
        queue.finish()
        profiler.tock()
        println(profiler.report("masks"))
      }

      evr
    }

    private def insideBinaryPass(workspace: WorkSpace, batch: Batch[W], span: Int, events: CLEvent*) = {
      var ev = events
      import workspace._
      val insideNT = new BinaryUpdateManager(data.inside.insideNTKernels, true, devInside, devInsideScale, devInside, devInsideScale, devInside, devInsideScale, insideBot, insideTop, insideBot, (b, e, l) => (e-1 to e-1))
      val insideTN = new BinaryUpdateManager(data.inside.insideTNKernels, true, devInside, devInsideScale, devInside, devInsideScale, devInside, devInsideScale, insideBot, insideBot, insideTop, (b, e, l) => (b+1 to b+1))
      val insideNN = new BinaryUpdateManager(data.inside.insideNNKernels, true, devInside, devInsideScale, devInside, devInsideScale, devInside, devInsideScale, insideBot, insideTop, insideTop, (b, e, l) => (b+1 to e-1))
      if (span == 2) {
        val insideTT = new BinaryUpdateManager(data.inside.insideTTKernels, true, devInside, devInsideScale, devInside, devInsideScale, devInside, devInsideScale, insideBot, insideBot, insideBot, (b, e, l) => (b+1 to b+1))
        ev = Seq(insideTT.doUpdates(workspace, batch, span, ev :_*))
      }

      ev = Seq(insideNT.doUpdates(workspace, batch, span, ev :_*))
      ev = Seq(insideTN.doUpdates(workspace, batch, span, ev :_*))
      ev = Seq(insideNN.doUpdates(workspace, batch, span, ev :_*))
      ev.head
    }

    private def outsideBinaryPass(workspace: WorkSpace, batch: Batch[W], span: Int, events: CLEvent) = {
      var ev = events

      import workspace._

      val outsideTN_R = new BinaryUpdateManager(data.outside.outside_R_TNKernels, false, devOutside, devOutsideScale, devInside, devInsideScale, devOutside, devOutsideScale, outsideTop, insideBot, outsideBot, (b, e, l) => (b-1 to b-1))

      ev = outsideTN_R.doUpdates(workspace, batch, span, ev)
      val outsideNT_L = new BinaryUpdateManager(data.outside.outside_L_NTKernels, false, devOutside, devOutsideScale, devOutside, devOutsideScale, devInside, devInsideScale, outsideTop, outsideBot, insideBot, (b, e, l) => (e+1 to e+1))
      ev = outsideNT_L.doUpdates(workspace, batch, span, ev)
      val outsideNN_L = new BinaryUpdateManager(data.outside.outside_L_NNKernels, false, devOutside, devOutsideScale, devOutside, devOutsideScale, devInside, devInsideScale, outsideTop, outsideBot, insideTop, (b, e, l) => (e+1 to l))
      ev = outsideNN_L.doUpdates(workspace, batch, span, ev)
      val outsideNN_R = new BinaryUpdateManager(data.outside.outside_R_NNKernels, false, devOutside, devOutsideScale, devInside, devInsideScale, devOutside, devOutsideScale, outsideTop, insideTop, outsideBot, (b, e, l) => (0 to b-1))
      ev = outsideNN_R.doUpdates(workspace, batch, span, ev)

      ev
    }

    private val insideBot = {(b: Batch[W], s: Int) =>  b.insideBotOffset(s)}
    private val insideTop = {(b: Batch[W], s: Int) =>  b.insideTopOffset(s) }
    private val outsideBot = {(b: Batch[W], s: Int) =>  b.outsideBotOffset(s) }
    private val outsideTop = {(b: Batch[W], s: Int) =>  b.outsideTopOffset(s) }

    def extractParses(batch: Batch[W], mask: DenseMatrix[Int], events: CLEvent*): ParSeq[BinarizedTree[C]] = {
      CLEvent.waitFor(events:_*)
      val in = if (profile) System.currentTimeMillis() else 0L
      val trees = for (s <- 0 until batch.numSentences par) yield try {
        val length = batch.sentences(s).length
        val cellOffset = batch.cellOffsets(s)
        val treeArray = mask(::, cellOffset until (cellOffset + 2 * length - 1))

        def top(cell: Int) = treeArray(0, cell)
        def bot(cell: Int) = treeArray(1, cell)
        def width(cell: Int) = treeArray(2, cell)
        def score(cell: Int) = java.lang.Float.intBitsToFloat(treeArray(3, cell))
        var begin = 0
        def rec(p: Int):BinarizedTree[C] = {
          val t = top(p)
          val b = bot(p)
          val w = width(p)

//          println(begin, if(t < 0) "" else structure.nontermIndex.get(t), t, if(w > 1) structure.nontermIndex.get(b) else structure.termIndex.get(b), b, w, score(p))
          val botPart = if(w == 1) {
            begin += 1
            val botSym: C = structure.refinements.labels.project(structure.termIndex.get(b))
            assert(botSym != "")
            NullaryTree(botSym, Span(begin - 1, begin))
          } else {
            val botSym: C = structure.refinements.labels.project(structure.nontermIndex.get(b))
            val lc = rec(p + 1)
            val lcWidth = lc.end - lc.begin
            assert(begin == lc.end, (begin, lc.begin, lc.end))
            val rc = rec(p + 2 * lcWidth)
            assert(begin == rc.end, (begin, rc.begin, rc.end))
            BinaryTree(botSym, lc, rc, Span(lc.begin, rc.end))
          }

          if(t == -1) {
            botPart
          } else {
            val topSym: C = structure.refinements.labels.project(structure.nontermIndex.get(t))
            UnaryTree(topSym, botPart, IndexedSeq.empty, botPart.span)
          }


        }
        val t = rec(0)
//        println(t.render(batch.sentences(s)))
        t
      } catch {
        case ex: Exception =>
          ex.printStackTrace()
          null
      }
      val out = if (profile) System.currentTimeMillis() else 0L
      if (profile) {
        println(s"Parse extraction took:  ${(out - in)/1000.0}s")
      }
      trees
    }

    private val unaryThreshold = 0.4

    def extractMBRParses(workspace: WorkSpace, batch: Batch[W], mask: PruningMask, events: CLEvent*): ParSeq[BinarizedTree[C]] = {
      require(mask.hasMasks, "Can't use null pruning mask for parse extraction!")
      events.foreach(_.waitFor())
      val in = if (profile) System.currentTimeMillis() else 0L
      val trees = for (s <- 0 until batch.numSentences par) yield try {
        val length = batch.sentences(s).length

        val bestScores = new TriangularArray[Float](length + 1)
        val bestSplits = new TriangularArray[Int](length + 1)
        val useUnary = new TriangularArray[Boolean](length + 1)


        for(span <- 1 to length; begin <- 0 to length - span) {
          val end = begin + span
          val topMask:DenseVector[Int] = mask.maskForTopCell(s, begin, end).get
          val botMask:DenseVector[Int] = mask.maskForBotCell(s, begin, end).get


          bestScores(begin, end) = {
            val topFloat = java.lang.Float.intBitsToFloat(topMask(0))
            val botFloat = java.lang.Float.intBitsToFloat(botMask(0))
            var topScaled = topFloat
            var botScaled = botFloat

            if(data.isScaling) {
              topScaled *= math.exp(batch.masks.insideTopScaleFor(s, begin, end) + batch.masks.outsideTopScaleFor(s, begin, end) - batch.masks.insideTopScaleFor(s, 0, length)).toFloat
              botScaled *= math.exp(batch.masks.insideScaleFor(s, begin, end) + batch.masks.outsideScaleFor(s, begin, end) - batch.masks.insideTopScaleFor(s, 0, length)).toFloat
            }

            if(length == 1) {
              println(batch.sentences(s) + " " + botScaled + " " + topScaled)
            }

            if(topScaled.isInfinite || botScaled.isInfinite) {
              println("Overflow! taking counter measures")
            }
            if (topScaled.isNaN) {
              topScaled = 0.0f
            }
            if (botScaled.isNaN) {
              botScaled = 0.0f
            }


            topScaled = math.min(topScaled, 1.0f)
            botScaled = math.min(botScaled, 1.0f)
            //            println(topScaled + botScaled)
            useUnary(begin, end) = (begin + 1 < end) || topScaled > unaryThreshold
            val score = if(useUnary(begin, end))
              (topScaled + botScaled).toFloat
            else
              botScaled.toFloat
            //            assert(score < 2.2, botScaled + " " + topScaled)
            score
          }

          if(span > 1) {
            var bestSplitScore = 0.0f
            var bestSplit = begin + 1
            for(split <- (begin+1) until end) {
              val splitScore = bestScores(begin, split) + bestScores(split, end)
              if(splitScore > bestSplitScore) {
                bestSplit = split
                bestSplitScore = splitScore
              }
            }
            bestSplits(begin, end) = bestSplit
            bestScores(begin, end) += bestSplitScore
          }
        }

        def extract(begin: Int, end: Int):BinarizedTree[C] = {
          val topMask:DenseVector[Int] = mask.maskForTopCell(s, begin, end).get
          val botMask:DenseVector[Int] = mask.maskForBotCell(s, begin, end).get
          val score = bestScores(begin, end)
          val bestBot = botMask(1)
          val bestTop = topMask(1)
          val lower = if(begin + 1 == end) {
            val label = structure.refinements.labels.coarseIndex.get(bestBot)
            NullaryTree(label, Span(begin, end))
          } else {
            val label = structure.refinements.labels.coarseIndex.get(bestBot)
            val split = bestSplits(begin, end)
            val left = extract(begin, split)
            val right = extract(split, end)
            BinaryTree(label, left, right, Span(begin, end))
          }

          if(useUnary(begin, end)) {
            val topLabel = structure.refinements.labels.coarseIndex.get(bestTop)
            UnaryTree(topLabel, lower, IndexedSeq.empty, Span(begin, end))
          } else {
            lower
          }
        }

        extract(0, length)

      } catch {
        case ex: Throwable => ex.printStackTrace(); null
      }
      val out = if (profile) System.currentTimeMillis() else 0L
      if (profile) {
        println(s"Parse extraction took:  ${(out - in)/1000.0}s")
      }
      trees

    }

    private class UnaryUpdateManager(kernels: CLUnaryRuleUpdater,
                                     chart: CLMatrix[Float],
                                     parentScale: CLBuffer[Float],
                                     childScale: CLBuffer[Float],
                                     parentChart: (Batch[W],Int)=>Int,
                                     childChart: (Batch[W],Int)=>Int) {



      private def flushQueue(workspace: WorkSpace, batch: Batch[W], span: Int, _ev: Seq[CLEvent]) = {
        import workspace._
        val scoreMatrix = chart
        var ev = _ev
        var offset = 0

        val queueSize = workQueue.queueSize(0)

        val evp = pPtrBuffer.writeArray(queue, workQueue.parentQueue(0), queueSize, ev:_*) profileIn hdTransferEvents
        val evl = lPtrBuffer.writeArray(queue, workQueue.leftQueue(0), queueSize, ev:_*) profileIn hdTransferEvents

        while (offset < queueSize) {
          if(offset != 0) println("uflush!")
          val toDoThisTime = math.min(numWorkCells, queueSize - offset)
          val zz = zmk.shapedFill(devRight(0 until toDoThisTime, ::), zero, ev:_*) profileIn memFillEvents

          val wl = transposeCopy.permuteTransposeCopy(devLeft(0 until toDoThisTime, ::), scoreMatrix, lPtrBuffer, offset, toDoThisTime, evl) profileIn transferEvents

          val endEvents = kernels.update(unaryEvents, devRight(0 until toDoThisTime, ::), parentScale, pPtrBuffer, offset, devLeft(0 until toDoThisTime, ::), childScale, lPtrBuffer, offset, wl, zz, evp)

          val evu = transposeCopy.permuteTransposeCopyOut(scoreMatrix, pPtrBuffer, offset, toDoThisTime, devRight(0 until toDoThisTime, ::), (evp +: endEvents):_*) profileIn unarySumEvents
          ev = Seq(evu)

          offset += toDoThisTime
        }

        workQueue.clear(0)


        ev
      }

      def doUpdates(workspace: WorkSpace, batch: Batch[W], span: Int, events: CLEvent*) = {
        var ev = events


        for {
          sent <- 0 until batch.numSentences
          len = batch.lengths(sent)
          start <- 0 to batch.sentences(sent).length - span
          if batch.isAllowedTopSpan(sent, start, start + span)
        } {
          val end = start + span
          val parentCell = parentChart(batch, sent) + ChartHalf.chartIndex(start, end, len)
          val childCell = childChart(batch, sent) + ChartHalf.chartIndex(start, end, len)
          workspace.workQueue.enqueue(0, parentCell, childCell)
        }

        if (workspace.workQueue.nonEmpty) {
          ev = flushQueue(workspace, batch, span, ev)
        }

        assert(ev.length == 1)
        ev.head
      }

    }

    val queueTimer = new Timer

    private class BinaryUpdateManager(updater: CLBinaryRuleUpdater,
                                      parentIsBot: Boolean,
                                      parentChartMatrix: CLMatrix[Float],
                                      parentScale: CLBuffer[Float],
                                      leftChartMatrix: CLMatrix[Float],
                                      leftScale: CLBuffer[Float],
                                      rightChartMatrix: CLMatrix[Float],
                                      rightScale: CLBuffer[Float],
                                      parentChart: (Batch[W], Int)=>Int,
                                      leftChart: (Batch[W], Int)=>Int,
                                      rightChart: (Batch[W], Int)=>Int,
                                      ranger: (Int, Int, Int)=>Range) {

      val numBlocks = updater.numKernelBlocks


      private val merge = (CLParser.this.data.last ne data) && CLParser.this.data.length != 1
      private val blocks = if(merge) IndexedSeq(0 until numBlocks) else (0 until numBlocks).groupBy(updater.kernels(_).parents.data.toIndexedSeq).values.toIndexedSeq
      private val blockMasks = Array.tabulate(blocks.length)(i => updater.kernels(blocks(i).head).parents)


       private def flushQueue(workspace: WorkSpace, batch: Batch[W], span: Int, _ev: Seq[CLEvent]): Seq[CLEvent] = {
         import workspace._

//         assert(queueSize == _todo, s"$queueSize ${_todo} $span")
//         CLEvent.waitFor(_ev:_*)
//         val oldLeftPtr = lPtrBuffer.read(queue, 0, queueSize, _ev:_*).getInts()
//         assert(oldLeftPtr.take(queueSize).toIndexedSeq == lArray.take(queueSize).toIndexedSeq, s"${oldLeftPtr.take(queueSize).toIndexedSeq} ${lArray.take(queueSize).toIndexedSeq}")
//         val oldRightPtr = rPtrBuffer.read(queue, 0, queueSize, _ev:_*).getInts()
//         assert(oldRightPtr.take(queueSize).toIndexedSeq == rArray.take(queueSize).toIndexedSeq, s"${oldRightPtr.take(queueSize).toIndexedSeq} ${rArray.take(queueSize).toIndexedSeq}")
//         val oldParentPtr = pPtrBuffer.read(queue, 0, queueSize, _ev:_*).getInts()
//         assert(oldParentPtr.take(queueSize).toIndexedSeq == pArray.take(queueSize).toIndexedSeq, s"${oldParentPtr.take(queueSize).toIndexedSeq} ${pArray.take(queueSize).toIndexedSeq}")
         var ev = _ev

         for(blockid <- 0 until blocks.size if workQueue.nonEmpty(blockid)) {

           val queueSize = workQueue.queueSize(blockid)
           var offset = 0
           // copy ptrs to opencl
           val evP = pPtrBuffer.writeArray(queue, workQueue.parentQueue(blockid), queueSize, ev:_*) profileIn hdTransferEvents
           val evL = lPtrBuffer.writeArray(queue, workQueue.leftQueue(blockid), queueSize, ev:_*) profileIn hdTransferEvents
           val evR = rPtrBuffer.writeArray(queue, workQueue.rightQueue(blockid), queueSize, ev:_*) profileIn hdTransferEvents

           ev = Seq(evP, evL, evR)

           while (offset < queueSize) {
             val toDoThisTime = math.min(numWorkCells, queueSize - offset)

             val updateDirectToChart = updater.directWriteToChart
             require(updateDirectToChart)


             // do transpose based on ptrs
             val evTransLeft  = transposeCopy.permuteTransposeCopy(devLeft(0 until toDoThisTime, ::), leftChartMatrix, lPtrBuffer, offset, toDoThisTime, ev:_*) profileIn transferEvents
             val evTransRight = transposeCopy.permuteTransposeCopy(devRight(0 until toDoThisTime, ::), rightChartMatrix, rPtrBuffer, offset, toDoThisTime, ev:_*) profileIn transferEvents

             val targetChart =  parentChartMatrix
             val kEvents = updater.update(blocks(blockid), binaryEvents,
               targetChart, parentScale, pPtrBuffer, offset,
               devLeft(0 until toDoThisTime, ::), leftScale, lPtrBuffer, offset,
               devRight(0 until toDoThisTime, ::), rightScale, rPtrBuffer, offset,
               maskCharts, evTransLeft, evTransRight)



             offset += toDoThisTime
             ev = kEvents
           }

         }
         workQueue.clear()
         ev
       }

//      var _todo = 0
      private val zeroArray = Array(0)
      import spire.syntax.cfor._

      private def determineBlocksToDo(mask: Option[DenseVector[Int]]) = mask match {
        case None => zeroArray
        case Some(m) =>
          val out = new mutable.ArrayBuilder.ofInt
          cfor(0)(_ < blocks.length, _ + 1) { i =>
            if(intersects(blockMasks(i), m)) {
              out += i
            }
          }
          out.result
      }

      def doUpdates(workspace: WorkSpace, batch: Batch[W], span: Int, events: CLEvent*) = {

         var ev = events
        queueTimer.tic()

        cfor(0)(_ < batch.numSentences, _ + 1) { sent =>
          val lastStart = batch.sentences(sent).length - span
          cfor(0)(_ <= lastStart, _ + 1) { start =>
            total += 1
            val end = start + span
            val mask = if(parentIsBot) batch.botMaskFor(sent, start, start + span) else batch.topMaskFor(sent, start, start + span)
          if(profile) mask match {
            case None =>
            case Some(x) => for(v <- x.valuesIterator) totalSyms += Integer.bitCount(v)
          }
            val blockids = determineBlocksToDo(mask)

            if(blockids.nonEmpty) {

              val len = batch.lengths(sent)

              val parentCell = parentChart(batch, sent) + ChartHalf.chartIndex(start, end, len)
              val leftChartOffset = leftChart(batch, sent)
              val rightChartOffset = rightChart(batch, sent)

              val leftIsTop = leftChart eq ActualParser.this.insideTop
              val rightIsTop = rightChart eq ActualParser.this.insideTop

              val splitRange = ranger(start, start + span, batch.sentences(sent).length)
              var split =  splitRange.start
              val splitEnd = splitRange.terminalElement
              val step = splitRange.step
              while (split != splitEnd) {
                if (split >= 0 && split <= batch.sentences(sent).length) {
                  val end = start + span
                  val leftChildAllowed = if (split < start) if(leftIsTop) batch.isAllowedTopSpan(sent, split, start) else batch.isAllowedSpan(sent,split, start) else if(leftIsTop) batch.isAllowedTopSpan(sent, start, split) else batch.isAllowedSpan(sent, start, split)
                  val rightChildAllowed = if (split < end) if(rightIsTop) batch.isAllowedTopSpan(sent, split, end) else batch.isAllowedSpan(sent,split,end) else if (rightIsTop) batch.isAllowedTopSpan(sent, end, split) else batch.isAllowedSpan(sent, end, split)

                  if (leftChildAllowed && rightChildAllowed) {
                    val leftChild = leftChartOffset + ChartHalf.chartIndex(split, start, len)
                    val rightChild = rightChartOffset + ChartHalf.chartIndex(split, end, len)

                    cfor(0)(_ < blockids.length, _ + 1) { b =>
                      workspace.workQueue.enqueue(blockids(b), parentCell, leftChild, rightChild)
                    }


                  }
                }
                split += step
              }
            } else {
              pruned += 1
            }

          }
        }

        queueTimer.toc()


         if(workspace.workQueue.nonEmpty)
           ev = flushQueue(workspace, batch, span, ev)

         assert(ev.length == 1)
         ev.head
       }

     }

  }



  private def intersects(blockMask: DenseVector[Int], spanMask: DenseVector[Int]):Boolean = {
    assert(blockMask.length == spanMask.length)
    var i = 0
    if(blockMask.stride == 1 && spanMask.stride == 1) {
      val bdata = blockMask.data
      val sdata = spanMask.data
      while(i < blockMask.length) {
        if( (bdata(i) & sdata(i + spanMask.offset)) != 0) return true
        i += 1
      }
    } else {
      while(i < blockMask.length) {
        if( (blockMask.unsafeValueAt(i) & spanMask.unsafeValueAt(i)) != 0) return true
        i += 1
      }
    }

    false

  }

}

object CLParser extends LazyLogging {

  case class Params(annotator: TreeAnnotator[AnnotatedLabel, String, AnnotatedLabel] = Xbarize(),
                    device: String = "nvidia",
                    profile: Boolean = false,
                    parseTrain: Boolean = false,
                    numToParse: Int = 1000, codeCache: File = new File("grammar.grz"), cache: Boolean = true,
                    maxParseLength: Int = 10000,
                    jvmParse: Boolean = false, parseTwice: Boolean = false,
                    textGrammarPrefix: String = null,
                    checkPartitions: Boolean = false,
                    justInsides: Boolean = false,
                    noExtraction: Boolean = false,
                    mem: String = "1g",
                    reproject: Boolean = false,
                    viterbi: Boolean = true,
                    logsum: Boolean = false,
                    numToDrop: Int = 0,
                    evalReference: Boolean = false,
                    printTrees: Boolean = false,
                    pauseBeforeRunning: Boolean = false)

  def repeatToSize[T:ClassTag](arr: IndexedSeq[T], size: Int) = {
    val len = arr.length
    val mult = roundUpToMultipleOf(size,len)/len
    IndexedSeq.fill(mult)(arr).flatten.take(size)
  }

  def main(args: Array[String]) = {
    import ParserParams.JointParams

    val params = CommandLineParser.readIn[JointParams[Params]](args)
    val myParams:Params = params.trainer
    import myParams._


    implicit val context: CLContext = {
      val (good, bad) = JavaCL.listPlatforms().flatMap(_.listAllDevices(true)).partition(d => device.r.findFirstIn(d.toString.toLowerCase()).nonEmpty)
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
    val whatToParse = if(parseTrain) params.treebank.trainTrees else params.treebank.trainTrees
    val gold = repeatToSize(whatToParse.filter(_.words.length <= maxParseLength).drop(numToDrop), numToParse)
    val toParse =  gold.map(_.words)


    val defaultGenerator = GenType.CoarseParent
    val prunedGenerator = GenType.CoarseParent

    val finePassSemiring = if(viterbi) {
      ViterbiRuleSemiring
    } else if (logsum) {
      LogSumRuleSemiring
    } else {
      RealSemiring
    }

    val parserDatas: IndexedSeq[CLParserData[AnnotatedLabel, AnnotatedLabel, String]] = if (cache && codeCache != null && codeCache.exists()) {
      CLParserData.readSequence[AnnotatedLabel, AnnotatedLabel, String](new ZipFile(codeCache))
    } else if (textGrammarPrefix == null) {
      IndexedSeq(CLParserData.make(GenerativeParser.annotated(annotator, transformed), defaultGenerator, true, finePassSemiring))
    } else {
      val paths = textGrammarPrefix.split(":")
      var grammars = paths.zipWithIndex.map{ case (f,i) => SimpleRefinedGrammar.parseBerkeleyText(f,  -12, CloseUnaries.None)}
      if(reproject && grammars.length > 1) {
        val (newc, newr) = reprojectGrammar(grammars.head, textGrammarPrefix.split(":").head, grammars.last, textGrammarPrefix.split(":").last)
        grammars = Array(newc, newr)
      }

      val fineLayer =  CLParserData.make(grammars.last, if(grammars.length > 1) prunedGenerator else defaultGenerator, true, finePassSemiring)
      val coarseGrammars = grammars.dropRight(1)
      val coarseData = coarseGrammars.map(CLParserData.make(_,  defaultGenerator, directWrite = true, ViterbiRuleSemiring))

      coarseData :+ fineLayer
    }

    if (cache && !codeCache.exists() && codeCache != null) {
      CLParserData.writeSequence(codeCache, parserDatas)
    }

    val kern = {
      fromParserDatas[AnnotatedLabel, AnnotatedLabel, String](parserDatas, profile, parseMemString(mem))
    }

    val grammars = parserDatas.map(_.grammar)


    val parser = if(grammars.length > 1) {
      val gr = grammars.last
      Parser(new ConstraintCoreGrammarAdaptor(gr.grammar, gr.lexicon, new ParserChartConstraintsFactory(Parser(grammars.head, new ViterbiDecoder[AnnotatedLabel, String]), (_:AnnotatedLabel).isIntermediate)),
        gr,
        if (kern.isViterbi) new ViterbiDecoder[AnnotatedLabel, String] else new MaxConstituentDecoder[AnnotatedLabel, String])
    } else {
      val grammar = grammars.last
      Parser(grammar, if (kern.isViterbi) new ViterbiDecoder[AnnotatedLabel, String] else new MaxConstituentDecoder[AnnotatedLabel, String])
    }


    if(evalReference) {
      val res = ParserTester.evalParser(gold, parser, "cpu-reference", 4)
      println(res)
    }

    if (justInsides || checkPartitions) {
      val partsX = logTime("CL Insides", toParse.length)( kern.partitions(toParse))
      println(partsX.last)
      if (checkPartitions) {
        val parts2 = toParse.par.map(parser.marginal(_).logPartition)
        println(parts2.last)
        println("max difference: " + (DenseVector(partsX.map(_.toDouble):_*) - DenseVector(parts2.seq:_*)).norm(Double.PositiveInfinity))
      }
      System.exit(0)
    }

    if (noExtraction) {
      val partsX = logTime("CL Inside/Outside", toParse.length)( kern.insideOutside(toParse))
      println(partsX)
      System.exit(0)
    }


    val trees = logTime("CL Parsing:", toParse.length)(kern.parse(toParse))
    println(eval(trees zip gold.map(_.tree) zip toParse, "opencl", printTrees))
    if (parseTwice) {
      val trees = logTime("CL Parsing x2:", toParse.length)(kern.parse(toParse))
      println(eval(trees zip gold.map(_.tree) zip toParse, "opencl-twice"))
    }
    if (jvmParse) {

      val margs = logTime("JVM Parse", toParse.length) {
        toParse.par.map { w =>
          val m = parser.apply(w)
          m
        }.seq.toIndexedSeq
      }
      println(eval(margs zip gold.map(_.tree) zip toParse, ""))
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

  def eval(trees: IndexedSeq[((BinarizedTree[AnnotatedLabel], BinarizedTree[AnnotatedLabel]), IndexedSeq[String])], name: String, printTrees: Boolean = false) = {
    val chainReplacer = AnnotatedLabelChainReplacer
    val outDir = new File(s"eval-$name")
    outDir.mkdirs()
    val goldOut = new PrintStream(new FileOutputStream(new File(outDir, "gold")))
    val guessOut = new PrintStream(new FileOutputStream(new File(outDir, "guess")))
    val eval: ParseEval[String] = new ParseEval(Set("","''", "``", ".", ":", ",", "TOP"))
    val res = trees filter (_._1 ne null) map { case ((guess, gold), words) =>
      val tree: Tree[String] = chainReplacer.replaceUnaries(guess).map(_.label)
      val guessTree = Trees.debinarize(Trees.deannotate(tree))
      val deBgold: Tree[String] = Trees.debinarize(Trees.deannotate(chainReplacer.replaceUnaries(gold).map(_.label)))
      val stats = eval.apply(guessTree, deBgold)
      goldOut.println(deBgold.render(words, false))
      guessOut.println(guessTree.render(words, false))
      if(printTrees) {
        println("Guess:\n"  + guessTree.render(words))
        println("Gold:\n"  + deBgold.render(words))
        println(stats)
        println("=====")
      }
      stats
    } reduceLeft (_ + _)

    goldOut.close()
    guessOut.close()
    res
  }

  def parseMemString(x: String) = x.last.toLower match {
    case 'g' => Math.scalb(x.dropRight(1).toDouble, 30).toLong
    case 'm' => Math.scalb(x.dropRight(1).toDouble, 20).toLong
    case 'k' => Math.scalb(x.dropRight(1).toDouble, 10).toLong
    case y:Char if y.isLetterOrDigit => throw new RuntimeException(s"bad mem string: $x")
    case _ => x.toLong

  }



  def reprojectGrammar(coarseGrammar: SimpleRefinedGrammar[AnnotatedLabel, AnnotatedLabel, String], coarseGrammarName: String,
                    fineGrammar: SimpleRefinedGrammar[AnnotatedLabel, AnnotatedLabel, String],  fineGrammarName: String) = {
    val symMap: Map[String, IndexedSeq[String]] = readHierarchy(fineGrammarName, coarseGrammarName)
    var reverseSymMap: Map[String, String] = for {
      (coarse, fines) <- symMap
      f <- fines
    } yield f -> coarse

    reverseSymMap += ("TOP_0" -> "TOP_0")
    val coarseLevelRefinedGrammar: BaseGrammar[AnnotatedLabel] = coarseGrammar.refinedGrammar
    val fineLevelRefinedGrammar: BaseGrammar[AnnotatedLabel] = fineGrammar.refinedGrammar
    val newBaseRefinements = GrammarRefinements.identity(coarseLevelRefinedGrammar)
    val newFineRefinements = GrammarRefinements(coarseLevelRefinedGrammar, fineLevelRefinedGrammar, {(x: AnnotatedLabel) => AnnotatedLabel(reverseSymMap(x.label))}, skipMissingCoarseRules = true)
    
    val newCoarseLexicon = new SplitLexicon(coarseGrammar.lexicon, coarseLevelRefinedGrammar.labelIndex, coarseGrammar.refinements.labels)

    val newBaseGrammar =  RefinedGrammar.unanchored[AnnotatedLabel, AnnotatedLabel, String](coarseLevelRefinedGrammar, newCoarseLexicon,
      newBaseRefinements,
      flattenRuleScores(coarseGrammar),
      new Array(coarseGrammar.refinedGrammar.labelIndex.size),
      coarseGrammar.tagScorer)

    val newFineGrammar = RefinedGrammar.unanchored[AnnotatedLabel, AnnotatedLabel, String](coarseLevelRefinedGrammar, newCoarseLexicon,
      newFineRefinements,
      flattenRuleScores(fineGrammar),
      new Array(fineGrammar.refinedGrammar.labelIndex.size),
      fineGrammar.tagScorer)

    (newBaseGrammar, newFineGrammar)
  }


  def readHierarchy(fineGrammarName: String, coarseGrammarName: String): Map[String, IndexedSeq[String]] = {
    val fine = fineGrammarName.replaceAll("^.*(?:^|[^0-9])([0-9]+)([^0-9]*)$", "$1").toInt
    val coarse = coarseGrammarName.replaceAll("^.*(?:^|[^0-9])([0-9]+)([^0-9]*)$", "$1").toInt

    val symsIter = for (l <- Source.fromFile(new File(fineGrammarName + ".hierarchy")).getLines()) yield {
      val Array(sym: String, rest: String) = l.split("\\s+", 2)
      val (hierarchy, leaves) = PennTreeReader.parseHard(rest)

      val totalHeight = hierarchy.leftHeight

      for {
        subtree <- hierarchy.preorder
        if subtree.leftHeight == (totalHeight - coarse)
      } yield {
        val coarseSym = s"${sym}_${subtree.label}"
        val fineSyms = if (fine - 1 == totalHeight) {
          subtree.span.map(leaves).map(q => s"${sym}_$q")
        } else {
          subtree.preorder.filter(_.leftHeight == (totalHeight - fine)).map(_.label).map(q => s"${sym}_$q").toIndexedSeq
        }
        coarseSym -> fineSyms
      }
    }

    val symMap = symsIter.flatten.toMap
    symMap
  }

  private def flattenRuleScores(grammar: SimpleRefinedGrammar[AnnotatedLabel, AnnotatedLabel, String]): Array[Double] = {
    Array.tabulate(grammar.refinedGrammar.index.size) { grammar.ruleScore(_) }
  }

  @SerialVersionUID(1L)
  private class SplitLexicon[C, L, W](baseLexicon: Lexicon[C, W], val labelIndex: Index[L],
                                      proj: ProjectionIndexer[C, L]) extends Lexicon[L, W] with Serializable {
    def anchor(w: IndexedSeq[W]): Anchoring = new Anchoring {
      val canchor = baseLexicon.anchor(w)
      def length: Int = w.length

      def allowedTags(pos: Int): Set[Int] = canchor.allowedTags(pos).flatMap(proj.refinementsOf(_:Int))
    }
  }

}


case class CLParserData[C, L, W](grammar: SimpleRefinedGrammar[C, L, W],
                                 structure: RuleStructure[C, L],
                                 semiring: RuleSemiring,
                                 inside: CLInsideKernels,
                                 outside: CLOutsideKernels,
                                 masks: CLMaskKernels,
                                 viterbi: CLViterbi,
                                 mbr: CLMBRKernels,
                                 scaling: CLScalingKernels,
                                 util: CLParserUtils) {

  def isViterbi = semiring.plusIsIdempotent
  def isScaling = semiring.isInstanceOf[RealSemiring.type]
  def isLogSum = semiring.isInstanceOf[LogSumRuleSemiring.type]

  def numSyms = structure.nontermIndex.size max structure.termIndex.size
  def maskSize = masks.maskSize

  def write(prefix:String, zout: ZipOutputStream) {
    ZipUtil.serializedEntry(zout, s"$prefix/grammar", grammar)
    ZipUtil.serializedEntry(zout, s"$prefix/structure", structure)
    ZipUtil.serializedEntry(zout, s"$prefix/semiring", semiring)
    inside.write(prefix, zout)
    outside.write(prefix, zout)
    util.write(prefix, zout)
    masks.write(prefix, zout)
    mbr.write(prefix, zout)
    viterbi.write(prefix, zout)
    scaling.write(prefix, zout)
  }
}

object CLParserData {
  def make[C, L, W](grammar: SimpleRefinedGrammar[C, L, W], genType: GenType, directWrite: Boolean, semiring: RuleSemiring)(implicit context: CLContext) = {
    implicit val semi = semiring
    val ruleScores: Array[Float] = Array.tabulate(grammar.refinedGrammar.index.size){r =>
      val projectedRule = grammar.refinements.rules.project(r)
      if(projectedRule < 0) {
        semi.fromLogSpace(-12)
      } else {
        val score = grammar.ruleScoreArray(projectedRule)(grammar.refinements.rules.localize(r))
        semi.fromLogSpace(score.toFloat)
      }
    }
    val structure = new RuleStructure(grammar.refinements, grammar.refinedGrammar, ruleScores)
    val inside = CLInsideKernels.make(structure, directWrite, semiring, genType)
    val outside =  CLOutsideKernels.make(structure, directWrite, semiring, genType)
    val util = CLParserUtils.make(structure)
    val masks = CLMaskKernels.make(structure)
    val viterbi = CLViterbi.make(structure)
    val mbr = CLMBRKernels.make(structure)
    val scaling = CLScalingKernels.make(structure)

    new CLParserData(grammar, structure, semi, inside, outside, masks, viterbi, mbr, scaling, util)
  }

  def read[C, L, W](prefix: String, file: ZipFile)(implicit context: CLContext): CLParserData[C, L, W] = {
    val gr = ZipUtil.deserializeEntry[SimpleRefinedGrammar[C, L, W]](file.getInputStream(file.getEntry(s"$prefix/grammar")))
    val structure = ZipUtil.deserializeEntry[RuleStructure[C, L]](file.getInputStream(file.getEntry(s"$prefix/structure")))
    val semiring = ZipUtil.deserializeEntry[RuleSemiring](file.getInputStream(file.getEntry(s"$prefix/semiring")))
    val inside = CLInsideKernels.read(prefix, file)
    val outside = CLOutsideKernels.read(prefix, file)
    val util = CLParserUtils.read(prefix, file)
    val masks = CLMaskKernels.read(prefix, file)
    val mbr = CLMBRKernels.read(prefix, file)
    val viterbi = CLViterbi.read(prefix, file)
    val scaling = CLScalingKernels.read(prefix, file)

    CLParserData(gr, structure, semiring, inside, outside, masks, viterbi, mbr, scaling, util)
  }

  def readSequence[C, L, W](file: ZipFile)(implicit context: CLContext):IndexedSeq[CLParserData[C, L, W]] = {
    val parsers = Source.fromInputStream(file.getInputStream(file.getEntry("parserNames")))(Codec.UTF8).getLines().map(_.trim).toArray
    for(p <- parsers) yield read[C, L, W](p, file)
  }

  def writeSequence[C, L, W](file: File, parsers: Seq[CLParserData[C, L, W]])(implicit context: CLContext) = {
    val out = new ZipOutputStream(new FileOutputStream(file))
    val names = Array.tabulate(parsers.length)(x => s"parser-$x")
    ZipUtil.addEntry(out, "parserNames", names.mkString("\n").getBytes("UTF-8"))

    for( (d, n) <- parsers zip names) {
      d.write(n, out)
    }

    out.close()
  }
}
