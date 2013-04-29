package puck.parser


import breeze.collection.mutable.TriangularArray
import breeze.config.{Configuration, CommandLineParser}
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.util.{Index, Encoder}
import collection.mutable.ArrayBuffer
import collection.{immutable, mutable}
import com.nativelibs4java.opencl.CLMem.Usage
import com.nativelibs4java.opencl._
import epic._
import epic.parser._
import epic.parser.models._
import epic.trees._
import epic.trees.annotations._
import java.io.File
import java.nio.{FloatBuffer, ByteBuffer}
import java.{util, lang}
import org.bridj.Pointer
import projections.{GrammarRefinements, ProjectionIndexer}
import puck.util.{MemBufPair, ZeroMemoryKernel}
import puck.parser.gen.{SemiringFloatOpsExp, LogSpaceFloatOpsExp, ParserGenerator}
import epic.sequences.CRF.TransitionVisitor


class GPUParser[C, L, W](coarseGrammar: BaseGrammar[C],
                         projections: GrammarRefinements[C, L],
                         grammar: BaseGrammar[L],
                         lexicon: Lexicon[L, W],
                         private var _ruleScores: Array[RuleScores],
                         var tagScorers: Array[(IndexedSeq[W],Int,Int)=>Double],
                         profile: Boolean = true,
                         maxSentences: Int = 1000)(implicit val context: CLContext) extends CLInsideAlgorithm[C, L] with CLOutsideAlgorithm[C, L] with CLPartitionCalculator[C, L] {
  import GPUParser._
  def ruleScores = _ruleScores

  val numGrammars = _ruleScores.length
  val structure = RuleStructure[C, L](projections, grammar)


  val parserGen = new ParserGenerator[L](structure, numGrammars) with LogSpaceFloatOpsExp

//  val outsideKernel = new OutsideKernel(structure, numGrammars)
//  val ecountsKernel = new ExpectedCountsKernel(structure, numGrammars)
  import GPUParser._
  import structure.{grammar=>_, _ }

  val nrules = grammar.index.size
  val totalRules: Int = ruleScores.length * ruleScores(0).scores.length
  val cellSize = numGrammars * structure.numNonTerms
  val termCellSize = numGrammars * structure.numTerms

  val (maxCells, maxTotalLength) = GPUCharts.computeMaxSizes(context.getDevices.map(_.getGlobalMemSize).min / 6, context.getMaxMemAllocSize, structure, numGrammars)
  parserGen.define("CHART_SIZE", maxCells)
  parserGen.define("TERM_CHART_SIZE", maxTotalLength)

  val coarseCellSize = (structure.numCoarseSyms+1) * numGrammars

  private implicit val queue = if(profile) context.createDefaultProfilingQueue() else context.createDefaultOutOfOrderQueueIfPossible()
//  private val projection = new ProjectionKernel(structure, numGrammars)
//  private val decoder = new MaxRecallKernel(structure, numGrammars)

  private val inside, outside = GPUCharts.forGrammar(structure, numGrammars, maxCells, maxTotalLength)
  private val offPair = MemBufPair[Int](Usage.Input, maxSentences + 1)
  private val offLenPair = MemBufPair[Int](Usage.Input, maxSentences)
  private val lenPair = MemBufPair[Int](Usage.Input, maxSentences)
//  private val decodePair = MemBufPair[Int](maxCells * 4, Usage.InputOutput) // top, bot, split, score

  private val rules = MemBufPair[Float](totalRules, Usage.Input)
  ruleScores = _ruleScores

  def ruleScores_=(newRules: Array[RuleScores]) {
    _ruleScores = newRules
    val arr = new Array[Float](rules.length.toInt)
    for(g <- 0 until numGrammars) {
      for(b <- 0 until ruleScores(g).scores.length) {
        arr(b * numGrammars + g) = parserGen.fromLogSpace(ruleScores(g).scores(b).toFloat)
      }
    }

    rules.data = arr
  }

  /*
  def parse(sentences: IndexedSeq[IndexedSeq[W]], masks: IndexedSeq[PruningMask] = null):IndexedSeq[BinarizedTree[C]] = synchronized {
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

  def partitions(sentences: IndexedSeq[IndexedSeq[W]]): IndexedSeq[Double] = {
    {for {
      partition <- getBatches(sentences, null).iterator
      batch = createBatch(partition)
      //    _ = getMarginals(batch)
      t <- doGetPartitions(batch)
    } yield {
      t
    }}.toIndexedSeq
  }

  private def computePartitions(batch: Batch, events: CLEvent*):Array[Double] = {
    val parts = this.partitions(inside.top.dev, offPair.dev, lenPair.dev, batch.numSentences, events: _*).map(_.toDouble)
    import batch._
    val outsideCharts = outside.scoresForBatch(offsets, lengthTotals)
    val insideCharts = inside.scoresForBatch(offsets, lengthTotals)
    for( (s, p) <- 0 until numSentences zip parts) {
      val ys = for(l <- 0 until nontermIndex.size) yield (outsideCharts(s).bot(0, lengths(s), 0, l) + insideCharts(s).bot(0, lengths(s), 0, l)).toDouble
      println( s"$s $p ${breeze.numerics.logSum(ys)}")
    }
//    for( (s, p) <- 0 until numSentences zip parts; i <- 0 until lengths(s)) {
//      val ys = for(l <- 0 until termIndex.size) yield (outsideCharts(s).tag(i, 0, l) + insideCharts(s).tag(i, 0, l)).toDouble
//      println( (s,i) + " " + p + " " + breeze.numerics.logSum(ys))
//    }
    parts
  }

  private def doGetPartitions(batch: Batch):Array[Double] = {
    val res = insideOutside(batch)
    computePartitions(batch, res)
  }



  case class Batch(lengths: Array[Int],
                   offsets: Array[Int],
                   lengthTotals: Array[Int],
                   totalLength: Int,
                   sentences: IndexedSeq[IndexedSeq[W]],
                   posTags: Array[Float],
                   mask: Array[Long]) {
    def numSentences = sentences.length
    def numCells = offsets.last
  }


  private def createBatch(sentences: IndexedSeq[(IndexedSeq[W], Option[PruningMask])]): Batch = {
    val lengths = sentences.map(_._1.length)
    val offsets = new ArrayBuffer[Int]()

    var offset = 0

    val partialLengths = new Array[Int](lengths.size+1)
    var totalLength = 0
    var i = 0
    while(i < lengths.length) {
      partialLengths(i) = totalLength
      totalLength += lengths(i)
      i += 1
    }
    partialLengths(partialLengths.size-1) = totalLength
    assert(maxTotalLength >= totalLength, maxTotalLength -> totalLength)

    val posTags = new Array[Float](maxTotalLength * cellSize)
    util.Arrays.fill(posTags, parserGen._zero)
    val fullMask = Array.fill(maxCells * structure.pruningMaskFieldSize)(-1L)


    for( ((s, mask), i) <- sentences.zipWithIndex) {
      offsets += offset
      for(pos <- (0 until s.length);
          aa <- lexicon.tagsForWord(s(pos));
          a = termIndex(aa)) {
        for(g <- 0 until numGrammars) {
          val score = tagScorers(g)(s, pos, grammar.labelIndex(aa))
          posTags((a * maxTotalLength + (partialLengths(i) + pos)) * numGrammars + g) = parserGen.fromLogSpace(score.toFloat)
          assert(!score.isNaN)
        }
      }
      for( m <- mask) {
        assert(m.bits.length == TriangularArray.arraySize(s.length + 1) * structure.pruningMaskFieldSize, m.bits.length + " " + TriangularArray.arraySize(s.length + 1))
        System.arraycopy(m.bits, 0, fullMask, offset * structure.pruningMaskFieldSize, m.bits.length)
      }

      offset += TriangularArray.arraySize(s.length+1)
    }
    offsets += offset

    Batch(lengths.toArray, offsets.toArray, partialLengths, totalLength, sentences.map(_._1), posTags, fullMask)
  }

  private def getBatches(sentences: IndexedSeq[IndexedSeq[W]], masks: IndexedSeq[PruningMask]): IndexedSeq[IndexedSeq[(IndexedSeq[W], Option[PruningMask])]] = {
    val result = ArrayBuffer[IndexedSeq[(IndexedSeq[W], Option[PruningMask])]]()
    var current = ArrayBuffer[(IndexedSeq[W], Option[PruningMask])]()
    var currentCellTotal = 0
    var currentLengthTotal = 0
    for( (s, i) <- sentences.zipWithIndex) {
      currentCellTotal += TriangularArray.arraySize(s.length+1)
      currentLengthTotal += s.length
      if(currentCellTotal > maxCells || current.size >= maxSentences || currentLengthTotal > maxTotalLength) {
        assert(current.nonEmpty)
        result += current
        currentCellTotal = TriangularArray.arraySize(s.length+1)
        currentLengthTotal = s.length
        current = ArrayBuffer()
      }
      current += (s -> Option(masks).map(_(i)))
    }

    if(current.nonEmpty) result += current
    result
  }

  /*
  private def doParse(batch: Batch):IndexedSeq[BinarizedTree[C]] = synchronized {
    import batch._
    var lastEvent = insideOutside(batch)
    val zt = proj.clear()

    lastEvent = projection.projectCells(numSentences,
      proj,
      inside,
      outside,
      offDev, offLengthsDev, lenDev, lengths.max, lastEvent, zt)

    lastEvent = memZero.zeroMemory(decodePair.dev.asCLFloatBuffer(), lastEvent)
    lastEvent = decoder.makeBackpointers(numSentences, decodePair.dev,
      proj,
      offDev, offLengthsDev, lenDev, lengths.max, lastEvent)

    val backPointers = decodePair.data
    // 0 is top, 1 is bot, 2 is split, 3 is score (unused, actually a float)
    val trees = for(i <- 0 until batch.sentences.length) yield {
//      for(len <- 1 to sentences(i).length; begin <- 0 until (sentences(i).length + 1 - len))  {
//        val end = begin + len
//        val bestTop = backPointers( (offsets(i) + TriangularArray.index(begin, end)) * 4 + 0)
//        val bestBot = backPointers( (offsets(i) + TriangularArray.index(begin, end)) * 4 + 1)
//        val split = backPointers( (offsets(i) + TriangularArray.index(begin, end)) * 4 + 2)
//        val bestScore = lang.Float.intBitsToFloat(backPointers( (offsets(i) + TriangularArray.index(begin, end)) * 4 + 3))
//        println(begin,split,end,bestBot,bestTop,bestScore)
//      }
      def extract(begin: Int, end: Int):BinarizedTree[C] = {
        val bestTop = backPointers( (offsets(i) + TriangularArray.index(begin, end)) * 4 + 0)
        val bestBot = backPointers( (offsets(i) + TriangularArray.index(begin, end)) * 4 + 1)
        val split = backPointers( (offsets(i) + TriangularArray.index(begin, end)) * 4 + 2)
        val bestScore = lang.Float.intBitsToFloat(backPointers( (offsets(i) + TriangularArray.index(begin, end)) * 4 + 3))
//        println(split + " " + begin + " " + end)
        val lower = if(begin + 1== end) {
          NullaryTree(coarseGrammar.labelIndex.get(bestBot), Span(begin, end))
        } else {
          assert(split > begin && split < end, (i, sentences(i),begin,split,end,coarseGrammar.labelIndex.get(bestBot),coarseGrammar.labelIndex.get(bestTop),bestScore))
          val left = extract(begin, split)
          val right = extract(split, end)
//          println(begin,split,end,bestBot,bestTop,bestScore)
          BinaryTree(coarseGrammar.labelIndex.get(bestBot), left, right, Span(begin, end))
        }

        UnaryTree[C](coarseGrammar.labelIndex.get(bestTop), lower, IndexedSeq.empty, Span(begin, end))
      }

      extract(0, sentences(i).length)
    }



    trees
  }
  */

  private def insideOutside(batch: Batch) = synchronized {
    import batch._
    val maxLength = lengths.max
    offPair.data = offsets
    lenPair.data = lengths
    offLenPair.data = lengthTotals

    inside.setTo(parserGen._zero)
    outside.setTo(parserGen._zero)
    queue.finish()
    inside.tags.data = batch.posTags


    var lastU: CLEvent = insidePass(numSentences, inside, offPair.dev, lenPair.dev, maxLength, offLenPair.dev, rules.dev)
    lastU = outsidePass(numSentences, outside, inside, offPair.dev, lenPair.dev, offLenPair.dev,  maxLength, rules.dev, lastU)


//    if(queue.getProperties.contains(CLDevice.QueueProperties.ProfilingEnable)) {
//      queue.finish()
//      val writeCounts = IndexedSeq(wIT, wOT, wIB, wOB, wO, wL).filter(_ ne null).map(e => e.getProfilingCommandEnd - e.getProfilingCommandStart).sum / 1E9
//      println("io write: " + writeCounts)
//    }
    lastU
  }

  override protected def finalize() {
    println("Release!")
    offPair.release()
    lenPair.release()
    offLenPair.release()
    inside.release()
    outside.release()
  }

}

object GPUParser {
  case class PruningMask(bits: Array[Long])

  case class Params(annotator: TreeAnnotator[AnnotatedLabel, String, AnnotatedLabel] = FilterAnnotations(),
                    useGPU: Boolean = true, numToParse: Int = 1000, numGrammars: Int = 1)

  def main(args: Array[String]) {
    import ParserParams.JointParams

    val (baseConfig, files) = CommandLineParser.parseArguments(args)
    val config = baseConfig backoff Configuration.fromPropertiesFiles(files.map(new File(_)))
    val params = try {
      config.readIn[JointParams[Params]]("test")
    } catch {
      case e: Exception =>
        e.printStackTrace()
        println(breeze.config.GenerateHelp[JointParams[Params]](config))
        sys.exit(1)
    }

    if(params.help) {
      println(breeze.config.GenerateHelp[JointParams[Params]](config))
      System.exit(1)
    }
    import params._
    import params.trainer._
    println("Training Parser...")
    println(params)
    val transformed = params.treebank.trainTrees.par.map { ti => annotator(ti) }.seq.toIndexedSeq
    val grammar: SimpleRefinedGrammar[AnnotatedLabel, AnnotatedLabel, String] = GenerativeParser.extractGrammar(AnnotatedLabel.TOP, transformed)

    val kern = fromSimpleGrammar[AnnotatedLabel, AnnotatedLabel, String](grammar, params.trainer.useGPU, numGrammars)
    val train = transformed.slice(0,numToParse)


    println("Parsing...")
    val timeIn = System.currentTimeMillis()
    val trees = kern.partitions(train.map(_.words.toIndexedSeq))
//    for( (guess, inst) <- trees zip train) {
//      println("========")
//      println(guess.render(inst.words, false))
//      println(inst.tree.render(inst.words, false))
//    }
    println("Done: " + (System.currentTimeMillis() - timeIn))

    val timeX = System.currentTimeMillis()
    val marg = train.map(_.words).map { s =>
      val m = ChartMarginal(AugmentedGrammar.fromRefined(grammar), s)
      val counts = m.logPartition
      counts
    }
    println("Done: " + (System.currentTimeMillis() - timeX))
    println(marg.sum + " " + trees.sum)
//    println( Encoder.fromIndex(grammar.grammar.index).decode(trees2.rules.reduce(_ + _) - marg.slice(0, grammar.grammar.index.size)))
//    println( trees2.rules.reduce(_ + _) - marg.slice(0, grammar.grammar.index.size))

//    println(Encoder.fromIndex(grammar.grammar.index).decode(marg.slice(0, grammar.grammar.index.size)))
//    def unroll(m: ChartMarginal[ParseChart.LogProbabilityParseChart, AnnotatedLabel, String]) = {
//      for(l <- 0 until grammar.labelIndex.size; ref <- grammar.refinements.labels.localRefinements(l)) yield{
//        m.outside.top(0,1,l, ref)
//      }
//      m.partition
//    }
//    println(marg)
  }

  def fromSimpleGrammar[L, L2, W](grammar: SimpleRefinedGrammar[L, L2, W], useGPU: Boolean = true, numGrammars: Int = 1) = {
    import grammar.refinedGrammar._

    implicit val context = if(useGPU) {
      JavaCL.createBestContext()
    } else {
      val cpuPlatform:CLPlatform = JavaCL.listPlatforms().filter(_.listCPUDevices(true).nonEmpty).head
      cpuPlatform.createContext(new java.util.HashMap(), cpuPlatform.listCPUDevices(true):_*)
    }
    println(context)

    val rscores = RuleScores.fromRefinedGrammar(grammar)
    val grammars = new Array[RuleScores](numGrammars)
    util.Arrays.fill(grammars.asInstanceOf[Array[AnyRef]], rscores)
    val scorers = Array.fill(numGrammars){ (w: IndexedSeq[W], pos: Int, label: Int) =>
      grammar.anchor(w).scoreSpan(pos, pos+1, label, 0)
    }

    val kern = new GPUParser(grammar.grammar, grammar.refinements, grammar.refinedGrammar, grammar.lexicon.flatMap(grammar.refinements.labels.refinementsOf _), grammars, scorers)

    kern
  }


}

