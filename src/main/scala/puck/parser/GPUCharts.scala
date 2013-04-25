package puck.parser

import com.nativelibs4java.opencl.{CLQueue, CLEvent, CLContext, CLBuffer}
import java.lang.{Float=>JFloat}
import com.nativelibs4java.opencl.CLMem.Usage
import puck.util.{MemBufPair, ZeroMemoryKernel}
import breeze.collection.mutable.TriangularArray

object Scaling {
  def SCALE_FACTOR = 10
}

/**
 *
 * @author dlwh
 */
case class GPUCharts[C, L](top: MemBufPair[Float],
                           bot: MemBufPair[Float],
                           tags: MemBufPair[Float],
                           struct: RuleStructure[C, L],
                           numGrammars: Int,
                           maxCells: Int,
                           maxTotalLength: Int) {
  def clear(events: CLEvent*)(implicit context: CLContext, queue: CLQueue) = setTo(0.0f, events:_*)

  def setTo(fl: Float, events: CLEvent*)(implicit context: CLContext, queue: CLQueue) = {
    val zmk = ZeroMemoryKernel(0.0f)
    var a  = zmk.fillMemory(top, fl, events:_*)
    a = zmk.fillMemory(bot, fl, Seq(a) ++  events:_*)
    zmk.fillMemory(tags, fl, Seq(a) ++  events:_*)
  }

  def release() {
    top.release()
    bot.release()
    tags.release()
  }

  def scoresForBatch(offsets: Array[Int], lengthOffsets: Array[Int])(implicit queue: CLQueue) = {
    new Scores(top.data, bot.data, tags.data, offsets, lengthOffsets)
  }


  case class Scores(top: Array[Float], bot: Array[Float], tags: Array[Float], offsets: Array[Int], lengthOffsets: Array[Int]) {
    def apply(sent: Int) = new SentenceChart(top, bot, tags, offsets(sent), lengthOffsets(sent), lengthOffsets(sent+1)-lengthOffsets(sent))
  }

  case class SentenceChart(top: Array[Float], bot: Array[Float], tags: Array[Float], offset: Int, lengthOff: Int, length: Int) {
    def top(begin: Int, end: Int, grammar: Int, label: Int):Double =  {
      val score = top(index(begin, end, grammar, label))
      //      java.lang.Math.scalb(score, -10 * ((end-begin)-1))
      score
//      math.log(score) - Scaling.SCALE_FACTOR * ((end-begin)-1) * math.log(2)
    }


    def bot(begin: Int, end: Int, grammar: Int, label: Int):Double =  {
      val score = bot(index(begin, end, grammar, label))
      //      java.lang.Math.scalb(score, -10 * ((end-begin)-1))
      //      math.log(score)
      score
//      math.log(score) - Scaling.SCALE_FACTOR * ((end-begin)-1) * math.log(2)
    }

    def tag(begin: Int, grammar: Int, label: Int) = {
      tags(label  * maxCells +(offset + begin)) * numGrammars + grammar
    }

    private def index(begin: Int, end: Int, grammar: Int, label: Int): Int = {
      (label  * maxCells +(offset + TriangularArray.index(begin, end))) * numGrammars + grammar
    }

    def rootScore(grammar: Int) = top(0, length, grammar, struct.root)
  }
}


object GPUCharts {
  def forGrammar[C, L](structure: RuleStructure[C, L], numGrammars: Int, maxCells: Int, maxTotalLength: Int)(implicit context: CLContext, queue: CLQueue) = {
    val cellSize = structure.numNonTerms * numGrammars
    val insideTopDev, insideBotDev = MemBufPair[Float](maxCells * cellSize, Usage.InputOutput)
    val posTagsDev = MemBufPair[Float](maxTotalLength * cellSize, Usage.Input)

    new GPUCharts(insideTopDev, insideBotDev, posTagsDev, structure, numGrammars, maxCells, maxTotalLength)
  }

  def computeMaxSizes[C, L](totalBytes: Long, maxAllocSize: Long, structure: RuleStructure[C, L], numGrammars: Int) = {
    val cellSize = structure.numNonTerms * numGrammars
    val posSize = structure.numTerms * numGrammars
    // XXX TODO
    val maxCells = ((totalBytes / cellSize).toInt / 4 / 2)  min 100000
    val maxTotalLength = (maxAllocSize / structure.numRules / 4).toInt min (maxCells * posSize / structure.numRules) min 50000

    maxCells -> maxTotalLength
  }



}
