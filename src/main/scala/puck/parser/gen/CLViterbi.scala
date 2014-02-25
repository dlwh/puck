package puck.parser.gen

import puck.parser._
import epic.trees.{BinaryRule, UnaryRule}
import java.util.zip.{ZipOutputStream, ZipFile}
import com.nativelibs4java.opencl._
import puck.util.ZipUtil
import puck.linalg.CLMatrix
import org.bridj.Pointer
import epic.trees.BinaryRule
import epic.trees.UnaryRule
import puck.PointerFreer
import scala.Array
import java.nio.{FloatBuffer, IntBuffer}
import epic.trees.BinaryRule
import epic.trees.UnaryRule
import puck.parser.SymId
import puck.parser.RuleStructure
import breeze.linalg._

/**
 * Implement's Canny's viterbi algorithm
 *
 * @author dlwh
 */
case class CLViterbi(wgSize: Array[Int], kernel: CLKernel,
                     parentOffsets: Array[Int],
                     ruleScores: Array[Float],
                     ruleLefts: Array[Int],
                     ruleRights: Array[Int],
                     BinaryOffsetsNN: Int, BinaryOffsetsNT: Int, BinaryOffsetsTN: Int,
                     BinaryOffsetsTT: Int, UnaryOffsets: Int, UnaryOffsetsT: Int)(implicit context: CLContext) {
  private val parentOffsetsDev = context.createIntBuffer(CLMem.Usage.Input, IntBuffer.wrap(parentOffsets), true)
  private val ruleScoresDev = context.createFloatBuffer(CLMem.Usage.Input, FloatBuffer.wrap(ruleScores), true)
  private val ruleLeftsDev = context.createIntBuffer(CLMem.Usage.Input, IntBuffer.wrap(ruleLefts), true)
  private val ruleRightsDev = context.createIntBuffer(CLMem.Usage.Input, IntBuffer.wrap(ruleRights), true)

  def viterbi(structure: RuleStructure[_, _],
              masks: CLMatrix[Int],
              inside: CLMatrix[Float],
              chartIndices: Array[Int],
              lengths: Array[Int],
              root: Int,
              events: CLEvent*)(implicit queue: CLQueue):CLEvent = {
    require(masks.cols == inside.cols)
    queue.finish()
//    return cpuViterbi(structure, masks,inside,chartIndices, lengths, root, events:_*)



    val ptrCI = Pointer.pointerToArray[java.lang.Integer](chartIndices)
    val intBufferCI = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, chartIndices.length)
    val evCI = intBufferCI.write(queue, 0, chartIndices.length, ptrCI, false, events:_*)

    val ptrL = Pointer.pointerToArray[java.lang.Integer](lengths)
    val intBufferL = queue.getContext.createIntBuffer(CLMem.Usage.InputOutput, lengths.length)
    val evL = intBufferL.write(queue, 0, lengths.length, ptrL, false, events:_*)

    kernel.setArgs(masks.data.safeBuffer,
      inside.data.safeBuffer, intBufferCI, intBufferL,
      Integer.valueOf(lengths.length), Integer.valueOf(inside.rows),
      Integer.valueOf(root), parentOffsetsDev, ruleScoresDev, ruleLeftsDev, ruleRightsDev)

    val ev = kernel.enqueueNDRange(queue, Array(lengths.length* wgSize(0), wgSize(1), wgSize(2)), wgSize, evCI, evL)
    PointerFreer.enqueue(ptrCI.release(), ev)
    PointerFreer.enqueue(intBufferCI.release(), ev)

    PointerFreer.enqueue(ptrL.release(), ev)
    PointerFreer.enqueue(intBufferL.release(), ev)
    ev
  }


  def write(out: ZipOutputStream) {
    ZipUtil.addKernel(out, "computeViterbiKernel", kernel)
    ZipUtil.serializedEntry(out, "ViterbiWGSize", wgSize)
    ZipUtil.serializedEntry(out, "ViterbiParents", parentOffsets)
    ZipUtil.serializedEntry(out, "ViterbiLeft", ruleLefts)
    ZipUtil.serializedEntry(out, "ViterbiRight", ruleRights)
    ZipUtil.serializedEntry(out, "ViterbiScores", ruleScores)
    ZipUtil.serializedEntry(out, "ViterbiOffsets", Array(BinaryOffsetsNN, BinaryOffsetsNT, BinaryOffsetsTN, BinaryOffsetsTT, UnaryOffsets, UnaryOffsetsT))
  }


  val NN = 0
  val NT = -1
  val TN = -2
  val TT = -3

  def bestBinary(
    info: SplitInfo,
    leftCell: DenseVector[Float],
    rightCell: DenseVector[Float],
    parent: Int,
    groupOffset: Int,
    split: Int) = {
    val firstRule = parentOffsets(parent + groupOffset)
    val lastRule = parentOffsets(parent + 1 + groupOffset)

    var bestScore = info.score
    for( r <- firstRule until lastRule) {
      val lc = ruleLefts(r)
      val rc = ruleRights(r)
      val rScore = ruleScores(r)
      val lcScore = leftCell(lc)
      val rcScore = rightCell(rc)
      if (rScore + lcScore + rcScore >= bestScore) {
        bestScore = rScore + lcScore + rcScore;
        info.left = lc;
        info.right = rc;
        info.split = split;
        info.score = bestScore;
      }
    }

  }

  def bestUnary(
    childCell: DenseVector[Float],
    parent: Int,
    groupOffset: Int) = {
    val firstRule = parentOffsets(parent + groupOffset)
    val lastRule = parentOffsets(parent + 1 + groupOffset)

    var bestScore = -3000.0f
    var bestSym = 0
    for( r <- firstRule until lastRule) {
      val lc = ruleLefts(r)
      val rScore = ruleScores(r)
      val lcScore = childCell(lc)
      if (rScore + lcScore>= bestScore) {
        bestScore = rScore + lcScore
        bestSym = lc
      }
    }
    assert(childCell(bestSym) != -Float.NegativeInfinity)

    bestSym

  }


  def cpuViterbi(structure: RuleStructure[_, _],
                  masks: CLMatrix[Int],
                 insides: CLMatrix[Float],
                 chartIndices: Array[Int],
                 lengths: Array[Int],
                 root: Int,
                 events: CLEvent*)(implicit queue: CLQueue):CLEvent = {
    CLEvent.waitFor(events:_*)
    import ChartHalf.{chartIndex=>cellIndex}

    for(s <- 0 until lengths.length) {
      val length = lengths(s)
      val insideBot = insides(::, chartIndices(s) until chartIndices(s + 1)).toDense
      val insideTop = insideBot(::, (length + 1) * length  / 2 until chartIndices(s + 1))
      val tree = masks(::, chartIndices(s) until chartIndices(s + 1)).toDense

      def ttop(p: Int) = tree(0, p)
      def tbot(p: Int) = tree(1, p)
      def twidth(p: Int) = tree(2, p)
      def tscore(p: Int) = tree(3, p)

      def stop(p: Int, x: Int) = tree(0, p) = x
      def sbot(p: Int, x: Int) = tree(1, p) = x
      def swidth(p: Int, x: Int) = tree(2, p) = x
      def sscore(p: Int, x: Float) = tree(3, p) = java.lang.Float.floatToRawIntBits(x)

      stop(0, root)
      swidth(0, length)

      var begin = 0

      for(p <- 0 until 2 * length - 1) {
        val width = twidth(p)
        val end = begin + width
        var botSym = tbot(p)

        if(ttop(p) != -1) {
          val unaryOff = if(width == 1) UnaryOffsetsT else UnaryOffsets

          botSym = bestUnary(insideBot(::, cellIndex(begin, end, length)), ttop(p), unaryOff);
          if(width != 1)
          println(s"($begin, $end) Unary from ${structure.nontermIndex.get(ttop(p))} to ${structure.nontermIndex.get(botSym)}")
          else
            println(s"($begin, $end) Unary from ${structure.nontermIndex.get(ttop(p))} to ${structure.termIndex.get(botSym)}")
          sbot(p, botSym)
        }

        if(width == 1) {
          println(s"$begin term ${structure.termIndex.get(botSym)}")
          sscore(p, insideBot(botSym, begin))
          begin += 1
        } else {
          println(s"${(begin, end)} ${insideBot(botSym, cellIndex(begin, end, length))} ${structure.nontermIndex.get(botSym)}")
          val info = SplitInfo(0, 0, begin + 1, -30000f)
           bestBinary(info, insideTop(::, cellIndex(begin, end -1, length)),
            insideBot(::, cellIndex(end -1, end, length)),
            botSym, BinaryOffsetsNT, NT)


           bestBinary(info, insideBot(::, cellIndex(begin, begin + 1, length)),
            insideTop(::, cellIndex(begin + 1, end, length)),
            botSym, BinaryOffsetsTN, TN)


          if(width == 2)
             bestBinary(info, insideBot(::, cellIndex(begin, end -1, length)),
              insideBot(::, cellIndex(end -1, end, length)),
              botSym, BinaryOffsetsTT, TT);


          for(split <- begin + 1 until end)
             bestBinary(info, insideTop(::, cellIndex(begin, split, length)),
              insideTop(::, cellIndex(split, end, length)),
              botSym, BinaryOffsetsNN, split)

          val best= info
          sscore(p, best.score)
          var bestSplit = best.split
          val bestConfig = if(bestSplit < 0) -bestSplit  else NN

          bestSplit = if(bestConfig == NN) bestSplit else if(bestConfig == -TN) begin + 1 else end - 1
          assert(bestSplit > begin && bestSplit < end)

          val lsym = if( (bestConfig >> 1) == 0)  structure.nontermIndex.get(info.left) else structure.termIndex.get(info.left)
          val rsym = if( (bestConfig & 1) == 0)  structure.nontermIndex.get(info.right) else structure.termIndex.get(info.right)
          println(s"($begin, $end) Binary ${structure.nontermIndex.get(botSym)} to $lsym($begin, $bestSplit) $rsym($bestSplit, $end) mode $bestConfig, score: ${info.score}")

          if ( (bestConfig >> 1) == 0) { // Left N
            stop(p+1, info.left)
          } else {
            stop(p + 1, -1)
            sbot(p + 1, info.left)
          }

          if( (bestConfig & 1) == 0) {
            stop(p + 2 * (bestSplit - begin), info.right)
          } else {
            stop(p + 2 * (bestSplit - begin), -1)
            sbot(p + 2 * (bestSplit - begin), info.right)
          }

          swidth(p + 1, bestSplit - begin)
          swidth(p + 2 * (bestSplit - begin), end - bestSplit)

        }
      }

      masks(::, chartIndices(s) until chartIndices(s + 1)) := tree

    }

    null
  }


  private case class SplitInfo(var left: Int, var right: Int, var split: Int, var score: Float)
}

object CLViterbi {
  def read(zf: ZipFile)(implicit ctxt: CLContext) = {
    val wgSize = ZipUtil.deserializeEntry[Array[Int]](zf.getInputStream(zf.getEntry("ViterbiWGSize")))
    val offsets@Array(binaryOffsetsNN, binaryOffsetsNT, binaryOffsetsTN, binaryOffsetsTT, unaryOffsets, unaryOffsetsT) = ZipUtil.deserializeEntry[Array[Int]](zf.getInputStream(zf.getEntry("ViterbiOffsets")))
    val parentOffsets = ZipUtil.deserializeEntry[Array[Int]](zf.getInputStream(zf.getEntry("ViterbiParents")))
    val ruleLefts = ZipUtil.deserializeEntry[Array[Int]](zf.getInputStream(zf.getEntry("ViterbiLeft")))
    val ruleRights = ZipUtil.deserializeEntry[Array[Int]](zf.getInputStream(zf.getEntry("ViterbiRight")))
    val ruleScores = ZipUtil.deserializeEntry[Array[Float]](zf.getInputStream(zf.getEntry("ViterbiScores")))
    CLViterbi(wgSize,
      ZipUtil.readKernel(zf, "computeViterbiKernel"), parentOffsets, ruleScores, ruleLefts, ruleRights,
    binaryOffsetsNN, binaryOffsetsNT, binaryOffsetsTN, binaryOffsetsTT, unaryOffsets, unaryOffsetsT)
  }


  def make[C, L](structure: RuleStructure[C, L])(implicit context: CLContext, semiring: RuleSemiring) = {
    val blockSize = 32

    val wgSize = if (true) { //context.getDevices.head.toString.contains("Apple") && context.getDevices.head.toString.contains("Intel Core")) {
      Array(1, 1, 1)
    } else {
      val wgSizes = context.getDevices.head.getMaxWorkItemSizes
      val x = wgSizes(0) min blockSize
      Array(1, x toInt, 1)
    }

    val g = structure

    val (unaryParents, unaryScores, unaryChildren) = makeUnaryRuleArrays(g, g.unaryRules)
    val (unaryParentsT, unaryScoresT, unaryChildrenT) = makeUnaryRuleArrays(g, g.unaryTermRules)
    val (binaryParentsNN, binaryScoresNN, binaryLeftNN, binaryRightNN) = makeBinaryRuleArrays(g, g.nontermRules)
    val (binaryParentsNT, binaryScoresNT, binaryLeftNT, binaryRightNT) = makeBinaryRuleArrays(g, g.rightTermRules)
    val (binaryParentsTN, binaryScoresTN, binaryLeftTN, binaryRightTN) = makeBinaryRuleArrays(g, g.leftTermRules)
    val (binaryParentsTT, binaryScoresTT, binaryLeftTT, binaryRightTT) = makeBinaryRuleArrays(g, g.bothTermRules)

    val offsets = Seq(binaryParentsNN, binaryParentsNT, binaryParentsTN, binaryParentsTT, unaryParents, unaryParentsT).foldLeft(Array(0)) { (offsets, parents) =>
      val start = offsets.last
      val myOffsets: Array[Int] = mkOffsetsForSortedArray(g, parents, start)
      assert(myOffsets.last - start == parents.length,myOffsets.last + " " + start +  " " + parents.toIndexedSeq)

      offsets ++ myOffsets
    }

    assert(offsets.length == g.nontermIndex.size * 6 + 1)

    val groupOffsets = Array.tabulate(6)(g.nontermIndex.size * _)
    println(groupOffsets.toIndexedSeq)
    val scores = epic.util.Arrays.concatenate(binaryScoresNN, binaryScoresNT, binaryScoresTN, binaryScoresTT, unaryScores, unaryScoresT)
    val left =   epic.util.Arrays.concatenate(binaryLeftNN,   binaryLeftNT,   binaryLeftTN,   binaryLeftTT,  unaryChildren, unaryChildrenT)
    val right =  epic.util.Arrays.concatenate(binaryRightNN,  binaryRightNT,  binaryRightTN,  binaryRightTT)
    val parents =  epic.util.Arrays.concatenate(binaryParentsNN,  binaryParentsNT,  binaryParentsTN,  binaryParentsTT)
    assert(offsets.last == scores.length,offsets.last + " " + scores.length)


    val offsetNames: Seq[String] = Seq(
      "BINARY_OFFSET_NN", "BINARY_OFFSET_NT", "BINARY_OFFSET_TN",
      "BINARY_OFFSET_TT",
      "UNARY_OFFSET", "UNARY_OFFSET_T")
    val defs = for((off, name) <- groupOffsets zip offsetNames) yield {
      s"#define $name $off"
    }

    verify(g.nontermIndex.size, parents, groupOffsets(0), offsets, 0)
    verify(g.nontermIndex.size, parents, groupOffsets(1), offsets, binaryParentsNN.length)
    verify(g.nontermIndex.size, parents, groupOffsets(2), offsets, binaryParentsNN.length + binaryParentsNT.length)
    verify(g.nontermIndex.size, parents, groupOffsets(3), offsets, binaryParentsNN.length + binaryParentsNT.length + binaryParentsTN.length)
    val ssym = structure.nontermIndex.toIndexedSeq.indexWhere(_.toString == "S_0")
    println(ssym)
    println(structure.bothTermRules.filter(_._1.parent.gpu == ssym))
    println(binaryParentsTT.count(_ == ssym))
    println(parents.slice(offsets(groupOffsets(3) + ssym), offsets(groupOffsets(3) + ssym + 1)).toIndexedSeq)


    val fullSource = (
        defs.mkString("\n")
        + source(structure, wgSize)
      )

    val prog = context.createProgram(fullSource)

    CLViterbi(wgSize, prog.createKernel("viterbi"), offsets, scores, left, right,
      groupOffsets(0),
      groupOffsets(1),
      groupOffsets(2),
      groupOffsets(3),
      groupOffsets(4),
      groupOffsets(5)
    )
  }


  private def mkOffsetsForSortedArray[L, C](g: RuleStructure[C, L], parents: Array[Int], start: Int): Array[Int] = {
    val myOffsets = Array.tabulate(g.nontermIndex.size) {
      sym =>
        var startOfBlock = parents.indexWhere(_ > sym)
        if (startOfBlock == -1) startOfBlock = parents.length
        startOfBlock + start
    }
    myOffsets
  }

  private def verify(numSyms: Int, array: Array[Int], goff: Int, offsets: Array[Int], cnt: Int) {
    for(i <- 0 until numSyms) {
      assert(array.slice(offsets(i), offsets(i+1)).forall(_ == i), i + " " + array.slice(offsets(i), offsets(i+1)).toIndexedSeq)

    }
  }



  private def source[C, L](g: RuleStructure[C, L], wgSize: Array[Int]) = {

    ("""

// TODO: silly global accesses to tree.


inline __global const float* CELL(__global const float* chart, int cellSize, int begin, int end, int length)  {
  int span = end-begin-1;
  return chart + cellSize * (begin + span * length - span * (span - 1) / 2);
}

inline int CHART_SIZE(int dim) { return dim * (dim + 1) / 2; }

typedef struct { int topSym, botSym, width; float score; } tree_t;
typedef struct { int left, right, split; float score; } splitInfo;

#define WG_SIZE %d

static int BestUnary(__global const float* insideBot, int parent,
    __constant const int* unaryOffsets,
    __global const float* unaryScores,
    __global const int* unaryChildren,
    __local float* bestScores,
    __local int* bestSyms, __global float* score) {
  int tid = get_local_id(1);
  int first_rule = unaryOffsets[parent];
  int last_rule = unaryOffsets[parent + 1];

  float bestScore = -30000.0f;
  int bestSym = 0;

  for(int r = first_rule + tid; r < last_rule; r += WG_SIZE) {
    int child = unaryChildren[r];
    float rScore = unaryScores[r];
    float cScore = insideBot[child];
    if (rScore + cScore >= bestScore) {
      bestScore = rScore + cScore;
      bestSym = child;
    }
  }

  bestScores[tid] = bestScore;
  bestSyms[tid] = bestSym;
  barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
  for(int i = WG_SIZE >> 1; i > 0; i = i >> 1) {
    if (tid < i) {
      float score = bestScores[tid + i];
      if (score > bestScores[tid]) {
        bestScores[tid] = score;
        bestSyms[tid] = bestSyms[tid + i];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(tid ==0) *score = bestScores[0];

  return bestSyms[0];
}

static splitInfo BestSplit(splitInfo myInfo,
  __local float* bestScores,
  __local int* bestLefts,
  __local int* bestRights,
                           __local int* bestSplits) {
  int tid = get_local_id(1);

  bestScores[tid] = myInfo.score;
  bestLefts[tid] = myInfo.left;
  bestRights[tid] = myInfo.right;
  bestSplits[tid] = myInfo.split;
  barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
  for(int i = WG_SIZE >> 1; i > 0; i = i >> 1) {
    if (tid < i) {
      float score = bestScores[tid + i];
      if (score > bestScores[tid]) {
        bestScores[tid] = score;
        bestLefts[tid] = bestLefts[tid + i];
        bestRights[tid] = bestRights[tid + i];
        bestSplits[tid] = bestSplits[tid + i];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  return (splitInfo){bestLefts[0], bestRights[0], bestSplits[0], bestScores[0]};
}

#define NN 0
#define NT -1
#define TN -2
#define TT -3

static void BestBinary(
    splitInfo* info,
    __global const float* leftCell,
    __global const float* rightCell,
    int parent,
    __constant const int* binaryOffsets,
    __global const int* binaryLeft,
    __global const int* binaryRight,
    __global const float* binaryScores,
    int split) {
  int tid = get_local_id(1);
  int first_rule = binaryOffsets[parent];
  int last_rule = binaryOffsets[parent + 1];

  float bestScore = info->score;
  for(int r = first_rule + tid; r < last_rule; r += WG_SIZE) {
    int lc = binaryLeft[r];
    int rc = binaryRight[r];
    float rScore = binaryScores[r];
    float lcScore = leftCell[lc];
    float rcScore = rightCell[rc];
    if (rScore + lcScore + rcScore >= bestScore) {
      bestScore = rScore + lcScore + rcScore;
      info->left = lc;
      info->right = rc;
      info->split = split;
      info->score = bestScore;
    }
  }

}


__attribute__((reqd_work_group_size(""" + wgSize.mkString(", ")
      + """)))
__kernel void viterbi(__global tree_t* treeOut, __global float* insides,
    __global const int* cellOffsets, __global const int* lengths,
    int numSentences, int cellSize, int root,
    __constant const int* parentOffsets,
    __global const float* ruleScores,
    __global const int* ruleLefts,
    __global const int* ruleRights) {
  __local float bestScores[WG_SIZE];
  __local int bestLefts[WG_SIZE];
  __local int bestRights[WG_SIZE];
  __local int bestSplits[WG_SIZE];
  int tid = get_local_id(1);
  for(int sent = get_global_id(0); sent < numSentences; sent += get_global_size(0)) {
    __global tree_t* tree = treeOut + cellOffsets[sent];
    int length = lengths[sent];
    __global float* insideBot = insides + cellOffsets[sent] * cellSize;
    __global float* insideTop = insideBot + CHART_SIZE(length) * cellSize;

    if (tid == 0) {
      tree[0].topSym = root;
      tree[0].width = length;
    }

    int begin = 0; // current leftmost position for span

    for (int p = 0; p < 2 * length - 1; p++) {
      int width = tree[p].width;
      int end = begin + width; // rightmost.
      int botSym = tree[p].botSym;

      // if -1, then we're skipping the unary layer and going straight to binary/terminal
      if (tree[p].topSym != -1) {
        int unaryOff = (width == 1) ? UNARY_OFFSET_T : UNARY_OFFSET;
        botSym = BestUnary(CELL(insideBot, cellSize, begin, end, length), tree[p].topSym,
            parentOffsets + unaryOff, ruleScores, ruleLefts, bestScores, bestLefts, &tree[p].score);

        if (tid == 0)
          tree[p].botSym = botSym;
      }

      if (width == 1) {
        tree[p].score = insideBot[cellSize * begin + botSym];
        // terminal, move on.
        begin += 1;
      } else {
        splitInfo info = (splitInfo){ 10, 10, begin + 1, -300000.0f };
        BestBinary(&info,
            CELL(insideTop, cellSize, begin, end - 1, length),
            CELL(insideBot, cellSize, end - 1, end, length),
            botSym, parentOffsets + BINARY_OFFSET_NT, ruleLefts, ruleRights, ruleScores, NT);

        BestBinary(&info,
            CELL(insideBot, cellSize, begin, begin + 1, length),
            CELL(insideTop, cellSize, begin + 1, end, length),
            botSym, parentOffsets + BINARY_OFFSET_TN, ruleLefts, ruleRights, ruleScores, TN);


        if (width == 2)
          BestBinary(&info,
              CELL(insideBot, cellSize, begin, begin + 1, length),
              CELL(insideBot, cellSize, begin + 1, end, length),
              botSym, parentOffsets + BINARY_OFFSET_TT, ruleLefts, ruleRights, ruleScores, TT);


        for(int split = begin + 1; split < end; split++) {
          __global const float* leftCell = CELL(insideTop, cellSize, begin, split, length);
          __global const float* rightCell = CELL(insideTop, cellSize, split, end, length);
          BestBinary(&info,
              leftCell,
              rightCell,
              botSym, parentOffsets + BINARY_OFFSET_NN, ruleLefts, ruleRights, ruleScores, split);
        }

        splitInfo best = BestSplit(info, bestScores, bestLefts, bestRights, bestSplits);


        if (tid == 0) {
          tree[p].score = best.score;
          int bestSplit = best.split;
          int bestConfig = bestSplit < 0 ? -bestSplit : NN;

          bestSplit = (bestConfig == NN) ? bestSplit
            : bestConfig == -TN ? begin + 1
            : end - 1; // TT or NT

          if ( (bestConfig >> 1) == 0) {// Left N
            tree[p + 1].topSym = info.left;
          } else { // Left T
            tree[p + 1].topSym = -1;
            tree[p + 1].botSym = info.left;
          }

          if ((bestConfig & 1) == 0) { // Right N
            tree[p + 2 * (bestSplit - begin)].topSym = info.right;
          } else { // Right T
            tree[p + 2 * (bestSplit - begin)].topSym = -1;
            tree[p + 2 * (bestSplit - begin)].botSym = info.right;
          }

          tree[p + 1].width = bestSplit - begin;
          tree[p + 2 * (bestSplit-begin)].width = end - bestSplit;
        }

      }

    }

  }
}

        """).stripMargin.format(wgSize(1))
  }

  private def makeUnaryRuleArrays[L, C](g: RuleStructure[C, L], rules: IndexedSeq[(UnaryRule[SymId[C, L]], Int)]): (Array[Int], Array[Float], Array[Int]) = {
    val triples = for ((rule, id) <- rules) yield {
      (rule.parent.gpu, rule.child.gpu, g.scores(id))
    }

    val (unaryParents, unaryChildren, unaryScores) = triples.sortBy(_._1).toArray.unzip3
    (unaryParents.toArray, unaryScores.toArray, unaryChildren.toArray)
  }

  private def makeBinaryRuleArrays[L, C](g: RuleStructure[C, L], rules: IndexedSeq[(BinaryRule[SymId[C, L]], Int)]): (Array[Int], Array[Float], Array[Int], Array[Int]) = {
    val triples = for ((rule, id) <- rules) yield {
      (rule.parent.gpu, (rule.left.gpu, rule.right.gpu), g.scores(id))
    }

    val (binaryParents, binaryChildren, binaryScores) = triples.sortBy(_._1).toArray.unzip3

    val (left, right) = binaryChildren.toArray.unzip
    (binaryParents.toArray, binaryScores.toArray, left.toArray, right.toArray)
  }
}
