package puck.parser.gen

import puck.parser.RuleStructure

/**
 * Implement's Canny's viterbi algorithm
 *
 * @author dlwh
 */
class CLViterbi {

}

object CLViterbi {
  private def source[C, L](g: RuleStructure[C, L], wgSize: Int) = {
    val unaryOffsets = ???
    val unaryScores = ???
    val unaryChildren = ???
    val binaryOffsets = ???
    val binaryScores = ???
    val binaryLeft = ???
    val binaryRight = ???
    """
      | // TODO: silly global accesses to tree.
      |
      | typedef struct { int top, bot, width, unused; } tree_t;
      | typedef struct { int left, right, split, score; } splitInfo;
      |
      | #define WG_SIZE %d
      |
      | static int BestUnary(__global const float* insideBot, int parent) {
      |   int tid = get_local_id(1);
      |   int first_rule = unaryOffsets[parent];
      |   int last_rule = unaryOffsets[parent + 1];
      |
      |   __local float bestScores[WG_SIZE];
      |   __local int bestSyms[WG_SIZE];
      |
      |   float bestScore = -300f;
      |   int bestSym = 0;
      |
      |   for(int r = first_rule; r < last_rule; r += WG_SIZE) {
      |     int child = unaryChildren[r];
      |     float rScore = unaryScore[r];
      |     float cScore = insideBot[child];
      |     if(rScore + cScore >= bestScore) {
      |       bestScore = rScore + cScore;
      |       bestSym = child;
      |     }
      |   }
      |
      |   bestScores[tid] = bestScore;
      |   bestSyms[tid] = bestSym;
      |   barrier(CLK_LOCAL_MEM_FENCE);
      |
      |   #pragma unroll
      |   for(int i = WG_SIZE >> 1; i > 0; i = i >> 1) {
      |     if(tid < i) {
      |       float score = bestScores[tid + i];
      |       if (score > bestScores[tid]) {
      |         bestScores[tid] = score;
      |         bestSyms[tid] = bestSyms[tid + i];
      |       }
      |     }
      |     barrier(CLK_LOCAL_MEM_FENCE);
      |   }
      |
      |   return bestSyms[0];
      | }
      |
      |SELECT rule array based on NT/NN/TT/TN by looking at (end-split) and (split-begin)
      |static splitInfo BestBinaryNN(__global const float* insideTop, int parent, int begin, int end, int length, int cellSize) {
      |   int tid = get_local_id(1);
      |   int first_rule = binaryOffsets[parent];
      |   int last_rule = binaryOffsets[parent + 1];
      |
      |   __local float bestScores[WG_SIZE];
      |   __local int bestLefts[WG_SIZE];
      |   __local int bestRights[WG_SIZE];
      |   __local int bestSplits[WG_SIZE];
      |
      |   float bestScore = -300f;
      |   int bestLeft = 0, bestRight = 0, bestSplit = 0;
      |
      |   for(int split = begin + 1; split < end; split += 1) {
      |     for(int r = first_rule; r < last_rule; r += WG_SIZE) {
      |       int lc = binaryLeft[r];
      |       int rc = binaryRight[r];
      |       float rScore = binaryScores[r];
      |       float lcScore = leftCell[lc];
      |       float rcScore = rightCell[rc];
      |       if(rScore + lcScore + rcScore >= bestScore) {
      |         bestScore = rScore + lcScore + rcScore;
      |         bestLeft = lc;
      |         bestRight = rc;
      |         bestSplit = split;
      |       }
      |     }
      |   }
      |
      |   bestScores[tid] = bestScore;
      |   bestLefts[tid] = bestLeft;
      |   bestRights[tid] = bestRight;
      |   bestSplits[tid] = bestSplit;
      |   barrier(CLK_LOCAL_MEM_FENCE);
      |
      |   #pragma unroll
      |   for(int i = WG_SIZE >> 1; i > 0; i = i >> 1) {
      |     if(tid < i) {
      |       float score = bestScores[tid + i];
      |       if (score > bestScores[tid]) {
      |         bestScores[tid] = score;
      |         bestLefts[tid] = bestLefts[tid + i];
      |         bestRights[tid] = bestRights[tid + i];
      |         bestSplits[tid] = bestSplits[tid + i];
      |       }
      |     }
      |     barrier(CLK_LOCAL_MEM_FENCE);
      |   }
      |
      |   return { bestLefts[0], bestRights[0], bestSplits[0] };
      | }
      |
      | __kernel void viterbi(__global tree_t* treeOut, __global float* insides, __global const int* cellOffsets, __global const int* lengths, int numSentences, int cellSize) {
      |
      |   int tid = get_local_id(1);
      |   for(int sent = get_global_id(0); sent < numSentences; sent += get_global_size(0)) {
      |     __global tree_t* tree = treeOut + cellOffsets[sent];
      |     int length = lengths[sent];
      |     __global float* inside = insides + cellOffsets[sent] * cellSize;
      |
      |     if(tid == 0) {
      |       tree[0].top = ROOT;
      |       tree[0].width = length;
      |     }
      |
      |     int i = 0; // current leftmost position for span
      |
      |     for (int p = 0; p < 2 * length - 2; p += 1) {
      |       int j = i + tree[p].width; // rightmost.
      |       int bot;
      |       if(tree[p].top != -1 ) {
      |         bot = BestUnary(CELL(inside, cellSize, i, j, length), tree[p].top);
      |         if(tid == 0)
      |           tree[p].bot = bot;
      |        }
      |
      |       if (tree[p].width == 1) {
      |         i += 1;
      |       } else {
      |       #define NN 0
      |       #define NT 1
      |       #define TN 2
      |       #define TT 3
      |         int bestConfig = NN;
      |         splitInfo info = BestBinary(inside + CHART_SIZE(length + 1) * cellSize, bot, i, j, length, cellSize);
      |
      |         splitInfo ntInfo = BestBinaryNT( CELL(inside + CHART_SIZE(length + 1), cellSize, i, j - 1, length),
      |                       CELL(inside, cellSize, j - 1, j, length), bot, j - 1);
      |
      |         if(ntInfo.score > info.score) {
      |           bestConfig = NT;
      |           info = ntInfo;
      |         }
      |
      |
      |         splitInfo tnInfo = BestBinaryTN( CELL(inside, cellSize, i, i + 1, length),
      |                       CELL(inside + CHART_SIZE(length + 1), cellSize, i + 1, j, length), bot, i + 1);
      |
      |
      |         if(tnInfo.score > info.score) {
      |           bestConfig = TN;
      |           info = tnInfo;
      |         }
      |
      |
      |         if(j - i == 2) {
      |           splitInfo ttInfo = BestBinaryTT( CELL(inside, cellSize, i, i + 1, length),
      |                                            CELL(inside, cellSize, i + 1, j, length), bot, i + 1);
      |           if(ttInfo.score > info.score) {
      |             bestConfig = TT;
      |             info = ttInfo;
      |           }
      |         }
      |
      |         if(tid == 0) {
      |           if(bestConfig >> 1 == 0) {// Left N
      |             tree[p + 1].top = info.left;
      |           } else { // Left 2
      |             tree[p + 1].top = -1;
      |             tree[p + 1].bot = info.left;
      |           }
      |           tree[p + 1].width = info.split - i;
      |           if( (bestConfig & 1) == 0) { // Right N
      |             tree[p + 2 * (k-i)].top = info.right;
      |           } else {
      |             tree[p + 2 * (k-i)].top = -1;
      |             tree[p + 2 * (k-i)].bot = info.right;
      |           }
      |           tree[p + 2 * (k-i)].width = info.split - k;
      |         }
      |
      |       }
      |
      |     }
      |
      |   }
      | }
    """.stripMargin.format(wgSize)
  }
}
