package puck.parser.gen

import puck.parser._
import epic.AwesomeScalaBitSet
import com.nativelibs4java.opencl.CLContext
import epic.trees.BinaryRule
import epic.trees.UnaryRule
import puck.parser.SymId
import scala.Some
import puck.parser.RuleStructure
import scala.collection.immutable.BitSet

/**
 * TODO
 *
 * @author dlwh
class NoninlinedRuleMultiply[C, L](structure: RuleStructure[C, L])(implicit semiring: RuleSemiring) extends GenRuleMultiply[C, L] {
  def binaryRuleApplication(rules: IndexedSeq[(BinaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLBinaryRuleUpdater = {
    val ruleStruct = s"""
      |typedef struct { int parent, left, right; float ruleScore; } binary_rule;
    """.stripMargin

    val ints = rules.sortBy(_._1.parent.gpu).map { case (rule, id) => Array(rule.parent.gpu, rule.left.gpu, rule.right.gpu, java.lang.Float.floatToRawIntBits(structure.scores(id)))}.flatten.toArray

    val kernel = CLMaskKernels.maskHeader(structure) +
      s"""
       | $ruleStruct
       |
       | #define NUM_RULES ${rules.length}
       | #define RULES_PER_BLOCK 64
       |
       |__kernel void $name(__global float* parents, __global int* parentIndex,
       |                   __global float* left, __global int* leftIndex,
       |                    __global float* right, __global int* rightIndex,
       |                    __global const mask_t* masks,
       |                   int numRows, int cellsToDo, __global const binary_rule* rules) {
       |    int numWorkers = get_global_size(0);
       |    __local binary_rule block[RULES_PER_BLOCK];
       |for(int tid = get_local_id(0); tid < RULES_PER_BLOCK * (sizeof(binary_rule)/sizeof(int)); tid += get_local_size(0)) {
       |          ((__local int*)block)[tid] = ((__global const int*)(rules))[0 * (sizeof(binary_rule)/sizeof(int)) + tid];
       |        }
       |        barrier(CLK_LOCAL_MEM_FENCE);
       |
       |
       |      for (int rule = 0; rule < NUM_RULES; rule += RULES_PER_BLOCK) {
       |        int numRulesToDo = min(NUM_RULES - rule, RULES_PER_BLOCK);
       //|        for(int tid = get_local_id(0); tid < numRulesToDo * (sizeof(binary_rule)/sizeof(int)); tid += get_local_size(0)) {
       //|          ((__local int*)block)[tid] = ((__global const int*)(rules))[rule * (sizeof(binary_rule)/sizeof(int)) + tid];
       //|        }
       //|        barrier(CLK_LOCAL_MEM_FENCE);
       |
       |    for(int row = get_global_id(0); row < cellsToDo; row += numWorkers) {
       |        int lastParent = -1; float parentScore = 0.0f;
       |        for(int local_rule = 0; local_rule < numRulesToDo; local_rule += 1) {
       |          __local binary_rule* r = &block[local_rule];
       //|          __global binary_rule* r = &rules[rule + local_rule];
       //|          if(row == 0) printf("%d %d %d %d %d %d\\n",r->parent,r->left,r->right,r2->parent,r2->left,r2->right);
       |          int parent = r->parent;
       |          if(parent != lastParent) {
       |             parents[lastParent * numRows + row] = parentScore;
       |             parentScore = parents[parent * numRows + row];
       |             lastParent = parent;
       |          }
       |          parentScore = max(parentScore, left[r->left * numRows + row] + right[r->right * numRows + row] + r->ruleScore);
       |        }
       |          if(-1 != lastParent) {
       |             parents[lastParent * numRows + row] = parentScore;
       |          }
       |
       |      }
       |
       |    }
       |}
      """.stripMargin

    val kernels = cl.createProgram(kernel).createKernels()
    val parents = (BitSet.empty ++ rules.map(_._1.parent.coarse)).toJavaBitSet
    CLBinaryRuleUpdater(IndexedSeq(new RuleKernel(kernels, parents)), Array(32 * 40, 1, 1), Array(32, 1, 1), Some(ints))
  }


  def unaryRuleApplication(rules: IndexedSeq[(UnaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLUnaryRuleUpdater = {
    val partitions  : IndexedSeq[IndexedSeq[(UnaryRule[SymId[C, L]], Int)]] = unaryClusterer.partitionUnaries(rules).toIndexedSeq
    assert(partitions.map(_.length).sum == rules.length)
    val texts = partitions.zipWithIndex.map { case (rulePartition, partitionIndex) =>
      val parents = rulePartition.map(_._1.parent).toSet
      val parentVariables = parents.iterator.map(p => p.gpu -> Variable(s"parent_${p.gpu}", p.fineSym.toString)).toMap
      s"""
    __kernel void ${name}_$partitionIndex(__global float* parents, __global float* children, int numRows, int cellsToDo) {
        int numWorkers = get_global_size(0);
        for(int row = get_global_id(0); row < cellsToDo; row += numWorkers) {
          ${parentVariables.values.map(_.declare).mkString("\n        ")}
          ${coreUnaryRuleLoop(rulePartition, parentVariables)}
          ${{for( (id,v) <- parentVariables) yield s"parents[numRows * $id + row] = ${v.repr};"}.mkString("\n        ")}
        }
    }
    """
    }
    val text = texts.mkString("\n\n")
    val kernels = cl.createProgram(text).build().createKernels
    val parents = (BitSet.empty ++ rules.map(_._1.parent.coarse)).toJavaBitSet
    CLUnaryRuleUpdater(IndexedSeq(new RuleKernel(kernels, parents)))
  }

  private def coreUnaryRuleLoop(rulePartition: IndexedSeq[(UnaryRule[SymId[C, L]], Int)], accumulator: Map[Int, Variable])(implicit cl: CLContext) = {
    val sb = new StringBuilder()
    for ((_lc, rr) <- rulePartition.groupBy(_._1.child)) {
      val lc = _lc.gpu
      val child = s"child_$lc"
      sb ++= s"        float $child = children[numRows * $lc + row];\n"
      for ((r, id) <- rr) {
        sb ++= s"            ${accumulator(r.parent.gpu).repr} = ${semiring.mad(accumulator(r.parent.gpu).repr, child,  floatToString(structure.scores(id)))};\n"
      }
    }
    sb.result()
  }



  case class Variable(name: String, descr: String) {

    def declare = s"float $name = ${floatToString(semiring.zero)}; // $descr"

    def repr = name
  }

  private def floatToString(x: Float) = x match {
    case Float.PositiveInfinity => "INFINITY"
    case Float.NegativeInfinity => "-INFINITY"
    case x if x.isNaN => "NAN"
    case x => s"${x}f"
  }

  def writeParent = {
    """ typedef union { int old; float oldf; } intbox;
      |
      |inline void write_parent(volatile __global float* loc, float value) {
      |  intbox old;
      |  value = max(*loc, value);
      |  old.oldf = value;
      |
      |  //while((old.old = atomic_cmpxchg((volatile __global int*)loc, old.old, *(int*)&value)) !=  *(int*)&value) value = max(value, old.oldf);
      |  *loc = value;//old.oldf;
      |}
    """.stripMargin
  }

  //  def postClusterer:GrammarPartitioner[C, L] = new AgglomerativeGrammarPartitioner(numRestarts = 20, maxPartitionLabelSize = 20)//55)//new ILPGrammarClusterer(12, 55)
  def clusterer:GrammarPartitioner[C, L] = new KMeansGrammarPartitioner(k = 20)//55)//new ILPGrammarClusterer(12, 55)
  def unaryClusterer:GrammarPartitioner[C, L] = new AgglomerativeGrammarPartitioner(numRestarts = 100, maxPartitionLabelSize = 200)//55)//new ILPGrammarClusterer(12, 55)

}
 **/
