package puck.parser.gen

import epic.trees._
import virtualization.lms.common.{RangeOps, Base}
import trochee.kernels.KernelOps
import trochee.basic.SpireOps

/**
 * 
 * @author dlwh
trait InliningRuleCountKernels[L] extends RuleCountsKernels[L] { self: Base with KernelOps with RangeOps with SpireOps =>

  protected def doECountBinaryUpdates(ruleAcc: Accumulator,
                                      parentOutside: ParseCell,
                                      leftInside: ParseCell,
                                      rightInside: ParseCell,
                                      sentScore: Rep[Real],
                                      rules: Rep[RuleCell],
                                      gram: Rep[Int],
                                      partition: IndexedSeq[(BinaryRule[Int], Int)]):Rep[Unit] = {
    val leftCache = collection.mutable.Map[Int, Rep[Real]]()
    val rightCache = collection.mutable.Map[Int, Rep[Real]]()
    for ( (p, rs) <- partition.groupBy(_._1.parent)) {
      val pscore = outsideParent(p)
      for( (left, rrs) <- rs.groupBy(_._1.left)) {
        val lscore = leftCache.getOrElseUpdate(left, leftInside(left))
        val leftJoint = pscore * lscore
        for( (r,id) <- rrs) {
          val right = r.right
          val rscore = rightCache.getOrElseUpdate(right, rightInside(right))
          val rightJoint = pscore * rscore
          ruleAcc.mad(id, leftJoint * rightJoint, ruleScore)

        }
      }
    }
  }

  protected def doECountUnaryUpdates(ruleAcc: Accumulator,
                                      parentOutside: ParseCell,
                                      childInside: ParseCell,
                                      rules: Rep[RuleCell],
                                      gram: Rep[Int],
                                      partition: IndexedSeq[(BinaryRule[Int], Int)]):Rep[Unit]
  protected def doBothRuleCountsUpdates(leftOut: Accumulator, rightOut: Accumulator, outsideParent: ParseCell, leftInside: ParseCell, rightInside: ParseCell, rules: Rep[RuleCell], gram: Rep[Int], partition: IndexedSeq[(BinaryRule[Int], Int)]): Rep[Unit] = {

  }

  protected def doNTRuleCountsRuleRightUpdates(out: Accumulator, parent: ParseCell, leftInside: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    val leftCache = collection.mutable.Map[Int, Rep[Real]]()
    for ( (p, rs) <- rulePartition.groupBy(_._1.parent)) {
      val pscore = parent(p)
      for( (left, rrs) <- rs.groupBy(_._1.left)) {
        val lscore = leftCache.getOrElseUpdate(left, leftInside(left))
        val leftJoint = pscore * lscore
        for( (r,id) <- rrs) {
          val right = r.right
          val ruleScore = rules.rules(id, gram)
          out.mad(right, leftJoint, ruleScore)
        }
      }
    }
  }

  protected def doNTRuleCountsRuleLeftUpdates(out: Accumulator, parent: ParseCell, rightInside: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    val rightCache = collection.mutable.Map[Int, Rep[Real]]()
    for ( (p, rs) <- rulePartition.groupBy(_._1.parent)) {
      val pscore = parent(p)
      for( (right, rrs) <- rs.groupBy(_._1.right)) {
        val rscore = rightCache.getOrElseUpdate(right, rightInside(right))
        val rightJoint = pscore * rscore
        for( (r,id) <- rrs) {
          val left = r.left
          val ruleScore = rules.rules(id, gram)
          out.mad(left, rightJoint, ruleScore)
        }
      }
    }
  }

  protected def doRuleCountsUnaries(bot: Accumulator, top: ParseCell, rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doRuleCountsUnaryUpdates(bot, top, grammar.unaryRules, rules, gram)
  }

  protected def doRuleCountsTermUnaries(bot: Accumulator, top: ParseCell, rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doRuleCountsUnaryUpdates(bot, top, grammar.unaryTermRules, rules, gram)
  }

  protected def doRuleCountsUnaryUpdates(top: Accumulator, bot: ParseCell, rulePartition: IndexedSeq[(UnaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    for( (parent, rr) <- rulePartition.groupBy(_._1.parent)) {
      val parentScore = bot(parent)
      for((r,id) <- rr) {
        top.mad(r.child, parentScore, rules.rules(id, gram))
      }
    }
  }

  protected def doRuleCountsBinaryUpdates(out: Accumulator, left: ParseCell, right: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    for( (lc, rr) <- rulePartition.groupBy(_._1.left)) {
      val leftScore = left(lc)
      for((rc,rrr) <- rr.groupBy(_._1.right)) {
        val rightScore = right(rc)
        val joint = leftScore * rightScore
        for((r,id) <- rrr) {
         out.mad(r.parent, joint, rules.rules(id, gram))
        }
      }
    }

  }
}


 */
