package puck.parser.gen

import epic.trees._
import virtualization.lms.common.{RangeOps, Base}
import trochee.kernels.KernelOps
import spire.implicits._
import trochee.basic.SpireOps

/**
 * 
 * @author dlwh
 */
trait InliningInsideKernels[L] extends UniformLoopInsideKernels[L] { self: Base with KernelOps with RangeOps with SpireOps =>

  protected def expprintf(string: String, args: Rep[_]*):Rep[Unit]


  protected def doInsideUnaryUpdates(top: Accumulator, bot: ParseCell, rulePartition: IndexedSeq[(UnaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    for( (parent, rr) <- rulePartition.groupBy(_._1.parent)) {
      for((r,id) <- rr) {
        val botScore = bot(r.child)
//        if(r.child == 43)
//          expprintf("Child! %f\n", botScore)
//        if(r.parent == grammar.root)
//          expprintf("ChildX! %f\n", botScore)
        top.mad(parent, botScore, rules.rules(id, gram))
      }


    }
  }

  protected def doInsideBinaryUpdates(out: Accumulator, left: ParseCell, right: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
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

trait UniformLoopInsideKernels[L] extends InsideKernels[L] { self: Base with KernelOps with RangeOps with SpireOps =>
  protected def doLeftInsideTermUpdates(out: Accumulator, leftTerm: ParseCell, right: ParseCell, rules: Rep[RuleCell], gram: Rep[Int], rulePartition: IndexedSeq[(BinaryRule[Int], Int)]): Rep[Unit] = {
    doInsideBinaryUpdates(out, leftTerm, right, rulePartition, rules, gram)
  }

  protected def doBothInsideTermUpdates(out: Accumulator, leftTerm: ParseCell, rightTerm: ParseCell, rules: Rep[RuleCell], gram: Rep[Int], rulePartition: IndexedSeq[(BinaryRule[Int], Int)]): Rep[Unit] = {
    doInsideBinaryUpdates(out, leftTerm, rightTerm, rulePartition, rules, gram)
  }

  protected def doRightInsideTermUpdates(out: Accumulator, left: ParseCell, rightTerm: ParseCell, rules: Rep[RuleCell], gram: Rep[Int], rulePartition: IndexedSeq[(BinaryRule[Int], Int)]): Rep[Unit] = {
    doInsideBinaryUpdates(out, left, rightTerm, rulePartition, rules, gram)
  }

  protected def doNTInsideRuleUpdates(out: Accumulator, left: ParseCell, right: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doInsideBinaryUpdates(out, left, right, rulePartition, rules, gram)
  }

  protected def doInsideUnaries(top: Accumulator, bot: ParseCell, rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doInsideUnaryUpdates(top, bot, grammar.unaryRules, rules, gram)
  }

  protected def doInsideTermUnaries(top: Accumulator, bot: ParseCell, rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doInsideUnaryUpdates(top, bot, grammar.unaryTermRules, rules, gram)
  }

  protected def doInsideBinaryUpdates(out: Accumulator, left: ParseCell, right: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit]
  protected def doInsideUnaryUpdates(top: Accumulator, bot: ParseCell, rulePartition: IndexedSeq[(UnaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit]
}
