package puck.newparser.generator

import epic.trees.{UnaryRule, BinaryRule}
import trochee.kernels._
import trochee.basic._
import scala.virtualization.lms.common.{IfThenElse, ArrayOps, NumericOps, Base}

/**
 * TODO
 *
 * @author dlwh
 **/
trait RuleMultiply[L] extends Base with KernelOps with ExtraBase with AccumulatorOps with IfThenElse {
  def binaryRuleKernel(rulePartition: IndexedSeq[(BinaryRule[Int], Int)], name: String) = kernel(name, { (parent: Rep[Array[Real] with Global],
                                                                                left: Rep[Array[Real] with Global],
                                                                                right: Rep[Array[Real] with Global],
                                                                                rules: Rep[Array[Real] with Constant],
                                                                                parentRows: Rep[Int], // number of tree indices
                                                                                numToDo: Rep[Int] // number of tree indices to do
                                                                                 ) =>
    val row = globalId(0)
    if(row < numToDo) {
      val out = accumulator(rulePartition.map(_._1.parent).toSet)
      for( (lc, rr) <- rulePartition.groupBy(_._1.left)) {
        val leftScore:Rep[Real] = left(parentRows * lc + row)
        for((rc,rrr) <- rr.groupBy(_._1.right)) {
          val rightScore = right(parentRows * rc + row)
          val joint:Rep[Real] = leftScore * rightScore
          for((r,id) <- rrr) {

            //if(joint !== zero) printf(unit("%s %d %d %d %d %f %f %f %f\n"), unit(name), row, r.parent, lc, rc, leftScore,rightScore,rules(id), joint * rules(id))
            //if(r.parent === 42) printf(unit("!!! %s %d %d %d %d %f %f %f %f\n"), unit(name), row, r.parent, lc, rc, leftScore,rightScore,rules(id), joint * rules(id))
            out.mad(r.parent, joint, rules(id))
          }
        }
      }
      out.foreachUsed{ (id:Int, value: Rep[Real]) =>
        val p = parent(parentRows * id + row)
        parent(parentRows * id + row) = p + value
      }
    }

  })

  def unaryRuleKernel(rulePartition: IndexedSeq[(UnaryRule[Int], Int)], name: String) = kernel(name, { (parent: Rep[Array[Real] with Global],
                                                                                 child: Rep[Array[Real] with Global],
                                                                                 rules: Rep[Array[Real] with Constant],
                                                                                 rows: Rep[Int], // number of tree indices in the array
                                                                                 botRows: Rep[Int], // number of tree indices in the array
                                                                                 numToDo: Rep[Int] // number of tree indices
                                                                                  ) =>
    val row = globalId(0)
    if(row < numToDo) {
      val out = accumulator(rulePartition.map(_._1.parent).toSet)
      for( (lc, rrr) <- rulePartition.groupBy(_._1.child)) {
        val childScore:Rep[Real] = child(botRows * lc + row)
        for((r,id) <- rrr) {
       //   if(childScore !== zero) printf(unit("%s %d %d %d %f %f %f\n"), unit(name), row, r.parent, lc, childScore, rules(id), childScore * rules(id))
          out.mad(r.parent, childScore, rules(id))
        }
      }
      out.foreachUsed{ (id:Int, value: Rep[Real]) =>
        val p = parent(rows * id + row)
        parent(rows * id + row) = p + value
      }
    }

  })

}
