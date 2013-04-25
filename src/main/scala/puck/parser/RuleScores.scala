package puck.parser

import epic.parser._
import epic.trees.BinaryRule

case class RuleScores(scores: Array[Double])

object RuleScores {
  def zeros[L](grammar: BaseGrammar[L]) = {
    val binaries = new Array[Double](grammar.index.size)

    RuleScores(binaries)
  }

  def fromRefinedGrammar[L, W](grammar: SimpleRefinedGrammar[L, _, W]) = {
    val binaries = new Array[Double](grammar.index.size)
    for( i <- 0 until grammar.index.size) {
      binaries(i) = grammar.ruleScore(i, 0)
    }

    RuleScores(binaries)
  }
}
