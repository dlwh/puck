package puck.parser

import epic.parser.projections.GrammarRefinements
import epic.parser.BaseGrammar
import epic.trees.{BinaryRule, UnaryRule}
import breeze.util.Index

/**
 *
 * @author dlwh
 */
case class RuleStructure[C, L](refinements: GrammarRefinements[C, L], grammar: BaseGrammar[L]) {

  def numSyms = grammar.labelIndex.size
  def numCoarseSyms = refinements.labels.coarseIndex.size

  val (symIndex,
  nontermIndex,
  termIndex,
  leftTermRules,
  rightTermRules,
  nontermRules,
  bothTermRules,
  unaryRules,
  unaryTermRules,
  identityTermUnaries,
  root: Int) = {
    val rules = grammar.index.zipWithIndex.toIndexedSeq
    // jesus
    val termIndex, nontermIndex,symIndex = Index[L]()
    val nonterms = rules.collect {
      case (BinaryRule(p,_,_),_) => p
      case (UnaryRule(p,c, _),_) if p != c => p
    }.toSet


    val rhsSyms = rules.flatMap {
      case (rule@BinaryRule(p,l,r),_) => Iterator(l,r)
      case (rule@UnaryRule(p,c,_),_) if p != c => Iterator(c)
      case (rule@UnaryRule(p,c,_),_) => Iterator.empty
    }.toSet

    val syms = nonterms ++ rhsSyms

    nonterms foreach {nontermIndex.index _}
    syms -- nonterms foreach {termIndex.index _}
    nontermIndex foreach {symIndex.index _}
    termIndex foreach {symIndex.index _}

    val root = grammar.root

    def doIndex(sym: L) = { val nt = nontermIndex(sym); if (nt >= 0) nt -> false else termIndex(sym) -> true }
    val (binaries, unaries) = rules.map {
      case (r,i) => (r.map(doIndex(_)), i)
    }.partition(_._1.isInstanceOf[BinaryRule[_]])

    val groupedByTerminess = binaries.asInstanceOf[IndexedSeq[(BinaryRule[(Int, Boolean)], Int)]].groupBy {case (r,i) => r.left._2 -> r.right._2}

    def patchRules(pair: (BinaryRule[(Int, Boolean)], Int)) = pair._1.map(_._1) -> pair._2
    val leftTermRules = groupedByTerminess.getOrElse(true -> false, IndexedSeq.empty).map(patchRules _)
    val rightTermRules = groupedByTerminess.getOrElse(false -> true, IndexedSeq.empty).map(patchRules _)
    val bothTermRules = groupedByTerminess.getOrElse(true -> true, IndexedSeq.empty).map(patchRules _)
    val nontermRules = groupedByTerminess.getOrElse(false -> false, IndexedSeq.empty).map(patchRules _)

    // exclude identity unaries for terminals, which have a terminal parent.
    val uByTerminess = unaries.asInstanceOf[IndexedSeq[(UnaryRule[(Int, Boolean)], Int)]].filter{case (r,_) => !r.parent._2} groupBy {case (r,i) => r.child._2}
    def patchU(pair: (UnaryRule[(Int, Boolean)], Int)) = pair._1.map(_._1) -> pair._2
    val tUnaries = uByTerminess.getOrElse(true, IndexedSeq.empty).map(patchU _)
    val ntUnaries = uByTerminess.getOrElse(false, IndexedSeq.empty).map(patchU _)
    val tIdentUnaries = unaries.collect { case pair@(UnaryRule(p, c, _), _) if p == c && p._2 => pair }

    (symIndex, nontermIndex, termIndex, leftTermRules,
      rightTermRules, nontermRules, bothTermRules, ntUnaries, tUnaries, tIdentUnaries, nontermIndex(root))
  }


  def numTerms = termIndex.size
  def numNonTerms = nontermIndex.size

  /** Maps an indexed terminal symbol back to the grammar's index*/
  val terminalMap = Array.tabulate(numTerms)(i => grammar.labelIndex(termIndex.get(i)))
  /** Maps an indexed nonterminal symbol back to the grammar's index*/
  val nonterminalMap = Array.tabulate(numNonTerms)(i => grammar.labelIndex(nontermIndex.get(i)))
  val reverseIndex = Array.fill[Int](grammar.labelIndex.size)(-1)
  for(i <- 0 until terminalMap.length) {
    reverseIndex(terminalMap(i)) = i
  }
  for(i <- 0 until nonterminalMap.length) {
    reverseIndex(nonterminalMap(i)) = i
  }
  def labelIndexToTerminal(label: Int) = reverseIndex(label)
  def labelIndexToNonterminal(label: Int) = reverseIndex(label)

  lazy val partitionsParent     : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(nontermRules, targetLabel = GrammarPartitioner.Parent).toIndexedSeq
  lazy val partitionsLeftChild  : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(nontermRules, targetLabel = GrammarPartitioner.LeftChild).toIndexedSeq
  lazy val partitionsRightChild : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(nontermRules, targetLabel = GrammarPartitioner.RightChild).toIndexedSeq

  lazy val partitionsLeftTermRules            : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(leftTermRules, targetLabel = GrammarPartitioner.Parent).toIndexedSeq
  lazy val partitionsLeftTermRules_LeftChild  : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(leftTermRules, targetLabel = GrammarPartitioner.LeftChild).toIndexedSeq
  lazy val partitionsLeftTermRules_RightChild : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(leftTermRules, targetLabel = GrammarPartitioner.RightChild).toIndexedSeq

  lazy val partitionsRightTermRules            : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(rightTermRules, targetLabel = GrammarPartitioner.Parent).toIndexedSeq
  lazy val partitionsRightTermRules_LeftChild  : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(rightTermRules, targetLabel = GrammarPartitioner.LeftChild).toIndexedSeq
  lazy val partitionsRightTermRules_RightChild : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(rightTermRules, targetLabel = GrammarPartitioner.RightChild).toIndexedSeq

  lazy val partitionsBothTermRules             : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(bothTermRules, targetLabel = GrammarPartitioner.Parent).toIndexedSeq
  lazy val partitionsBothTermRules_LeftChild   : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(bothTermRules, targetLabel = GrammarPartitioner.LeftChild).toIndexedSeq
  lazy val partitionsBothTermRules_RightChild  : IndexedSeq[IndexedSeq[(BinaryRule[Int], Int)]] = GrammarPartitioner.partition(bothTermRules, targetLabel = GrammarPartitioner.RightChild).toIndexedSeq

  def numRules = grammar.index.size

  /**
   * 1 if the corresponding unary rule is not a terminal->terminal unary rule.
   */
  val nonIdentityMask = {
    val arr = Array.fill(numRules)(1.0)
    for(t <- identityTermUnaries) {
      arr(t._2) = 0.0
    }
    arr
  }

  // pruning stuff
  val pruningMaskFieldSize = numCoarseSyms/64 + {if(numCoarseSyms % 64 != 0) 1 else 0}

  def pruningMaskForSyms(syms: Iterable[Int]) = {
    val coarsened = syms.map(refinements.labels.project(_)).toSet
    val mask = Array.fill(pruningMaskFieldSize)("")
    for( (field, coarses) <- coarsened.groupBy(_ / 64)) {
      mask(field) = coarses.map(c => "(1L << %d)".format(c-64*field)).mkString("(", "|", ")")
    }
    mask
  }

  def pruningCheckForSyms(syms: Iterable[Int], id: Int) = {
    val mask = pruningMaskForSyms(syms)
    val checks = mask.zipWithIndex.filter(_._1.nonEmpty).map{ case (mask, field) => "(((mask)).allowed[" + field + "] &(" + mask +"))"}
    checks.mkString("#define IS_ANY_IN_BLOCK_" +id +"(mask)  (", "||", ")")
  }
}

