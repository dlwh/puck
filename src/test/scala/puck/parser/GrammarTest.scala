package trochee.parser

import org.scalatest.FunSuite
import java.io.File

/**
 *
 * @author dlwh
 */
class GrammarTest extends FunSuite {
  test("Load from file") {
    Grammar.parseFile(new File("src/main/resources/trochee/parser/demo.grammar.txt"))
  }

}
