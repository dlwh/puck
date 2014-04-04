package puck.parser

import epic.trees.annotations.{Xbarize, TreeAnnotator}
import epic.trees._
import java.io.File
import breeze.config.CommandLineParser
import java.util.zip.ZipFile
import com.nativelibs4java.opencl.{JavaCL, CLContext}
import java.util.Collections
import com.typesafe.scalalogging.slf4j.Logging
import breeze.optimize.BatchDiffFunction
import puck.{BatchFunctionAnnotatorService, AnnotatorService}
import scala.io.{Source, Codec}
import java.util.concurrent.atomic.{AtomicLong, AtomicInteger}
import scala.concurrent.ExecutionContext

/**
 * TODO
 *
 * @author dlwh
 */
object RunParser extends Logging {
  import ExecutionContext.Implicits.global

  case class Params(device: String = "nvidia",
                    profile: Boolean = false,
                    numToParse: Int = 1000,
                    grammar: File = new File("grammar.grz"),
                    maxParseLength: Int = 10000,
                    mem: String = "3g")

  def main(args: Array[String]) {
    val (config, files) = CommandLineParser.parseArguments(args)
    val params:Params = config.readIn[Params]("")
    import params._

    implicit val context: CLContext = {
      val (good, bad) = JavaCL.listPlatforms().flatMap(_.listAllDevices(true)).partition(d => params.device.r.findFirstIn(d.toString.toLowerCase()).nonEmpty)
      if(good.isEmpty) {
        JavaCL.createContext(Collections.emptyMap(), bad.sortBy(d => d.toString.toLowerCase.contains("geforce")).last)
      } else {
        JavaCL.createContext(Collections.emptyMap(), good.head)
      }

    }
    logger.info(s"Using context $context")

    val parserData = CLParserData.readSequence[AnnotatedLabel, AnnotatedLabel, String](new ZipFile(params.grammar))

    val parser = new CLParser(parserData, CLParser.parseMemString(mem), profile = profile)

    val service: BatchFunctionAnnotatorService[IndexedSeq[String], BinarizedTree[AnnotatedLabel]] = AnnotatorService.fromBatchFunction(parser.parse(_))

    logger.info("Up and running")

    val fileIter = if(files.nonEmpty) files.iterator.map(Source.fromFile(_)(Codec.UTF8)) else Iterator(Source.fromInputStream(System.in))

    val consumedIndex = new AtomicLong()
    var producedIndex = 0L
    val timeIn = System.currentTimeMillis()

    for(f <- fileIter; line <- f.getLines()) {
      val words = line.trim.split(" ")
      val i = producedIndex
      producedIndex += 1
      service(words).foreach { guess =>
        val tree: Tree[String] = AnnotatedLabelChainReplacer.replaceUnaries(guess).map(_.label)
        val guessTree = Trees.debinarize(Trees.deannotate(tree))
        val rendered = guessTree.render(words, newline = false)
        consumedIndex.synchronized {
          while (consumedIndex.get() != i) {
            consumedIndex.wait()
          }
          println(rendered)
          consumedIndex.incrementAndGet()
          consumedIndex.notifyAll()
        }
      }
    }
    service.flush()

    consumedIndex.synchronized {
      while (consumedIndex.get() != producedIndex) {
        consumedIndex.wait()
      }
    }

    val timeOut = System.currentTimeMillis()
    val wallTime = (timeOut - timeIn) / 1E3
    logger.info(f"Parsing took ${wallTime}s seconds. ${producedIndex/wallTime}%.3f sentences per second.")



  }

}
