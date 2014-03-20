package puck

import scala.concurrent._
import scala.concurrent.duration.Duration
import java.util.concurrent.{ArrayBlockingQueue, TimeUnit}
import scala.collection.mutable.ArrayBuffer

/**
 * TODO
 *
 * @author dlwh
 **/
trait AnnotatorService[-In, +Out] extends (In=>Future[Out]) {
  def flush()
  def flushInterval:Duration
}

class FunctionAnnotatorService[-In, +Out](f: In=>Out)(implicit context: ExecutionContext) extends AnnotatorService[In, Out] {
  override def flushInterval: Duration = Duration.Zero

  override def flush() {}

  override def apply(v1: In): Future[Out] = {
    scala.concurrent.future(f(v1))
  }
}


class BatchFunctionAnnotatorService[In, Out](f: IndexedSeq[In]=>IndexedSeq[Out], val flushInterval: Duration = Duration(1, TimeUnit.SECONDS))(implicit context: ExecutionContext) extends AnnotatorService[In, Out] { service =>
  private var queue = new ArrayBuffer[(In, Promise[Out])]()

  override def flush() {
    monitorThread.interrupt()
  }

  private val monitorThread = new Thread() {
    override def run(): Unit = {
      try {
        Thread.sleep(flushInterval.toMillis)
      } catch {
        case ex: InterruptedException =>
      }

      val theQueue = service.synchronized {
        val q = queue
        queue = new ArrayBuffer()
        q
      }

      if (theQueue.nonEmpty) {
        for( (out, promise) <- f(theQueue.map(_._1)) zip theQueue.map(_._2)) {
          promise.success(out)
        }
      }
      run()
    }
    setDaemon(true)
    start()
  }

  override def apply(v1: In): Future[Out] = synchronized {
    val promise = Promise[Out]()
    queue +=  (v1 -> promise)
    promise.future
  }
}
object AnnotatorService {

  def fromFunction[In, Out](f: In=>Out)(implicit context: ExecutionContext) = new FunctionAnnotatorService(f)

  def fromBatchFunction[In, Out](f: IndexedSeq[In]=>IndexedSeq[Out],
                                 flushInterval: Duration = Duration(1, TimeUnit.SECONDS))(implicit context: ExecutionContext) = {
    new BatchFunctionAnnotatorService(f, flushInterval)
  }

}
