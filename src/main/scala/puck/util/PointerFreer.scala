package puck.util

import java.util.concurrent.{ThreadFactory, LinkedBlockingDeque, TimeUnit, ThreadPoolExecutor}
import com.nativelibs4java.opencl.CLEvent

/**
 * TODO
 *
 * @author dlwh
 **/
object PointerFreer {
  // daemon threads so this exits on done
  val queue = new ThreadPoolExecutor(1, 3, 1, TimeUnit.SECONDS, new LinkedBlockingDeque[Runnable](), new ThreadFactory() {
    def newThread(r: Runnable): Thread = {
      val t = new Thread(r)
      t.setPriority(Thread.MIN_PRIORITY)
      t.setDaemon(true)
      t
    }
  })

  def enqueue(ptr: =>Unit, events: CLEvent*) = {
    queue.execute(new Runnable() {
      def run(): Unit = {
        CLEvent.waitFor(events:_*)
        ptr
      }
    })
  }


}
