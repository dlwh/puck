package puck.util

import scala.collection.generic.CanBuildFrom

/**
 *
 *
 * @author dlwh
 */
object CollectionImplicits {
  implicit class RichCollection[CC, T](c: CC)(implicit tv: CC<:<Iterable[T]) {
    def scanLeft[B, CCB](x: B)(f: (B, T)=>B)(implicit cbf: CanBuildFrom[CC, B, CCB]):CCB = {
      val bldr = cbf(c)
      var b = x
      bldr += b
      for(t <- c) {
        b = f(b, t)
        bldr += b
      }
      bldr.result()
    }
  }

}
