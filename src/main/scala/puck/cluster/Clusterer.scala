package puck.cluster

import breeze.math.{MutableInnerProductSpace, MutablizingAdaptor, InnerProductSpace}
import breeze.math.MutablizingAdaptor.InnerProductSpaceAdaptor
import breeze.stats.distributions.{Multinomial, RandBasis, Rand}
import epic.util.{Optional, NotProvided}
import scala.collection.mutable.ArrayBuffer
import breeze.linalg._
import breeze.util.Implicits._
import breeze.util._
import breeze.numerics._


/**
 *
 *
 * @author dlwh
 */
trait Clusterer[T] {

  def cluster(points: IndexedSeq[T]):IndexedSeq[IndexedSeq[T]]

}


class KMeans[T](k: Int, tolerance: Double = 1E-4, distance: Optional[(T,T)=>Double] = NotProvided)(implicit random: RandBasis = Rand, innerProductSpace: InnerProductSpace[T, Double]) extends Clusterer[T] {
  private val ensured = MutablizingAdaptor.ensureMutable(innerProductSpace)
  import ensured.{wrap, unwrap}
  import ensured.Wrapper
  import ensured.mutaVspace._
  private val metric = distance.getOrElse( (a:T,b:T) => math.pow(norm(wrap(a) - wrap(b)), 2))

  case class State(means: IndexedSeq[T], clusters: IndexedSeq[IndexedSeq[T]], error: Double, previousError: Double=Double.PositiveInfinity) {
    def converged = closeTo(error,previousError,tolerance)
  }


  def cluster(points: IndexedSeq[T]): IndexedSeq[IndexedSeq[T]] = {
    val lastState = iterates(points).last
    lastState.clusters
  }

  def iterates(points: IndexedSeq[T]) = {
    if(points.length <= k) Iterator(State(points, points.map(IndexedSeq(_)), 0.0))
    else {
      Iterator.iterate(initialState(points)){ current =>
        val distances = points.par.map { x =>
          val distances = current.means.map{m => metric(x,m)}
          val minPoint = distances.argmin // which cluster minimizes euclidean distance
          (x, minPoint,distances(minPoint))
        }

        val newClusters = distances.groupBy(_._2).seq.values.map{_.map(_._1)}
        val newMeans = newClusters.par.map(clust => clust.foldLeft(zeros(wrap(clust.head)))(_ += wrap(_)) /= clust.size.toDouble)
        val error = distances.map(tuple => math.pow(tuple._3, 2)).sum

        State(newMeans.map(unwrap).seq.toIndexedSeq, newClusters.map(_.toIndexedSeq).toIndexedSeq, error, current.error)
      }.takeUpToWhere(state => state.converged)
    }


  }

  private def initialState(points: IndexedSeq[T]):State = {
    val probs = Array.fill(points.length)(1.0)
    val means = new ArrayBuffer[T]()

    while(means.length < k) {
      val newMean = points(Multinomial(new DenseVector(probs)).sample())
      means += newMean
      for(i <- (0 until probs.length).par) {
        probs(i) = math.min(probs(i), metric(newMean, points(i)))
      }
    }

    new State(means, null, probs.sum)
  }
}
