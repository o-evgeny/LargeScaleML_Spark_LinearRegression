package org.apache.spark.ml.made


import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import breeze.linalg.{DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.{VectorAssembler}
import scala.util.Random
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions.length
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {



  val precision = 0.05


  "LRPipeline" should "precision predict" in {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .set_LR(0.75)
      .set_nIters(100)

    val featureAssembler = new VectorAssembler()
      .setInputCols(Array("f1", "f2", "f3","f4","f5"))
      .setOutputCol("features")
    val addr=getClass.getResource("/data.csv").getPath
    var df = spark.read.format("csv").option("header", "true").load(addr)
    df = df.withColumn("f1", df("f1").cast(DoubleType))
    df = df.withColumn("f2", df("f2").cast(DoubleType))
    df = df.withColumn("f3", df("f3").cast(DoubleType))
    df = df.withColumn("f4", df("f4").cast(DoubleType))
    df = df.withColumn("f5", df("f5").cast(DoubleType))
    df = df.withColumn("label", df("label").cast(DoubleType))
    val preparedData = featureAssembler.transform(df).select("features", "label")
    val labels=preparedData.select("label").collect.map(_.getDouble(0))


    val model = lr.fit(preparedData)

    //сравним точность предсказаний для первых 5 строк датасета с точностью +- 0.05

     val pred0 = model.predict(Vectors.dense(0.6668765861978959,0.096078812567846,0.9462906748951908,0.11622215377011025,0.6995605283007603).toDense)
      println(s"True y: $labels(0).toDouble",s" Predicted y: $pred0")
     pred0 should be (labels(0)+-precision)

    val pred1 = model.predict(Vectors.dense(0.7109794374018417,0.08439647759621327,0.6094024066002077,0.13407813730941542,0.8669872196051772).toDense)
    pred1 should be (labels(1)+-precision)

    val pred2 = model.predict(Vectors.dense(0.7969186600388952,0.06662137436151871,0.22272181814805447,0.7605637711429065,0.9576379730215111).toDense)
    pred2 should be (labels(2)+-precision)

    val pred3 = model.predict(Vectors.dense(0.5101335754629772,0.10029108404261367,0.7063855195571435,0.47813174203071207,0.3871379393994848).toDense)
    pred3 should be (labels(3)+-precision)

    val pred4 = model.predict(Vectors.dense(0.9984719755109714,0.006369654419102422,0.8321138194092732,0.5224842920218686,0.8437985103285152).toDense)
    pred4 should be (labels(4)+-precision)



  }





}
object LinearRegressionTest extends WithSpark {


}