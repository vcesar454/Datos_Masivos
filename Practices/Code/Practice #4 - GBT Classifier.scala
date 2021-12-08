import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.Pipeline

val data = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt")

val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))
val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.numClasses = 2
boostingStrategy.treeStrategy.maxDepth = 5
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

val labelAndPreds = testData.map { point =>
val prediction = model.predict(point.features)
    (point.label, prediction)
}

val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println(s"Test Error = $testErr")
println(s"Learned classification GBT model:\n ${model.toDebugString}")