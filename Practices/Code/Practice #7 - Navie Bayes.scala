import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

val model = new NaiveBayes().fit(trainingData)
val predictions = model.transform(testData)
predictions.show()

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test set accuracy = $accuracy")