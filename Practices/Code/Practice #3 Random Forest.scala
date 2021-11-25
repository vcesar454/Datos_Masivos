import org.apache.spark.ml.Pipeline
import org.apache.spark.classification.RandomForestClassificationModel
import org.apache.spark.classification.RandomForestClassifier
import org.apache.spark.MulticlassClassificationEvaluator
import org.apache.spark.feature.IndexToString
import org.apache.spark.feature.StringIndexer
import org.apache.spark.feature.VectorIndexer
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

val data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(data)

val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4)
    .fit(data)

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

val randomforest = new RandomForestClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("indexedFeatures")
    .setNumTrees(10)

val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
val model = pipeline.fit(trainingData)

val predictions = model.transform(testData)
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")