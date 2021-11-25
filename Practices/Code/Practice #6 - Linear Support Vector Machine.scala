
//From Spark Documentation
import org.apache.spark.ml.classification.LinearSVC

val training = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
val lsvcModel = lsvc.fit(training)
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

//Testing LSVM on the Titanic Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.VectorAssembler

val spark = SparkSession.builder().getOrCreate()
val titanic_df = spark.read.format("csv").option("inferSchema",
"true").option("header", "true").load("test.csv")
titanic_df.show()


val data = titanic_df.drop("PassengerId","Name","Ticket","Cabin","Embarked","Sex","SibSp","
Parch")
data.show()

val feature = new VectorAssembler().setInputCols(Array("Pclass","Age","Fare","Sex_index")).setOutputCol("features")
val feature_vector= feature.transform(data)
feature_vector.select("Survived","Pclass","Age","Fare","Sex_index","features").show()

val Array(trainingData, testData) = feature_vector.randomSplit(Array(0.7, 0.3))

val lnsvc = new LinearSVC().setLabelCol("Survived").setFeaturesCol("features")
val lnsvc_model = lnsvc.fit(trainingData)

val lnsvc_prediction = lnsvc_model.transform(testData)
lnsvc_prediction.select("prediction", "Survived", "features").show()

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("Survived").setPredictionCol("prediction").setMetricName("accuracy")
val lnsvc_accuracy = evaluator.evaluate(lnsvc_prediction)

print("Accuracy of Support Vector Machine is = " + (lnsvc_accuracy))
print(" and Test Error of Support Vector Machine = " + (1.0 - lnsvc_accuracy))