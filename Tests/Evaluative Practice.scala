import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline



//Creating session
val session = SparkSession.builder().getOrCreate

//Reading the iris dataset provided by jcromerohdz as a text file
val iris_data = session.read.option("header","true").option("inferSchema", true).csv("iris.csv")

//Name of columns
iris_data.columns

//getting to know the dataset
iris_data.printSchema()

//Printing the first five rows of data
iris_data.head(5)

//Print some features of the dataset
iris_data.describe().show()


//Setting the input columns to a single one as pFeatures
val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width")).setOutputCol("pFeatures")
val pFeatures = assembler.transform(iris_data)
pFeatures.show(5)

//Indexing the labels (species)
val SpeciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("indexedSpecies").fit(pFeatures)
println(s"labels: ${SpeciesIndexer.labels.mkString("[", ", ", "]")}")

//Indexing the features
val featuresIndexer = new VectorIndexer().setInputCol("pFeatures").setOutputCol("indexedFeatures").setMaxCategories(4).fit(pFeatures)

//Split the dataset into two parts, one for training set and another for testing.
val splits = pFeatures.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)

val layers = Array[Int](4,5,4,3)

//Training the trainer
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedSpecies").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(System.currentTimeMillis).setMaxIter(200)

val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(SpeciesIndexer.labels)

val pipeline = new Pipeline().setStages(Array(SpeciesIndexer, featuresIndexer, trainer, labelConverter))

val model = pipeline.fit(train)

val predictions = model.transform(test)
predictions.show(10)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedSpecies").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)
println(accuracy*100)