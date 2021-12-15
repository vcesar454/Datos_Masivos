//LSVM algorithm 
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{StringIndexer, IndexToString}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.MulticlassMetrics

def SVM(N_iterations : Int): Unit = {
    
    var i = 1;

    val sparkSession1 = SparkSession.builder.getOrCreate()
    //Import the dataset
    val bankDF = sparkSession1.read.format("csv").option("inferSchema", "true").option("delimiter", ";").option("header", "true").load("dataset/bank.csv")


    val vector_assembler = new VectorAssembler().setInputCols(Array("balance", "day", "duration", "campaign", "pdays", "previous")).setOutputCol("features")
    var label_Indexer = new StringIndexer().setInputCol("y").setOutputCol("label")

    //Spliting the dataset
    val Array(trainingData, testingData) = bankDF.randomSplit(Array(0.7, 0.3))

    //mapItera is used to record the iteration number and accuracy
    val mapItera = scala.collection.mutable.Map(0 -> 0.0)

    while(i <= N_iterations){

        val t1 = System.nanoTime

        //Model
        val LinearSVMC = new LinearSVC().setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setMaxIter(10).setRegParam(0.1)
        val pipeline1 = new Pipeline().setStages(Array(vector_assembler, label_Indexer, LinearSVMC))
        val model = pipeline1.fit(trainingData)
        

        //Getting the predictions
        val results = model.transform(testingData)
        val predictions = results.select("prediction", "label")

        //Getting accuracy
        val metrics = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
        val accuracy = metrics.evaluate(predictions)
        
        //Printing Results!
        println(s"Model accuracy = $accuracy")

        mapItera += (i -> accuracy)

        val duration = (System.nanoTime - t1) / 1e9d
        print(s"Execution Time: $duration")
        i = i + 1
    }

    println("Printing the map for GBT Classifier:\n")
    mapItera.keys.foreach{ e => 
        print("Iteration = " + e + " ")
        println("Accuracy = " + mapItera(e)) 
    }
}

SVM(30)


//^^^^^^^^^^^^^^^^^^^^^^^DECISION TREE CLASSIFIER^^^^^^^^^^^^^^^^^^^^^^
//Decision Tree
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

def DecisionTreeA(N_iterations : Int): Unit = {

    var i = 1;
    //Creating and reading the dataset
    val sparkSession1 = SparkSession.builder.getOrCreate()
    val bankDF = sparkSession1.read.format("csv").option("inferSchema", "true").option("delimiter", ";").option("header", "true").load("dataset/bank.csv")

    //Assembeling the features column
    val feature_assembler = new VectorAssembler().setInputCols(Array("balance", "day", "duration", "campaign", "pdays", "previous")).setOutputCol("features")
    val features = feature_assembler.transform(bankDF)

    //Indexing the output label
    val labelsIndexed = new StringIndexer().setInputCol("y").setOutputCol("y_label")
    val dataIndexed = labelsIndexed.fit(features).transform(features)

    //Indexing all y labels and features
    val labelIndexer = new StringIndexer().setInputCol("y_label").setOutputCol("indexed_y_label").fit(dataIndexed)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexed_Features").setMaxCategories(4)

    //Spliting the dataset into two parts
    val Array(trainingData, testingData) = dataIndexed.randomSplit(Array(0.7, 0.3))

    //mapItera is used to record the iteration number and accuracy
    val mapItera = scala.collection.mutable.Map(0 -> 0.0)

    while(i <= N_iterations){
        //Measuring Execution time per run
        val t1 = System.nanoTime
        
        //Creating, the Classifier and indicationg the labels column and features column; reverting the indexed labels back to strings; lastly setting
        //up the Pipeline
        val dt = new DecisionTreeClassifier().setLabelCol("indexed_y_label").setFeaturesCol("indexed_Features")
        val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
        val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

        //Model
        val model = pipeline.fit(trainingData)

        //Predincting Results using the testing data
        val predictions = model.transform(testingData)

        //Evaluating the accuracy of the model
        val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexed_y_label").setPredictionCol("prediction").setMetricName("accuracy")

        //Getting the accuracy and storing it as is
        val accuracy = evaluator.evaluate(predictions)
        
        println(s"Accuracy = $accuracy" + "\n")
        mapItera += (i -> accuracy)
        val duration = (System.nanoTime - t1) / 1e9d
        print(s"Execution Time: $duration")
        i = i + 1
    }
    println("Printing the map for GBT Classifier:\n")
    mapItera.keys.foreach{ e => 
        print("Iteration = " + e + " ")
        println("Accuracy = " + mapItera(e)) 
    }
}

DecisionTreeA(30)

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

//^^^^^^^^^^^^^^^^^^^^^^^LOGISTIC REGRESSION CLASSIFIER^^^^^^^^^^^^^^^^^^^^^^
//Logistic Regression
def LogisticRegressionA(N_iterations : Int): Unit = {

    var i = 1;
    //Creating and reading the dataset
    val sparkSession2 = SparkSession.builder.getOrCreate()
    val bankDF = sparkSession2.read.format("csv").option("inferSchema", "true").option("delimiter", ";").option("header", "true").load("dataset/bank.csv")

    //Setting the "y" column as label
    val logregdata = bankDF.select(bankDF("y").as("ylabel"), $"balance", $"day", $"duration", $"campaign", $"pdays", $"previous")

    //Indexing labels to numeric values
    val labelsIndexed = new StringIndexer().setInputCol("ylabel").setOutputCol("label")
    val data = labelsIndexed.fit(logregdata).transform(logregdata)

    //Vector assembler for features
    val assembler = (new VectorAssembler().setInputCols(Array("balance", "day", "duration", "campaign", "pdays", "previous")).setOutputCol("features"))
    
    //Spliting the dataset into two parts
    val Array(trainingData, testingData) = data.randomSplit(Array(0.7, 0.3))

    //mapItera is used to record the iteration number and accuracy
    val mapItera = scala.collection.mutable.Map(0 -> 0.0)

    while(i <= N_iterations){
        //Measuring Execution time per run
        val t1 = System.nanoTime
        
        //Creating and training the model
        var lr = new LogisticRegression()
        var pipeline = new Pipeline().setStages(Array(assembler, lr))
        var model = pipeline.fit(trainingData)

        //Testing the model with the testing dataset
        val results = model.transform(testingData)

        //Getting predictions
        val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
        val metrics = new MulticlassMetrics(predictionAndLabels)

        //Getting the accuracy and storing it as is
        val accuracy = metrics.accuracy
        
        println(s"Accuracy = $accuracy" + "\n")
        mapItera += (i -> accuracy)
        val duration = (System.nanoTime - t1) / 1e9d
        print(s"Execution Time: $duration"+ "\n")
        i = i + 1
    }
    println("Printing the map for GBT Classifier:\n")
    mapItera.keys.foreach{ e => 
        print("Iteration = " + e + " ")
        println("Accuracy = " + mapItera(e)) 
    }
}
LogisticRegressionA(30)

//^^^^^^^^^^^^^^^^^^^^^^^MULTILAYER PERCEPTRON CLASSIFIER^^^^^^^^^^^^^^^^^^^^^^
//Multilayer Perceptron Classifier
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

def MultilayerPerceptronA(N_iterations : Int): Unit = {

    var i = 1;
    
    //Creating and reading the dataset
    val sparkSession3 = SparkSession.builder.getOrCreate()
    val bankDF = sparkSession3.read.format("csv").option("inferSchema", "true").option("delimiter", ";").option("header", "true").load("dataset/bank.csv")

    val assembler = new VectorAssembler().setInputCols(Array("balance", "day", "duration", "campaign", "pdays", "previous")).setOutputCol("features")
    val features = assembler.transform(bankDF)

    val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label")
    val dataIndexed = labelIndexer.fit(features).transform(features)

    val Array(trainingSet, testingSet) = dataIndexed.randomSplit(Array(0.7, 0.3))

    val layers = Array[Int](6,4,4,2)

    //mapItera is used to record the iteration number and accuracy
    val mapItera = scala.collection.mutable.Map(0 -> 0.0)
       
    while(i <= N_iterations){
        //Measuring Execution time per run
        val t1 = System.nanoTime

        val trainer = new MultilayerPerceptronClassifier()
        val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

        val model = trainer.fit(trainingSet)  

        val result = model.transform(testingSet)      

        val predictionAndLabels = result.select("prediction", "label")

        val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
        
        val accuracy = evaluator.evaluate(predictionAndLabels)

        println(s"Accuracy = $accuracy" + "\n")
        mapItera += (i -> accuracy)
        val duration = (System.nanoTime - t1) / 1e9d
        print(s"Execution Time: $duration"+ "\n")
        i = i + 1
    }

    println("Printing the map for GBT Classifier:\n")
    mapItera.keys.foreach{ e => 
        print("Iteration = " + e + " ")
        println("Accuracy = " + mapItera(e) + "\n") 
    }

}