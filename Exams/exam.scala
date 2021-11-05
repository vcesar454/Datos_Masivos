import org.apache.spark.sql.SparkSession
val session = SparkSession.builder().getOrCreate

val df_netflix = session.read.option("header",
"true").option("inferSchema", true).csv("Netflix_2011_2016.csv")

df_netflix.columns

df_netflix.printSchema()

df_netflix.head(5)

df_netflix.describe().show

val df_netflix2 = df_netflix.withColumn("HV Ratio",
df_netflix("High")/df_netflix("Volume"))

df_netflix.select(mean("Open")).show()

df_netflix.select(mean("Close")).show()

df_netflix.select(max("Volume")).show()
df_netflix.select(min("Volume")).show()

val Day = df_netflix.where($"Close" < 600).count()

val Day = df_netflix.where($"High" > 500).count().toFloat

df_netflix.select(corr("High", "Volume")).show()

df_netflix.groupBy(($"Date")).max("High").show()

val df_netflix3 = df_netflix.groupBy(year($"Date"),
month($"Date")).mean("Close"). toDF("Year","Month","Mean")
df_netflix3.orderBy($"Year",$"Month").show()
