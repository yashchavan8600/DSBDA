import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object TitanicAnalysis {

    def main(args: Array[String]): Unit = {

        // Create Spark Session
        val spark = SparkSession.builder()
            .appName("Titanic Analysis")
            .master("local[*]")
            .getOrCreate()

        // Load Titanic Dataset
        val df = spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv("titanic_data.csv")

        // Display first 5 rows
        df.show(5)

        // Print column names
        println("Columns in Dataset:")
        df.columns.foreach(println)

        // Print dataset schema
        df.printSchema()

        // Check missing values
        println("Missing values in Age column:")
        println(df.filter(col("Age").isNull).count())

        // Fill missing Age values with average age
        val avgAge = df.select(avg("Age")).first().getDouble(0)

        val newDf = df.na.fill(avgAge, Seq("Age"))

        // Show updated dataset
        newDf.show(5)

        // Calculate average Fare
        println("Average Fare:")
        newDf.select(avg("Fare")).show()

        // Group passengers by Passenger Class
        println("Passenger Count by Pclass:")
        newDf.groupBy("Pclass").count().show()

        // Stop Spark Session
        spark.stop()
    }
}