from pyspark.sql import SparkSession

def main():
    # Initialize a SparkSession
    spark = SparkSession.builder \
        .appName("MySimpleSparkApp") \
        .getOrCreate()

    print("SparkSession created successfully!")

    # Create a simple DataFrame
    data = [("Alice", 1),
            ("Bob", 2),
            ("Charlie", 3)]
    columns = ["name", "id"]
    df = spark.createDataFrame(data, columns)

    # Show the DataFrame in the logs
    df.show()

    # Stop the SparkSession
    spark.stop()
    print("Spark job finished successfully.")

if __name__ == '__main__':
    main()