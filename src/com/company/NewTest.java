package com.company;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import scala.Tuple2;

public class NewTest {

    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example").setMaster("local[2]").set("spark.executor.memory","1g");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        int rank = 10;
        int numIterations = 10;


        SQLContext sqlC = new SQLContext(jsc);
        Dataset<Row> myRow = sqlC.read().format("com.crealytics.spark.excel")
                .option("sheetName", "PySheet1") // Required
                .option("useHeader", "true") // Required
                .option("maxRowsInMemory", 10)
                .load("test.xlsx");

        myRow.printSchema();


        Dataset<Rating> newRatings = myRow.map((MapFunction<Row,Rating>) s -> {
            return new Rating(s.getString(0).hashCode(),
                    s.getString(1).hashCode(),
                    Double.parseDouble(s.getString(2)));
        }, Encoders.bean(Rating.class));

        MatrixFactorizationModel newModel = ALS.train(newRatings.rdd(), rank, numIterations, 0.01);
        for(int i=0;i<20;i++) {
            System.out.println("print predict" + String.valueOf(i) + ":" + newModel.recommendProducts("Stu_00afbbe152564114428f42ea10109783".hashCode(), 20)[i]);
        }

//        List<String> myStrings = namesDS.takeAsList(20000).subList(1,19999);
        System.out.println("COUNT "+newRatings.count());

        System.out.println("row size"+String.valueOf(myRow.count()));
//        MatrixFactorizationModel sameModel = MatrixFactorizationModel.load(jsc.sc(),
//                "target/tmp/myCollaborativeFilter");

    }


}
