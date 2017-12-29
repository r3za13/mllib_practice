package com.company;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkConf.*;
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
import scala.reflect.api.TypeTags;

import java.util.List;

public class Main {

    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example").setMaster("local[2]").set("spark.executor.memory","1g");
        JavaSparkContext jsc = new JavaSparkContext(conf);

// Load and parse the data
        String path = "Readme.me";
        JavaRDD<String> data = jsc.textFile(path);
        JavaRDD<Rating> ratings = data.map(s -> {
            String[] sarray = s.split(",");
            return new Rating(Integer.parseInt(sarray[0]),
                    Integer.parseInt(sarray[1]),
                    Double.parseDouble(sarray[2]));
        });

// Build the recommendation model using ALS
        int rank = 10;
        int numIterations = 10;
        MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);

// Evaluate the model on rating data
        JavaRDD<Tuple2<Object, Object>> userProducts =
                ratings.map(r -> new Tuple2<>(r.user(), r.product()));
        JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD()
                        .map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating()))
        );
        JavaRDD<Tuple2<Double, Double>> ratesAndPreds = JavaPairRDD.fromJavaRDD(
                ratings.map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating())))
                .join(predictions).values();
        double MSE = ratesAndPreds.mapToDouble(pair -> {
            double err = pair._1() - pair._2();
            return err * err;
        }).mean();
        System.out.println("Mean Squared Error = " + MSE);

// Save and load model
        for(int i=0;i<4;i++)
        {
            System.out.println("print predict"+String.valueOf(i)+":"+model.recommendProducts(1,4)[i]);
        }
//        System.out.println("print predict "+model.recommendProducts(1,4)[2].product()+","+model.recommendProducts(1,4)[3].product());
//        model.save(jsc.sc(), "target/tmp/myCollaborativeFilter");

        SQLContext sqlC = new SQLContext(jsc);
        Dataset<Row> myRow = sqlC.read().format("com.crealytics.spark.excel")
                .option("sheetName", "Sheet1") // Required
                .option("useHeader", "true") // Required
                .option("maxRowsInMemory", 10)
                .load("dataset.xlsx");

        myRow.printSchema();
//        JavaRDD<Rating> myRatings = myRow.flatMap(myr -> myr.)
//        Dataset<String> namesDS = myRow.map(
//
//                (MapFunction<Row, String>) row -> "Name: " + row.getString(2),
//                Encoders.STRING());
//        (MapFunction<Row, Rating>) row -> new Rating(Integer.parseInt(row.getString(2)),Integer.parseInt(row.getString(3)),row.getInt(12)));
//        Dataset<Rating> newRatings = myRow.map((MapFunction<Row,Rating>)thisRow -> new Rating(1,2,2) ,Encoders.product(Encoders.kryo(Rating.class)));


        Dataset<Rating> newRatings = myRow.map((MapFunction<Row,Rating>) s -> {
            return new Rating(s.getString(2).hashCode(),
                    s.getString(3).hashCode(),
                    Double.parseDouble(s.getString(12)));
        },Encoders.bean(Rating.class));

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
