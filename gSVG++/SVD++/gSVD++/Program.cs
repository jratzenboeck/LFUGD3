using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyMediaLite;
using MyMediaLite.Data;
using MyMediaLite.IO;
//using MyMediaLite.ItemRecommendation;
using MyMediaLite.RatingPrediction;
using MyMediaLite.Eval;

namespace gSVD__
{
    class Program
    {
        static void Main(string[] args)
        {
            String path_training_data = "user_ratedmovies.dat";
            String path_user_data = "users.dat";
            String path_atr_data = "movie_genres.dat";
            String path_pred_data = "predict.dat";

            var user_mapping = new Mapping();
            var item_mapping = new Mapping();

            //var user_mapping = new EntityMapping();
            //var item_mapping = new EntityMapping();
            //String path_training_data = "1.rec.train";
            //String path_test_data = "1.rec.test";
            //String path_training_data = "u1.base";
            //String path_test_data = "u1.test";
            //var training_data = ItemData.Read(path_training_data);
            //var test_data = ItemData.Read(path_test_data);

            var training_data = RatingData.Read(path_training_data, user_mapping, item_mapping);
            var relevant_items = item_mapping.InternalIDs;
            var atr_data = AttributeData.Read(path_atr_data, item_mapping);
            var pred_data = RatingData.Read(path_pred_data, user_mapping, item_mapping);
            //var user_data = ItemData.Read(path_user_data, user_mapping);
            Console.WriteLine("I'm here");
            //Console.WriteLine(String.Join(";", user_data.AllUsers));
            //var test_data = RatingData.Read(path_test_data);

            //var recommender = new ItemKNN();
            //recommender.Feedback = training_data;
            //recommender.Train();

            //var recommender = new GSVDPlusPlus();
            //var recommender = new MatrixFactorization();
            //recommender.Ratings = training_data;
            // recommender.MaxRating = 5;
            // recommender.MinRating = 1;
            //recommender.ItemAttributes = atr_data;
            //recommender.NumFactors = 5;
            //recommender.NumIter = 10;

            //recommender.Train();


            var recommender = new SVDPlusPlus();
            recommender.Ratings = training_data;
            //recommender.AdditionalFeedback = user_data;
            Console.WriteLine("I'm here2");
            Console.WriteLine(recommender.ToString());
            //recommender.ItemAttributes = atr_data;
            recommender.NumFactors = 10;

            Console.WriteLine("I'm here3");
            Console.WriteLine(recommender.ToString());
            recommender.Train();
            Console.WriteLine("I'm here4");


            //var results = recommender.DoIterativeCrossValidation(recommdender, 10, 1, 1, true);
            var results = recommender.DoCrossValidation(10, false, true);
            ////var results = recommender.Evaluate(test_data, training_data);
            foreach (var key in results.Keys)
                Console.WriteLine("{0}={1}", key, results[key]);
            Console.WriteLine(results);
            //recommender.Predict();
            recommender.WritePredictions(pred_data, "test45.dat", user_mapping, item_mapping,  "{0}|{1}|{2}");

            Console.WriteLine(recommender.Predict(1, 1));
            Console.ReadLine();

        }
    }
}
