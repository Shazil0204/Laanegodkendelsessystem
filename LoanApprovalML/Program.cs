using LoanApprovalML.DataModels;
using LoanApprovalML.Services;
using Microsoft.ML;

namespace LoanApprovalML
{
    class Program
    {
        static void Main(string[] args)
        {
            var trainer = new Trainer();
            trainer.Train("data.csv");

            Console.WriteLine("Do you want to test the saved model? (yes/no)");
            var answer = Console.ReadLine()?.ToLower();

            if (answer == "yes")
            {
                var mlContext = new MLContext();
                // Load the saved model
                ITransformer loadedModel = mlContext.Model.Load("model.zip", out var schema);

                // Create prediction engine
                var predEngine = mlContext.Model.CreatePredictionEngine<InputData, ModelOutput>(loadedModel);

                // Dummy test data
                var testSamples = new InputData[]
                {
                    new InputData { MonthlyIncome = 4000, LoanAmount = 10000, ReturnTime = 24, Age = 30, JobType = "Full-time" },
                    new InputData { MonthlyIncome = 2000, LoanAmount = 5000, ReturnTime = 36, Age = 22, JobType = "Student" },
                    new InputData { MonthlyIncome = 6000, LoanAmount = 20000, ReturnTime = 12, Age = 40, JobType = "Part-time" },
                    new InputData { MonthlyIncome = 60, LoanAmount = 200000, ReturnTime = 120, Age = 100, JobType = "Part-time" }
                };

                foreach (var sample in testSamples)
                {
                    var prediction = predEngine.Predict(sample);
                    Console.WriteLine($"Loan for {sample.JobType} earning {sample.MonthlyIncome} -> Approved: {prediction.Prediction}, Probability: {prediction.Probability:P2}");
                }
            }
            else
            {
                Console.WriteLine("Exiting program.");
            }
        }
    }
}
