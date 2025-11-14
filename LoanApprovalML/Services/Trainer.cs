using Microsoft.ML;
using Microsoft.ML.Transforms;
using LoanApprovalML.DataModels;

namespace LoanApprovalML.Services
{
    public class Trainer
    {
        private readonly MLContext _mlContext;
        private readonly DataLoader _dataLoader;
        private readonly Evaluator _evaluator;

        public Trainer()
        {
            _mlContext = new MLContext();
            _dataLoader = new DataLoader(_mlContext);
            _evaluator = new Evaluator(_mlContext);
        }

        public void Train(string dataPath)
        {
            // 1. Load data
            var data = _dataLoader.LoadData(dataPath);

            // 2. Split data
            var trainTest = _mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // 3. Build pipeline
            var pipeline = _mlContext.Transforms.Categorical.OneHotEncoding("JobType")
                .Append(_mlContext.Transforms.Concatenate("Features", "MonthlyIncome", "LoanAmount", "ReturnTime", "Age", "JobType"))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "IsApproved"));

            // 4. Train
            var model = pipeline.Fit(trainTest.TrainSet);

            // 5. Evaluate
            _evaluator.Evaluate(model, trainTest.TestSet);

            // 6. Save model
            _mlContext.Model.Save(model, data.Schema, "model.zip");

            Console.WriteLine("Model saved as model.zip");
        }
    }
}
