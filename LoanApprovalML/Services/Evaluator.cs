using Microsoft.ML;
using Microsoft.ML.Data;

namespace LoanApprovalML.Services
{
    public class Evaluator
    {
        private readonly MLContext _mlContext;

        public Evaluator(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        public void Evaluate(ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "IsApproved");

            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine("Confusion Matrix:");
            Console.WriteLine($"True Positive: {metrics.ConfusionMatrix.Counts[0][0]}");
            Console.WriteLine($"False Positive: {metrics.ConfusionMatrix.Counts[0][1]}");
            Console.WriteLine($"False Negative: {metrics.ConfusionMatrix.Counts[1][0]}");
            Console.WriteLine($"True Negative: {metrics.ConfusionMatrix.Counts[1][1]}");
        }
    }
}
