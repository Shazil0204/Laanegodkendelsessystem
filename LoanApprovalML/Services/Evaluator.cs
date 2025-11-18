using Microsoft.ML;
using Microsoft.ML.Data;
using LoanApprovalML.DataModels;

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

            // Calculate MSE manually
            var mse = CalculateMSE(model, testData);

            Console.WriteLine("=== Model Evaluation Metrics ===");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Precision: {metrics.PositivePrecision:F4}");
            Console.WriteLine($"Recall: {metrics.PositiveRecall:F4}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
            Console.WriteLine($"AUC (Area Under Curve): {metrics.AreaUnderRocCurve:F4}");
            Console.WriteLine($"MSE (Mean Squared Error): {mse:F4}");
            Console.WriteLine($"RMSE (Root Mean Squared Error): {Math.Sqrt(mse):F4}");
            
            Console.WriteLine("\n=== Confusion Matrix ===");
            Console.WriteLine($"True Positive: {metrics.ConfusionMatrix.Counts[0][0]}");
            Console.WriteLine($"False Positive: {metrics.ConfusionMatrix.Counts[0][1]}");
            Console.WriteLine($"False Negative: {metrics.ConfusionMatrix.Counts[1][0]}");
            Console.WriteLine($"True Negative: {metrics.ConfusionMatrix.Counts[1][1]}");

            // Calculate decision boundary formula
            var formula = CalculateDecisionBoundaryFormula(model, testData);
            
            Console.WriteLine("\n=== Decision Boundary ===");
            Console.WriteLine($"Formula: {formula}");
            
            // Additional interpretation
            Console.WriteLine("\n=== Interpretation ===");
            if (metrics.Accuracy > 0.85)
                Console.WriteLine("✓ Excellent model accuracy");
            else if (metrics.Accuracy > 0.75)
                Console.WriteLine("✓ Good model accuracy");
            else
                Console.WriteLine("⚠ Consider improving the model");

            if (mse < 0.1)
                Console.WriteLine("✓ Very low prediction error (MSE)");
            else if (mse < 0.2)
                Console.WriteLine("✓ Low prediction error (MSE)");
            else
                Console.WriteLine("⚠ High prediction error - consider model tuning");
        }

        private double CalculateMSE(ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<InputData, ModelOutput>(model);
            
            var testDataEnumerable = _mlContext.Data.CreateEnumerable<InputData>(testData, reuseRowObject: false);
            
            double sumSquaredErrors = 0;
            int count = 0;
            
            foreach (var testRow in testDataEnumerable)
            {
                var prediction = predictionEngine.Predict(testRow);
                
                // Convert boolean to double (true = 1, false = 0)
                double actualValue = testRow.IsApproved ? 1.0 : 0.0;
                double predictedValue = prediction.Probability;
                
                // Calculate squared error
                double error = actualValue - predictedValue;
                sumSquaredErrors += error * error;
                count++;
            }
            
            return count > 0 ? sumSquaredErrors / count : 0;
        }

        private string CalculateDecisionBoundaryFormula(ITransformer model, IDataView testData)
        {
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<InputData, ModelOutput>(model);
            var testSamples = _mlContext.Data.CreateEnumerable<InputData>(testData, reuseRowObject: false).ToList();
            
            if (testSamples.Count == 0)
            {
                return "No test data available for boundary calculation";
            }

            var boundaryPoints = new List<(double x, double y)>();
            
            // Find the range of data
            var minLoanAmount = testSamples.Min(s => s.LoanAmount);
            var maxLoanAmount = testSamples.Max(s => s.LoanAmount);
            var minIncome = testSamples.Min(s => s.MonthlyIncome);
            var maxIncome = testSamples.Max(s => s.MonthlyIncome);

            // Calculate decision boundary by finding points where probability ≈ 0.5
            var step = (maxLoanAmount - minLoanAmount) / 30; // 30 points across the range
            
            for (double loanAmount = minLoanAmount; loanAmount <= maxLoanAmount; loanAmount += step)
            {
                var income = FindBoundaryIncome(predictionEngine, loanAmount, minIncome, maxIncome);
                if (income.HasValue)
                {
                    boundaryPoints.Add((loanAmount, income.Value));
                }
            }

            if (boundaryPoints.Count >= 2)
            {
                var (slope, intercept, rSquared) = CalculateLinearRegression(boundaryPoints);
                
                if (rSquared > 0.7)
                {
                    return $"f(y) = {slope:F4}x + {intercept:F0} (R² = {rSquared:F3})";
                }
                else
                {
                    return $"Non-linear boundary (R² = {rSquared:F3})";
                }
            }
            
            return "Could not determine decision boundary";
        }

        private double? FindBoundaryIncome(PredictionEngine<InputData, ModelOutput> predEngine, 
            double loanAmount, double minIncome, double maxIncome)
        {
            const double tolerance = 0.05;
            const int maxIterations = 15;
            
            double low = minIncome;
            double high = maxIncome;
            
            for (int i = 0; i < maxIterations; i++)
            {
                double mid = (low + high) / 2;
                
                var testData = new InputData
                {
                    LoanAmount = (float)loanAmount,
                    MonthlyIncome = (float)mid,
                    Age = 35,
                    ReturnTime = 24,
                    JobType = "Full-time"
                };
                
                var prediction = predEngine.Predict(testData);
                double probability = prediction.Probability;
                
                if (Math.Abs(probability - 0.5) < tolerance)
                {
                    return mid;
                }
                
                if (probability > 0.5)
                {
                    high = mid;
                }
                else
                {
                    low = mid;
                }
                
                if (high - low < 100) break;
            }
            
            return null;
        }

        private (double slope, double intercept, double rSquared) CalculateLinearRegression(
            List<(double x, double y)> points)
        {
            int n = points.Count;
            double sumX = points.Sum(p => p.x);
            double sumY = points.Sum(p => p.y);
            double sumXY = points.Sum(p => p.x * p.y);
            double sumXX = points.Sum(p => p.x * p.x);
            double sumYY = points.Sum(p => p.y * p.y);
            
            double slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
            double intercept = (sumY - slope * sumX) / n;
            
            double meanY = sumY / n;
            double totalSumSquares = points.Sum(p => Math.Pow(p.y - meanY, 2));
            double residualSumSquares = points.Sum(p => Math.Pow(p.y - (slope * p.x + intercept), 2));
            double rSquared = 1 - (residualSumSquares / totalSumSquares);
            
            return (slope, intercept, rSquared);
        }
    }
}
