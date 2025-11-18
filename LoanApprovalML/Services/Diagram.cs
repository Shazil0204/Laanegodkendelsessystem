using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using LoanApprovalML.DataModels;
using Microsoft.ML;

namespace LoanApprovalML.Services
{
    public class Diagram
    {
        public void DrawApprovalDiagram()
        {
            // Load the trained model
            var mlContext = new MLContext();
            ITransformer model = mlContext.Model.Load("model.zip", out var schema);
            var predEngine = mlContext.Model.CreatePredictionEngine<InputData, ModelOutput>(model);

            // Load actual data from CSV file
            var dataLoader = new DataLoader(mlContext);
            var dataView = dataLoader.LoadData("data.csv");

            // Convert IDataView to List<InputData> for easier processing
            var samples = mlContext.Data.CreateEnumerable<InputData>(dataView, reuseRowObject: false).ToList();

            Console.WriteLine($"Loaded {samples.Count} data points from CSV for visualization");

            // Separate approved vs rejected for plotting
            var approvedX = new List<double>();
            var approvedY = new List<double>();
            var rejectedX = new List<double>();
            var rejectedY = new List<double>();

            foreach (var s in samples)
            {
                var pred = predEngine.Predict(s);
                if (pred.Prediction)
                {
                    approvedX.Add(s.LoanAmount);
                    approvedY.Add(s.MonthlyIncome);
                }
                else
                {
                    rejectedX.Add(s.LoanAmount);
                    rejectedY.Add(s.MonthlyIncome);
                }
            }

            // Create a plot (ScottPlot 5.x API)
            var plt = new ScottPlot.Plot();

            if (approvedX.Count > 0)
            {
                var approvedScatter = plt.Add.Scatter(approvedX.ToArray(), approvedY.ToArray());
                approvedScatter.Color = ScottPlot.Colors.Green;
                approvedScatter.LegendText = $"Approved ({approvedX.Count})";
                approvedScatter.LineWidth = 0; // Remove lines between points
                approvedScatter.MarkerSize = 8; // Set marker size
            }

            if (rejectedX.Count > 0)
            {
                var rejectedScatter = plt.Add.Scatter(rejectedX.ToArray(), rejectedY.ToArray());
                rejectedScatter.Color = ScottPlot.Colors.Red;
                rejectedScatter.LegendText = $"Rejected ({rejectedX.Count})";
                rejectedScatter.LineWidth = 0; // Remove lines between points
                rejectedScatter.MarkerSize = 8; // Set marker size
            }

            // Calculate and draw decision boundary
            var boundaryPoints = CalculateDecisionBoundary(predEngine, samples);
            
            if (boundaryPoints.Count > 1)
            {
                var boundaryX = boundaryPoints.Select(p => p.x).ToArray();
                var boundaryY = boundaryPoints.Select(p => p.y).ToArray();
                
                var boundaryLine = plt.Add.Scatter(boundaryX, boundaryY);
                boundaryLine.Color = ScottPlot.Colors.Blue;
                boundaryLine.LegendText = "Decision Boundary";
                boundaryLine.LineWidth = 3;
                boundaryLine.MarkerSize = 0; // Only show line, no markers
            }

            // Configure the plot
            plt.Title("Loan Approval Predictions with Decision Boundary");
            plt.Axes.Bottom.Label.Text = "Loan Amount";
            plt.Axes.Left.Label.Text = "Monthly Income";
            plt.ShowLegend();

            // Save the plot as PNG
            plt.SavePng("LoanPredictions.png", 800, 600);
            Console.WriteLine($"Plot saved as LoanPredictions.png");
            Console.WriteLine($"Approved: {approvedX.Count}, Rejected: {rejectedX.Count}");
        }

        private List<(double x, double y)> CalculateDecisionBoundary(
            PredictionEngine<InputData, ModelOutput> predEngine, 
            List<InputData> samples)
        {
            var boundaryPoints = new List<(double x, double y)>();
            
            // Find the range of data
            var minLoanAmount = samples.Min(s => s.LoanAmount);
            var maxLoanAmount = samples.Max(s => s.LoanAmount);
            var minIncome = samples.Min(s => s.MonthlyIncome);
            var maxIncome = samples.Max(s => s.MonthlyIncome);

            Console.WriteLine($"Plotting decision boundary for data range: Loan ${minLoanAmount:N0}-${maxLoanAmount:N0}, Income ${minIncome:N0}-${maxIncome:N0}");

            // Calculate decision boundary by finding points where probability ≈ 0.5
            var step = (maxLoanAmount - minLoanAmount) / 50; // 50 points across the range
            
            for (double loanAmount = minLoanAmount; loanAmount <= maxLoanAmount; loanAmount += step)
            {
                // Binary search to find income where probability ≈ 0.5
                var income = FindBoundaryIncome(predEngine, loanAmount, minIncome, maxIncome);
                if (income.HasValue)
                {
                    boundaryPoints.Add((loanAmount, income.Value));
                }
            }

            return boundaryPoints;
        }

        private double? FindBoundaryIncome(PredictionEngine<InputData, ModelOutput> predEngine, 
            double loanAmount, double minIncome, double maxIncome)
        {
            const double tolerance = 0.05; // Accept probability between 0.45-0.55 as boundary
            const int maxIterations = 20;
            
            double low = minIncome;
            double high = maxIncome;
            
            for (int i = 0; i < maxIterations; i++)
            {
                double mid = (low + high) / 2;
                
                var testData = new InputData
                {
                    LoanAmount = (float)loanAmount,
                    MonthlyIncome = (float)mid,
                    Age = 35, // Average age
                    ReturnTime = 24, // Average return time
                    JobType = "Full-time" // Most common job type
                };
                
                var prediction = predEngine.Predict(testData);
                double probability = prediction.Probability;
                
                if (Math.Abs(probability - 0.5) < tolerance)
                {
                    return mid; // Found boundary
                }
                
                if (probability > 0.5)
                {
                    high = mid; // Too high probability, reduce income
                }
                else
                {
                    low = mid; // Too low probability, increase income
                }
                
                if (high - low < 100) break; // Converged enough
            }
            
            return null; // Couldn't find clear boundary
        }

    }
}