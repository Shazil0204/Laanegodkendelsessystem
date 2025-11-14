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

            // Configure the plot
            plt.Title("Loan Approval Predictions (Real Data from CSV)");
            plt.Axes.Bottom.Label.Text = "Loan Amount";
            plt.Axes.Left.Label.Text = "Monthly Income";
            plt.ShowLegend();

            // Save the plot as PNG
            plt.SavePng("LoanPredictions.png", 600, 400);
            Console.WriteLine($"Plot saved as LoanPredictions.png");
            Console.WriteLine($"Approved: {approvedX.Count}, Rejected: {rejectedX.Count}");
        }
    }
}