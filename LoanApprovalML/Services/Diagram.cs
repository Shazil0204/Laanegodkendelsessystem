using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using LoanApprovalML.DataModels;
using Microsoft.ML;

namespace LoanApprovalML.Services
{
    /// <summary>
    /// This class is like an artist that creates visual graphs of our AI's decisions!
    /// 
    /// It takes our trained AI model and creates a pretty picture showing:
    /// - Green dots = loans that got approved
    /// - Red dots = loans that got rejected  
    /// - Blue line = the "decision boundary" (where the AI changes its mind)
    /// 
    /// The graph plots Loan Amount (X-axis) vs Monthly Income (Y-axis), so you can
    /// visually see patterns like "people with high income get bigger loans approved"
    /// </summary>
    public class Diagram
    {
        /// <summary>
        /// This is the main method that creates our beautiful visualization!
        /// It's like making a map that shows where our AI approves vs rejects loans.
        /// </summary>
        public void DrawApprovalDiagram()
        {
            // Step 1: Load our trained AI from the saved file
            // This is like opening a saved video game - we get our AI back exactly as we trained it
            var mlContext = new MLContext();
            ITransformer model = mlContext.Model.Load("model.zip", out var schema);
            var predEngine = mlContext.Model.CreatePredictionEngine<InputData, ModelOutput>(model);

            // Step 2: Load the actual historical loan data from our CSV file
            // We want to see what our AI thinks about real loan applications
            var dataLoader = new DataLoader(mlContext);
            var dataView = dataLoader.LoadData("data.csv");

            // Convert the data into a list we can easily work with
            var samples = mlContext.Data.CreateEnumerable<InputData>(dataView, reuseRowObject: false).ToList();

            Console.WriteLine($"Loaded {samples.Count} data points from CSV for visualization");

            // Step 3: Separate the data into two groups based on our AI's predictions
            // Think of sorting M&Ms into different colored bowls!
            var approvedX = new List<double>();  // X-coordinates (loan amounts) for approved loans
            var approvedY = new List<double>();  // Y-coordinates (monthly income) for approved loans
            var rejectedX = new List<double>();  // X-coordinates for rejected loans  
            var rejectedY = new List<double>();  // Y-coordinates for rejected loans

            // Ask our AI about each historical loan application
            foreach (var s in samples)
            {
                var pred = predEngine.Predict(s);  // "Hey AI, what do you think about this loan?"
                
                if (pred.Prediction)  // AI says "APPROVE"
                {
                    approvedX.Add(s.LoanAmount);
                    approvedY.Add(s.MonthlyIncome);
                }
                else  // AI says "REJECT"
                {
                    rejectedX.Add(s.LoanAmount);
                    rejectedY.Add(s.MonthlyIncome);
                }
            }

            // Step 4: Create our graph using ScottPlot (a graphing library)
            // Think of this as getting a blank canvas ready for painting
            var plt = new ScottPlot.Plot();

            // Add green dots for approved loans (if we have any)
            if (approvedX.Count > 0)
            {
                var approvedScatter = plt.Add.Scatter(approvedX.ToArray(), approvedY.ToArray());
                approvedScatter.Color = ScottPlot.Colors.Green;              // Green = good/approved
                approvedScatter.LegendText = $"Approved ({approvedX.Count})"; // Show count in legend
                approvedScatter.LineWidth = 0;                               // No lines connecting dots
                approvedScatter.MarkerSize = 8;                              // Make dots easy to see
            }

            // Add red dots for rejected loans (if we have any)
            if (rejectedX.Count > 0)
            {
                var rejectedScatter = plt.Add.Scatter(rejectedX.ToArray(), rejectedY.ToArray());
                rejectedScatter.Color = ScottPlot.Colors.Red;                // Red = bad/rejected
                rejectedScatter.LegendText = $"Rejected ({rejectedX.Count})"; // Show count in legend
                rejectedScatter.LineWidth = 0;                               // No lines connecting dots
                rejectedScatter.MarkerSize = 8;                              // Make dots easy to see
            }

            // Step 5: Decision boundary line removed per user request
            // Now showing only the approved (green) and rejected (red) data points

            // Step 6: Make our graph look professional with labels and titles
            plt.Title("Loan Approval Predictions");                        // Main title (removed "with Decision Boundary")
            plt.Axes.Bottom.Label.Text = "Loan Amount";                     // X-axis label
            plt.Axes.Left.Label.Text = "Monthly Income";                    // Y-axis label
            plt.ShowLegend();                                               // Show the color-coded legend

            // Step 7: Save our masterpiece as a PNG image file
            plt.SavePng("LoanPredictions.png", 800, 600);  // 800x600 pixel image
            Console.WriteLine($"Plot saved as LoanPredictions.png");
            Console.WriteLine($"Approved: {approvedX.Count}, Rejected: {rejectedX.Count}");
        }

        /// <summary>
        /// This method figures out where to draw the blue decision boundary line on our graph.
        /// The boundary represents the exact spot where our AI changes its mind from 
        /// "approve this loan" to "reject this loan".
        /// 
        /// It's like finding the line that separates the green dots from the red dots!
        /// </summary>
        private List<(double x, double y)> CalculateDecisionBoundary(
            PredictionEngine<InputData, ModelOutput> predEngine, 
            List<InputData> samples)
        {
            var boundaryPoints = new List<(double x, double y)>();
            
            // First, figure out the range of our data so we know where to look
            var minLoanAmount = samples.Min(s => s.LoanAmount);    // Smallest loan in our data
            var maxLoanAmount = samples.Max(s => s.LoanAmount);    // Biggest loan in our data
            var minIncome = samples.Min(s => s.MonthlyIncome);     // Lowest income in our data
            var maxIncome = samples.Max(s => s.MonthlyIncome);     // Highest income in our data

            Console.WriteLine($"Plotting decision boundary for data range: Loan ${minLoanAmount:N0}-${maxLoanAmount:N0}, Income ${minIncome:N0}-${maxIncome:N0}");

            // Now we'll check many different loan amounts across this range
            // For each loan amount, we'll find the income level where our AI is exactly 50% sure
            var step = (maxLoanAmount - minLoanAmount) / 50; // Check 50 different loan amounts
            
            for (double loanAmount = minLoanAmount; loanAmount <= maxLoanAmount; loanAmount += step)
            {
                // For this specific loan amount, what income gives us exactly 50% probability?
                // That's a point on our decision boundary!
                var income = FindBoundaryIncome(predEngine, loanAmount, minIncome, maxIncome);
                if (income.HasValue)
                {
                    boundaryPoints.Add((loanAmount, income.Value));
                }
            }

            return boundaryPoints;  // Return all the boundary points we found
        }

        /// <summary>
        /// This method is like playing "guess the number" with our AI!
        /// For a specific loan amount, we want to find the exact income level where
        /// our AI says "I'm 50% sure about approving this loan" (right on the fence).
        /// 
        /// We use binary search - start with a high and low guess, then keep splitting
        /// the difference until we find that sweet spot where probability ≈ 50%.
        /// </summary>
        private double? FindBoundaryIncome(PredictionEngine<InputData, ModelOutput> predEngine, 
            double loanAmount, double minIncome, double maxIncome)
        {
            const double tolerance = 0.05;  // Accept 45%-55% as "close enough" to 50%
            const int maxIterations = 20;   // Don't search forever - 20 tries should find it
            
            double low = minIncome;    // Start with lowest possible income
            double high = maxIncome;   // And highest possible income
            
            // Keep narrowing down our search until we find the boundary
            for (int i = 0; i < maxIterations; i++)
            {
                double mid = (low + high) / 2;  // Try the middle value
                
                // Create a fake loan application with this income to test our AI
                var testData = new InputData
                {
                    LoanAmount = (float)loanAmount,
                    MonthlyIncome = (float)mid,
                    Age = 35,                     // Use typical/average values for other fields
                    ReturnTime = 24,              // 2 years is pretty common
                    JobType = "Full-time"         // Most stable employment type
                };
                
                // Ask our AI: "What's your confidence level for this loan?"
                var prediction = predEngine.Predict(testData);
                double probability = prediction.Probability;
                
                // Are we close enough to 50%? Mission accomplished!
                if (Math.Abs(probability - 0.5) < tolerance)
                {
                    return mid; // Found our boundary income!
                }
                
                // Binary search magic: adjust our search range based on the AI's answer
                if (probability > 0.5)
                {
                    // AI is too confident about approval - try lower income to make it less sure
                    high = mid;
                }
                else
                {
                    // AI is not confident enough - try higher income to boost confidence
                    low = mid;
                }
                
                // If our search range gets too narrow, we're close enough
                if (high - low < 100) break; // Within $100 is pretty precise!
            }
            
            return null; // Couldn't find a clear 50% boundary point
        }

    }
}