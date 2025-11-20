using Microsoft.ML;
using Microsoft.ML.Data;
using LoanApprovalML.DataModels;

namespace LoanApprovalML.Services
{
    /// <summary>
    /// This class is like a "report card generator" for our AI.
    /// After we train our AI, we need to know: "How good is it really?"
    /// 
    /// The Evaluator tests our AI on data it has never seen before and gives us
    /// scores like accuracy, precision, recall, etc. It's like giving a final exam
    /// to see how well the AI learned to make loan decisions.
    /// 
    /// It also tries to figure out the "decision boundary" - basically, what rule
    /// is the AI using to decide between approve/reject?
    /// </summary>
    public class Evaluator(MLContext mlContext)
    {
        // We need MLContext to run our tests and calculations
        private readonly MLContext _mlContext = mlContext;

        /// <summary>
        /// This is the main "report card" method. It tests our AI and prints out all the scores.
        /// Think of it like grading a student's test and telling them how they did.
        /// </summary>
        /// <param name="model">Our trained AI model</param>
        /// <param name="testData">Test data the AI has never seen before</param>
        public void Evaluate(ITransformer model, IDataView testData)
        {
            // First, let our AI make predictions on the test data
            // This is like giving the AI a final exam
            var predictions = model.Transform(testData);
            
            // Calculate various metrics to see how well our AI did
            // ML.NET can automatically calculate most of these for us
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "IsApproved");

            // Calculate MSE (Mean Squared Error) manually because ML.NET doesn't do this for binary classification
            // MSE tells us on average how far off our probability predictions were
            var mse = CalculateMSE(model, testData);

            // Print out all the scores! 
            Console.WriteLine("=== Model Evaluation Metrics ===");
            
            // Accuracy: What percentage of predictions were correct? (Higher = better)
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            
            // Precision: Of all the loans we said "approve", how many were actually correct?
            // High precision = we don't approve bad loans
            Console.WriteLine($"Precision: {metrics.PositivePrecision:F4}");
            
            // Recall: Of all the loans that should be approved, how many did we catch?
            // High recall = we don't miss good loan applications
            Console.WriteLine($"Recall: {metrics.PositiveRecall:F4}");
            
            // F1 Score: A balance between precision and recall (Higher = better)
            Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
            
            // AUC: How good is our AI at distinguishing between approve/reject? (Closer to 1.0 = better)
            Console.WriteLine($"AUC (Area Under Curve): {metrics.AreaUnderRocCurve:F4}");
            
            // MSE & RMSE: How far off were our probability predictions on average? (Lower = better)
            Console.WriteLine($"MSE (Mean Squared Error): {mse:F4}");
            Console.WriteLine($"RMSE (Root Mean Squared Error): {Math.Sqrt(mse):F4}");
            
            // The Confusion Matrix shows exactly where our AI got things right and wrong
            Console.WriteLine("\n=== Confusion Matrix ===");
            Console.WriteLine($"True Positive: {metrics.ConfusionMatrix.Counts[0][0]}");   // Correctly approved
            Console.WriteLine($"False Positive: {metrics.ConfusionMatrix.Counts[0][1]}");  // Wrongly approved (bad!)
            Console.WriteLine($"False Negative: {metrics.ConfusionMatrix.Counts[1][0]}");  // Wrongly rejected
            Console.WriteLine($"True Negative: {metrics.ConfusionMatrix.Counts[1][1]}");   // Correctly rejected

            // Try to figure out the mathematical formula our AI is using
            // This is like trying to understand the "rule" in the AI's head
            var formula = CalculateDecisionBoundaryFormula(model, testData);
            
            Console.WriteLine("\n=== Decision Boundary ===");
            Console.WriteLine($"Formula: {formula}");
            
            // Give some human-friendly interpretation of the results 
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

        /// <summary>
        /// MSE (Mean Squared Error) measures how far off our probability predictions are on average.
        /// For example, if the AI says "80% chance of approval" but the loan was actually rejected,
        /// that's a pretty big error! This method calculates the average of all such errors.
        /// 
        /// Think of it like this: if you're guessing people's heights and you're always off by 5 inches,
        /// your MSE would show how consistently wrong you are.
        /// </summary>
        private double CalculateMSE(ITransformer model, IDataView testData)
        {
            // Get predictions from our model
            var predictions = model.Transform(testData);
            
            // Create a "prediction engine" - this lets us ask our AI questions one at a time
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<InputData, ModelOutput>(model);
            
            // Convert our test data into a list we can loop through
            var testDataEnumerable = _mlContext.Data.CreateEnumerable<InputData>(testData, reuseRowObject: false);
            
            double sumSquaredErrors = 0;  // Keep track of total error
            int count = 0;                // Count how many predictions we made
            
            // Go through each test case and calculate the error
            foreach (var testRow in testDataEnumerable)
            {
                var prediction = predictionEngine.Predict(testRow);
                
                // Convert the actual answer to a number (true = 1, false = 0)
                double actualValue = testRow.IsApproved ? 1.0 : 0.0;
                
                // The AI's prediction is a probability between 0 and 1
                double predictedValue = prediction.Probability;
                
                // Calculate how far off we were and square it (to make negatives positive)
                double error = actualValue - predictedValue;
                sumSquaredErrors += error * error;
                count++;
            }
            
            // Return the average squared error (that's what "mean" means in MSE)
            return count > 0 ? sumSquaredErrors / count : 0;
        }

        /// <summary>
        /// This method tries to figure out the mathematical rule our AI is using to make decisions.
        /// It's like trying to understand: "What's the line that separates approved from rejected loans?"
        /// 
        /// Imagine plotting all loan applications on a graph with LoanAmount on X-axis and Income on Y-axis.
        /// This method tries to find the line that separates the green dots (approved) from red dots (rejected).
        /// 
        /// The result might be something like "f(y) = 2.5x + 1000" which could mean:
        /// "Your income needs to be at least 2.5 times your loan amount plus $1000 to get approved"
        /// </summary>
        private string CalculateDecisionBoundaryFormula(ITransformer model, IDataView testData)
        {
            // Set up tools to ask our AI questions
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<InputData, ModelOutput>(model);
            var testSamples = _mlContext.Data.CreateEnumerable<InputData>(testData, reuseRowObject: false).ToList();
            
            if (testSamples.Count == 0)
            {
                return "No test data available for boundary calculation";
            }

            var boundaryPoints = new List<(double x, double y)>();
            
            // Find the range of our data (min/max loan amounts and incomes)
            // This tells us the "playing field" we're working with
            var minLoanAmount = testSamples.Min(s => s.LoanAmount);
            var maxLoanAmount = testSamples.Max(s => s.LoanAmount);
            var minIncome = testSamples.Min(s => s.MonthlyIncome);
            var maxIncome = testSamples.Max(s => s.MonthlyIncome);

            // Now we're going to find points where our AI is "on the fence" (50% probability)
            // These points form the decision boundary line
            var step = (maxLoanAmount - minLoanAmount) / 30; // Check 30 different loan amounts
            
            for (double loanAmount = minLoanAmount; loanAmount <= maxLoanAmount; loanAmount += step)
            {
                // For this loan amount, what income gives us exactly 50% approval probability?
                var income = FindBoundaryIncome(predictionEngine, loanAmount, minIncome, maxIncome);
                if (income.HasValue)
                {
                    boundaryPoints.Add((loanAmount, income.Value));
                }
            }

            // If we found enough boundary points, try to fit a line through them
            if (boundaryPoints.Count >= 2)
            {
                var (slope, intercept, rSquared) = CalculateLinearRegression(boundaryPoints);
                
                // R² tells us how well a straight line fits our points
                // If R² > 0.7, our decision boundary is roughly a straight line
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

        /// <summary>
        /// This method uses "binary search" to find the exact income level where our AI is unsure (50% probability).
        /// Think of it like playing "hot or cold" to find the decision boundary.
        /// 
        /// For a given loan amount, what income level makes the AI say "I'm exactly 50% sure about this"?
        /// That's the boundary between approve and reject!
        /// 
        /// Binary search works like guessing a number: if the AI says "too high", we guess lower.
        /// If it says "too low", we guess higher. We keep narrowing down until we hit 50%.
        /// </summary>
        private double? FindBoundaryIncome(PredictionEngine<InputData, ModelOutput> predEngine, 
            double loanAmount, double minIncome, double maxIncome)
        {
            const double tolerance = 0.05;    // Accept anything between 45%-55% as "close enough to 50%"
            const int maxIterations = 15;     // Don't search forever - 15 guesses should be enough
            
            double low = minIncome;   // Start with the lowest possible income
            double high = maxIncome;  // And the highest possible income
            
            // Keep guessing until we find the boundary or run out of tries
            for (int i = 0; i < maxIterations; i++)
            {
                double mid = (low + high) / 2;  // Guess the middle value
                
                // Create a fake loan application with this income level
                // We use average values for other fields to keep things simple
                var testData = new InputData
                {
                    LoanAmount = (float)loanAmount,
                    MonthlyIncome = (float)mid,
                    Age = 35,                    // Average age
                    ReturnTime = 24,             // Average return time in months
                    JobType = "Full-time"        // Most stable job type
                };
                
                // Ask our AI: "What do you think about this loan application?"
                var prediction = predEngine.Predict(testData);
                double probability = prediction.Probability;
                
                // Are we close enough to 50%? If so, we found our boundary!
                if (Math.Abs(probability - 0.5) < tolerance)
                {
                    return mid; // Found it!
                }
                
                // Binary search logic: adjust our search range based on the result
                if (probability > 0.5)
                {
                    high = mid; // Probability too high, try lower income
                }
                else
                {
                    low = mid; // Probability too low, try higher income
                }
                
                // If our search range gets too small, we're close enough
                if (high - low < 100) break;
            }
            
            return null; // Couldn't find a clear boundary point
        }

        /// <summary>
        /// This method fits a straight line through our boundary points using linear regression.
        /// Remember from school: y = mx + b? This calculates the 'm' (slope) and 'b' (intercept).
        /// 
        /// It also calculates R² (R-squared) which tells us how well a straight line fits our points:
        /// - R² = 1.0 means "perfect straight line"
        /// - R² = 0.8 means "pretty good straight line" 
        /// - R² = 0.3 means "not really a straight line at all"
        /// 
        /// If R² is high, we can describe the decision boundary with a simple formula!
        /// </summary>
        private (double slope, double intercept, double rSquared) CalculateLinearRegression(
            List<(double x, double y)> points)
        {
            int n = points.Count;
            
            // Calculate all the sums we need for the linear regression formula
            // This is basic statistics - finding the line that best fits the points
            double sumX = points.Sum(p => p.x);     // Sum of all X values (loan amounts)
            double sumY = points.Sum(p => p.y);     // Sum of all Y values (incomes)
            double sumXY = points.Sum(p => p.x * p.y); // Sum of X*Y products
            double sumXX = points.Sum(p => p.x * p.x); // Sum of X squared
            double sumYY = points.Sum(p => p.y * p.y); // Sum of Y squared
            
            // Calculate slope (m) and intercept (b) using the least squares formula
            // This gives us the line that minimizes the distance to all points
            double slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
            double intercept = (sumY - slope * sumX) / n;
            
            // Calculate R² to see how good our line fit is
            double meanY = sumY / n;  // Average Y value
            double totalSumSquares = points.Sum(p => Math.Pow(p.y - meanY, 2));      // Total variation
            double residualSumSquares = points.Sum(p => Math.Pow(p.y - (slope * p.x + intercept), 2)); // Unexplained variation
            double rSquared = 1 - (residualSumSquares / totalSumSquares);  // R² = explained variation / total variation
            
            return (slope, intercept, rSquared);
        }
    }
}
