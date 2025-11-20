using Microsoft.ML;
using Microsoft.ML.Transforms;
using LoanApprovalML.DataModels;

namespace LoanApprovalML.Services
{
    /// <summary>
    /// This is the "teacher" class - it's responsible for training our AI to make loan decisions.
    /// Think of it like a driving instructor teaching someone how to drive, but instead
    /// we're teaching a computer how to approve or reject loan applications.
    /// 
    /// The process is:
    /// 1. Show the AI lots of examples (historical loan data)
    /// 2. Let it learn patterns (people with high income usually get approved)
    /// 3. Test how well it learned
    /// 4. Save the trained AI so we can use it later
    /// </summary>
    public class Trainer
    {
        // These are our helper tools for training
        private readonly MLContext _mlContext;      // The main ML.NET workspace
        private readonly DataLoader _dataLoader;    // Loads data from CSV files
        private readonly Evaluator _evaluator;      // Tests how good our AI is

        public Trainer()
        {
            // Set up all our tools when we create a new Trainer
            _mlContext = new MLContext();
            _dataLoader = new DataLoader(_mlContext);
            _evaluator = new Evaluator(_mlContext);
        }

        /// <summary>
        /// This is the main training method - where all the magic happens!
        /// It takes a CSV file full of loan applications and teaches our AI to make decisions.
        /// </summary>
        /// <param name="dataPath">Path to our CSV file with loan application data</param>
        public void Train(string dataPath)
        {
            // Step 1: Load all the loan application data from our CSV file
            // Think of this as opening a big spreadsheet of past loan decisions
            var data = _dataLoader.LoadData(dataPath);

            // Step 2: Split the data into training and testing parts
            // It's like this: use 80% to teach the AI, save 20% to test how well it learned
            // This prevents "cheating" - we never test on data the AI has seen before!
            var trainTest = _mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // Step 3: Build our "learning pipeline" - this is the recipe for training our AI
            /*
            🔧 In simpler terms:
            - Step 1: Convert job types (like "Student", "Full-time") into numbers the AI can understand
            - Step 2: Bundle all the information together into one package
            - Step 3: Choose a learning algorithm (SDCA Logistic Regression) to find patterns
            */
            var pipeline = _mlContext.Transforms.Categorical.OneHotEncoding("JobType")
                .Append(_mlContext.Transforms.Concatenate("Features", "MonthlyIncome", "LoanAmount", "ReturnTime", "Age", "JobType"))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "IsApproved"));

            // Step 4: Actually train our AI using the training data
            // This is where the computer learns patterns like "people with income > loan amount usually get approved"
            var model = pipeline.Fit(trainTest.TrainSet);

            // Step 5: Test our AI on data it has never seen before
            // This tells us how good our AI really is at making loan decisions
            _evaluator.Evaluate(model, trainTest.TestSet);

            // Step 6: Save our trained AI to a file so we can use it later
            // Think of this like saving a video game - we can load it up later without re-training
            _mlContext.Model.Save(model, data.Schema, "model.zip");

            Console.WriteLine("Model saved as model.zip");
        }
    }
}
