using LoanApprovalML.DataModels;
using Microsoft.ML;

namespace LoanApprovalML.Services
{
    /// <summary>
    /// This class handles all the loan application processing and user input.
    /// It's like the loan officer who collects your application details and 
    /// submits them to the AI for decision-making.
    /// </summary>
    public class LoanApplicationProcessor
    {
        private readonly MLContext _mlContext;

        public LoanApplicationProcessor(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        /// <summary>
        /// Main method that handles the entire loan application process
        /// </summary>
        public void ProcessLoanApplication()
        {
            try
            {
                // Load the trained AI model
                ITransformer loadedModel = _mlContext.Model.Load("model.zip", out var schema);
                var predictionEngine = _mlContext.Model.CreatePredictionEngine<InputData, ModelOutput>(loadedModel);

                // Collect loan application data from user
                var userApplication = CollectLoanApplicationData();

                // Get AI decision and display results
                DisplayLoanDecision(predictionEngine, userApplication);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error processing loan application: {ex.Message}");
            }
        }

        /// <summary>
        /// Collects all loan application data from the user with validation
        /// </summary>
        private InputData CollectLoanApplicationData()
        {
            Console.WriteLine("\n📋 Loan Application Form");
            Console.WriteLine("Please provide the following information:\n");

            var application = new InputData();

            // Collect each piece of data with proper validation
            application.MonthlyIncome = GetMonthlyIncome();
            application.LoanAmount = GetLoanAmount();
            application.ReturnTime = GetReturnTime();
            application.Age = GetAge();
            application.JobType = GetJobType();

            return application;
        }

        /// <summary>
        /// Gets monthly income with validation and error handling
        /// </summary>
        private float GetMonthlyIncome()
        {
            while (true)
            {
                try
                {
                    Console.Write("💰 Monthly Income (e.g., 4000): DKK ");
                    string? input = Console.ReadLine();
                    
                    if (string.IsNullOrWhiteSpace(input))
                    {
                        Console.WriteLine("⚠️  Please enter a valid income amount.\n");
                        continue;
                    }

                    float income = float.Parse(input);
                    
                    if (income <= 0)
                    {
                        Console.WriteLine("⚠️  Income must be greater than 0. Please try again.\n");
                        continue;
                    }

                    if (income > 1000000)
                    {
                        Console.WriteLine("⚠️  Income seems unusually high. Please verify and re-enter.\n");
                        continue;
                    }

                    return income;
                }
                catch (FormatException)
                {
                    Console.WriteLine("⚠️  Invalid format! Please enter a number (e.g., 4000).\n");
                }
                catch (OverflowException)
                {
                    Console.WriteLine("⚠️  Number is too large! Please enter a reasonable income amount.\n");
                }
            }
        }

        /// <summary>
        /// Gets loan amount with validation and error handling
        /// </summary>
        private float GetLoanAmount()
        {
            while (true)
            {
                try
                {
                    Console.Write("🏠 Loan Amount (e.g., 10000): DKK ");
                    string? input = Console.ReadLine();
                    
                    if (string.IsNullOrWhiteSpace(input))
                    {
                        Console.WriteLine("⚠️  Please enter a valid loan amount.\n");
                        continue;
                    }

                    float loanAmount = float.Parse(input);
                    
                    if (loanAmount <= 0)
                    {
                        Console.WriteLine("⚠️  Loan amount must be greater than 0. Please try again.\n");
                        continue;
                    }

                    if (loanAmount > 10000000)
                    {
                        Console.WriteLine("⚠️  Loan amount seems unusually high. Please verify and re-enter.\n");
                        continue;
                    }

                    return loanAmount;
                }
                catch (FormatException)
                {
                    Console.WriteLine("⚠️  Invalid format! Please enter a number (e.g., 10000).\n");
                }
                catch (OverflowException)
                {
                    Console.WriteLine("⚠️  Number is too large! Please enter a reasonable loan amount.\n");
                }
            }
        }

        /// <summary>
        /// Gets return time with validation and error handling
        /// </summary>
        private float GetReturnTime()
        {
            while (true)
            {
                try
                {
                    Console.Write("📅 Payback Time in months (e.g., 24 for 2 years): ");
                    string? input = Console.ReadLine();
                    
                    if (string.IsNullOrWhiteSpace(input))
                    {
                        Console.WriteLine("⚠️  Please enter a valid number of months.\n");
                        continue;
                    }

                    float returnTime = float.Parse(input);
                    
                    if (returnTime <= 0 || returnTime > 360)
                    {
                        Console.WriteLine("⚠️  Return time must be between 1 and 360 months (30 years). Please try again.\n");
                        continue;
                    }

                    return returnTime;
                }
                catch (FormatException)
                {
                    Console.WriteLine("⚠️  Invalid format! Please enter a number (e.g., 24).\n");
                }
                catch (OverflowException)
                {
                    Console.WriteLine("⚠️  Number is too large! Please enter a reasonable number of months.\n");
                }
            }
        }

        /// <summary>
        /// Gets age with validation and error handling
        /// </summary>
        private float GetAge()
        {
            while (true)
            {
                try
                {
                    Console.Write("🎂 Your Age (e.g., 30): ");
                    string? input = Console.ReadLine();
                    
                    if (string.IsNullOrWhiteSpace(input))
                    {
                        Console.WriteLine("⚠️  Please enter a valid age.\n");
                        continue;
                    }

                    float age = float.Parse(input);
                    
                    if (age < 18 || age > 120)
                    {
                        Console.WriteLine("⚠️  Age must be between 18 and 120. Please try again.\n");
                        continue;
                    }

                    return age;
                }
                catch (FormatException)
                {
                    Console.WriteLine("⚠️  Invalid format! Please enter a number (e.g., 30).\n");
                }
                catch (OverflowException)
                {
                    Console.WriteLine("⚠️  Number is too large! Please enter a reasonable age.\n");
                }
            }
        }

        /// <summary>
        /// Gets job type with validation and error handling
        /// </summary>
        private string GetJobType()
        {
            while (true)
            {
                try
                {
                    Console.Write("💼 Job Type (Full-time/Part-time/Student/Unemployed): ");
                    string? input = Console.ReadLine();
                    
                    if (string.IsNullOrWhiteSpace(input))
                    {
                        Console.WriteLine("⚠️  Please enter a job type.\n");
                        continue;
                    }

                    // Normalize the input to match expected values
                    string normalizedJob = input.Trim().ToLower();
                    
                    return normalizedJob switch
                    {
                        "full-time" or "fulltime" or "full time" or "1" => "Full-time",
                        "part-time" or "parttime" or "part time" or "2" => "Part-time",
                        "student" or "3" => "Student",
                        "unemployed" or "4" => "Unemployed",
                        _ => throw new ArgumentException("Invalid job type")
                    };
                }
                catch (ArgumentException)
                {
                    Console.WriteLine("⚠️  Please enter one of: Full-time, Part-time, Student, or Unemployed\n");
                    Console.WriteLine("     You can also use numbers: 1=Full-time, 2=Part-time, 3=Student, 4=Unemployed\n");
                }
                catch (Exception)
                {
                    Console.WriteLine("⚠️  Error processing job type. Please try again.\n");
                }
            }
        }

        /// <summary>
        /// Gets AI decision and displays formatted results to the user
        /// </summary>
        private void DisplayLoanDecision(PredictionEngine<InputData, ModelOutput> predictionEngine, InputData application)
        {
            try
            {
                Console.WriteLine("\n" + new string('═', 50));
                Console.WriteLine("🤖 AI DECISION PROCESSING...");
                Console.WriteLine(new string('═', 50));

                // Add a small delay for dramatic effect
                Thread.Sleep(1000);

                var prediction = predictionEngine.Predict(application);

                // Display application summary
                Console.WriteLine("\n📄 APPLICATION SUMMARY:");
                Console.WriteLine($"   👤 Applicant: {application.JobType} worker, Age {application.Age}");
                Console.WriteLine($"   💰 Monthly Income: DKK {application.MonthlyIncome:N0}");
                Console.WriteLine($"   🏠 Requested Loan: DKK {application.LoanAmount:N0}");
                Console.WriteLine($"   📅 Payback Period: {application.ReturnTime} months");
                Console.WriteLine($"   📊 Monthly Payment: ~DKK {(application.LoanAmount / application.ReturnTime):N0}");
                Console.WriteLine($"   📈 Debt-to-Income Ratio: {((application.LoanAmount / application.ReturnTime) / application.MonthlyIncome * 100):F1}%");

                // Display AI decision
                Console.WriteLine("\n🤖 AI DECISION:");
                if (prediction.Prediction)
                {
                    Console.WriteLine("   ✅ LOAN APPROVED!");
                    Console.WriteLine("   🎉 Congratulations! Your loan application has been approved!");
                }
                else
                {
                    Console.WriteLine("   ❌ LOAN REJECTED");
                    Console.WriteLine("   😔 Sorry, your loan application was not approved at this time.");
                }

                Console.WriteLine($"   🎯 Confidence Level: {prediction.Probability:P2}");

                // Provide helpful advice based on the result
                Console.WriteLine("\n💡 RECOMMENDATIONS:");
                if (prediction.Prediction)
                {
                    Console.WriteLine("   • Review loan terms carefully before signing");
                    Console.WriteLine("   • Consider setting up automatic payments");
                    Console.WriteLine("   • Keep your income stable during the loan period");
                }
                else
                {
                    if (prediction.Probability < 0.3)
                    {
                        Console.WriteLine("   • Consider improving your income-to-loan ratio");
                        Console.WriteLine("   • Try a smaller loan amount or longer payback period");
                        Console.WriteLine("   • Build up your employment history if possible");
                    }
                    else
                    {
                        Console.WriteLine("   • Your application was close to approval");
                        Console.WriteLine("   • Consider minor adjustments and reapplying");
                        Console.WriteLine("   • Try increasing your income or reducing the loan amount slightly");
                    }
                }

                Console.WriteLine(new string('═', 50));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error making prediction: {ex.Message}");
                Console.WriteLine("Please ensure the model was trained properly.");
            }
        }
    }
}