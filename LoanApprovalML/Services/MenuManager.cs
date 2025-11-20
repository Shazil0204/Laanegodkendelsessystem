using LoanApprovalML.DataModels;
using LoanApprovalML.Services;
using Microsoft.ML;

namespace LoanApprovalML.Services
{
    /// <summary>
    /// This class manages the main menu and user interaction flow.
    /// It's like the reception desk at a bank - it directs users to the right services.
    /// </summary>
    public class MenuManager
    {
        private readonly MLContext _mlContext;
        private readonly Trainer _trainer;
        private readonly LoanApplicationProcessor _loanProcessor;
        private readonly Diagram _diagram;

        public MenuManager()
        {
            _mlContext = new MLContext();
            _trainer = new Trainer();
            _loanProcessor = new LoanApplicationProcessor(_mlContext);
            _diagram = new Diagram();
        }

        /// <summary>
        /// Shows the main menu and handles user choice
        /// </summary>
        public void ShowMainMenu()
        {
            Console.WriteLine("\n╔══════════════════════════════════════════╗");
            Console.WriteLine("║          🏦 LOAN APPROVAL SYSTEM         ║");
            Console.WriteLine("╠══════════════════════════════════════════╣");
            Console.WriteLine("║  1. 🎓 Train AI Model                    ║");
            Console.WriteLine("║  2. 🧪 Test Loan Application             ║");
            Console.WriteLine("║  3. 📊 Create Visualization Diagram      ║");
            Console.WriteLine("║  4. 🚪 Exit Program                      ║");
            Console.WriteLine("╚══════════════════════════════════════════╝");
            Console.Write("Enter your choice (1-4): ");
        }

        /// <summary>
        /// Gets a valid menu choice from the user with error handling
        /// </summary>
        public int GetUserChoice()
        {
            while (true)
            {
                try
                {
                    string? input = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(input))
                    {
                        Console.Write("⚠️  Please enter a number (1-4): ");
                        continue;
                    }

                    int choice = int.Parse(input);
                    if (choice >= 1 && choice <= 4)
                    {
                        return choice;
                    }
                    else
                    {
                        Console.Write("⚠️  Please enter a number between 1 and 4: ");
                    }
                }
                catch (FormatException)
                {
                    Console.Write("⚠️  Invalid input! Please enter a number (1-4): ");
                }
                catch (OverflowException)
                {
                    Console.Write("⚠️  Number too large! Please enter 1, 2, 3, or 4: ");
                }
            }
        }

        /// <summary>
        /// Trains a new AI model using the training data
        /// </summary>
        public void TrainModel()
        {
            Console.WriteLine("\n🎓 Training AI Model...");
            Console.WriteLine("═══════════════════════════════════════");
            
            try
            {
                if (!File.Exists("data.csv"))
                {
                    Console.WriteLine("❌ Error: data.csv file not found!");
                    Console.WriteLine("   Please make sure the training data file is in the same folder as the program.");
                    return;
                }

                Console.WriteLine("📚 Loading training data from data.csv...");
                _trainer.Train("data.csv");
                Console.WriteLine("✅ Model training completed successfully!");
                Console.WriteLine("💾 Model saved as 'model.zip' and ready for testing.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error during training: {ex.Message}");
                Console.WriteLine("Please check your data.csv file and try again.");
            }
        }

        /// <summary>
        /// Tests the trained model with user loan application data
        /// </summary>
        public void TestLoanApplication()
        {
            Console.WriteLine("\n🧪 Testing Loan Application");
            Console.WriteLine("═══════════════════════════════════════");

            try
            {
                if (!File.Exists("model.zip"))
                {
                    Console.WriteLine("❌ Error: No trained model found!");
                    Console.WriteLine("   Please train the model first using option 1.");
                    return;
                }

                Console.WriteLine("📖 Loading trained AI model...");
                _loanProcessor.ProcessLoanApplication();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error during testing: {ex.Message}");
                Console.WriteLine("Please train the model first or check if model.zip exists.");
            }
        }

        /// <summary>
        /// Creates a visualization diagram of loan approvals
        /// </summary>
        public void CreateVisualization()
        {
            Console.WriteLine("\n📊 Creating Visualization Diagram");
            Console.WriteLine("═══════════════════════════════════════");

            try
            {
                if (!File.Exists("model.zip"))
                {
                    Console.WriteLine("❌ Error: No trained model found!");
                    Console.WriteLine("   Please train the model first using option 1.");
                    return;
                }

                if (!File.Exists("data.csv"))
                {
                    Console.WriteLine("❌ Error: data.csv file not found!");
                    Console.WriteLine("   Please make sure the data file is available for visualization.");
                    return;
                }

                Console.WriteLine("🎨 Creating visualization...");
                _diagram.DrawApprovalDiagram();
                Console.WriteLine("✅ Visualization created successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error creating visualization: {ex.Message}");
                Console.WriteLine("Please check if both model.zip and data.csv exist.");
            }
        }

        /// <summary>
        /// Asks if the user wants to continue using the program
        /// </summary>
        public bool AskToContinue()
        {
            Console.WriteLine("\n" + new string('─', 50));
            Console.Write("Would you like to perform another operation? (y/n): ");
            
            while (true)
            {
                try
                {
                    string? input = Console.ReadLine()?.Trim().ToLower();
                    if (string.IsNullOrEmpty(input))
                    {
                        Console.Write("Please enter 'y' for yes or 'n' for no: ");
                        continue;
                    }

                    if (input == "y" || input == "yes")
                    {
                        return true;
                    }
                    else if (input == "n" || input == "no")
                    {
                        return false;
                    }
                    else
                    {
                        Console.Write("Please enter 'y' for yes or 'n' for no: ");
                    }
                }
                catch (Exception)
                {
                    Console.Write("Please enter 'y' for yes or 'n' for no: ");
                }
            }
        }
    }
}