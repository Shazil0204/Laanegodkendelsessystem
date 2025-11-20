using LoanApprovalML.Services;

namespace LoanApprovalML
{
    /// <summary>
    /// Main program class - This is the entry point of our Loan Approval System.
    /// Think of this as the front door of our digital bank that welcomes users
    /// and directs them to the right services.
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            // Welcome the user to our loan approval system
            Console.WriteLine("🏦 Welcome to the AI-Powered Loan Approval System!");
            Console.WriteLine("This system helps evaluate loan applications using artificial intelligence.\n");

            // Create our menu manager - this handles all user interactions
            var menuManager = new MenuManager();
            bool continueRunning = true;

            // Main program loop - keeps running until user chooses to exit
            while (continueRunning)
            {
                try
                {
                    // Show the main menu and get user's choice
                    menuManager.ShowMainMenu();
                    int userChoice = menuManager.GetUserChoice();

                    // Handle the user's choice
                    switch (userChoice)
                    {
                        case 1:
                            // Train a new AI model
                            menuManager.TrainModel();
                            break;

                        case 2:
                            // Test the AI with a loan application
                            menuManager.TestLoanApplication();
                            break;

                        case 3:
                            // Create a visualization diagram
                            menuManager.CreateVisualization();
                            break;

                        case 4:
                            // Exit the program
                            continueRunning = false;
                            Console.WriteLine("\n👋 Thank you for using the Loan Approval System!");
                            Console.WriteLine("Have a great day!");
                            break;

                        default:
                            // This shouldn't happen due to validation, but just in case
                            Console.WriteLine("⚠️  Invalid option selected. Please try again.");
                            break;
                    }

                    // Ask if user wants to continue (except when exiting)
                    if (continueRunning && userChoice != 4)
                    {
                        continueRunning = menuManager.AskToContinue();
                        
                        if (!continueRunning)
                        {
                            Console.WriteLine("\n👋 Thank you for using the Loan Approval System!");
                            Console.WriteLine("Have a great day!");
                        }
                    }
                }
                catch (Exception ex)
                {
                    // Handle any unexpected errors gracefully
                    Console.WriteLine($"\n❌ An unexpected error occurred: {ex.Message}");
                    Console.WriteLine("The program will continue running. Please try again.");
                    
                    // Ask if they want to continue after an error
                    continueRunning = menuManager.AskToContinue();
                }
            }

            // Give the user a moment to read final messages before closing
            Console.WriteLine("\nPress any key to close the program...");
            Console.ReadKey();
        }
    }
}
