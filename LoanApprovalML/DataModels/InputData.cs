using Microsoft.ML.Data;

namespace LoanApprovalML.DataModels
{
    /// <summary>
    /// This class represents a single loan application - like a form someone fills out
    /// when they want to borrow money. Each property is like a field on that form.
    /// The [LoadColumn(X)] attributes tell ML.NET which column in our CSV file
    /// contains each piece of information.
    /// </summary>
    public class InputData
    {
        // Column 0: How much money does this person make each month?
        // This is super important - people with higher income are more likely to pay back loans
        [LoadColumn(0)]
        public float MonthlyIncome { get; set; }

        // Column 1: How much money do they want to borrow?
        // Obviously, asking for $1M when you make $2K/month is probably a bad idea!
        [LoadColumn(1)]
        public float LoanAmount { get; set; }

        // Column 2: How many months do they want to take to pay it back?
        // Longer time = smaller monthly payments, but more interest
        [LoadColumn(2)]
        public float ReturnTime { get; set; }

        // Column 3: How old is this person?
        // Age can matter - very young people might be risky, very old people might have trouble working
        [LoadColumn(3)]
        public float Age { get; set; }

        // Column 4: What kind of job do they have?
        // Full-time jobs are usually more stable than part-time or student jobs
        [LoadColumn(4)]
        public string JobType { get; set; } = "unemployed";  // Default to unemployed if not specified

        // Column 5: This is the ANSWER we're trying to predict!
        // Did the bank actually approve this loan application? (true = yes, false = no)
        // This is what we call the "label" - it's what we're teaching our AI to predict
        [LoadColumn(5)]
        public bool IsApproved { get; set; } // Label
    }
}
