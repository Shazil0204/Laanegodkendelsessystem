using Microsoft.ML.Data;

namespace LoanApprovalML.DataModels
{
    /// <summary>
    /// This class represents what our AI tells us after we ask it to make a prediction.
    /// Think of it as the AI's answer when we ask "Should we approve this loan?"
    /// </summary>
    public class ModelOutput
    {
        // This is the AI's final decision: true = "Yes, approve the loan", false = "No, reject it"
        // ML.NET calls this "PredictedLabel" internally, but we can use any name we want
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        // This is how confident the AI is in its decision (0.0 to 1.0)
        // If this is 0.8, it means "I'm 80% sure this should be approved"
        // If it's 0.3, it means "I'm only 30% sure, so probably reject it"
        public float Probability { get; set; }
        
        // This is the raw "score" the AI calculated before converting to probability
        // It's a more technical number that most people don't need to worry about
        // (The comment "don't ask" suggests even the original programmer finds this confusing!)
        public float Score { get; set; } // don't ask
    }
}
