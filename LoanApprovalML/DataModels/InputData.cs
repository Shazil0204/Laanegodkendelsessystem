using Microsoft.ML.Data;

namespace LoanApprovalML.DataModels
{
    public class InputData
    {
        [LoadColumn(0)]
        public float MonthlyIncome { get; set; }

        [LoadColumn(1)]
        public float LoanAmount { get; set; }

        [LoadColumn(2)]
        public float ReturnTime { get; set; }

        [LoadColumn(3)]
        public float Age { get; set; }

        [LoadColumn(4)]
        public string? JobType { get; set; }

        [LoadColumn(5)]
        public bool IsApproved { get; set; } // Label
    }
}
