using Microsoft.ML;
using LoanApprovalML.DataModels;

namespace LoanApprovalML.Services
{
    /// <summary>
    /// This class is like a file reader specifically designed for our loan data.
    /// Its job is to take a CSV file (like an Excel spreadsheet saved as text)
    /// and convert it into something our machine learning system can understand.
    /// 
    /// Think of it as a translator between "human readable spreadsheet" and "AI readable data"
    /// </summary>
    public class DataLoader(MLContext mlContext)
    {
        // We need MLContext to do the actual file reading - it's like our ML.NET toolkit
        private readonly MLContext _mlContext = mlContext;

        /// <summary>
        /// This method opens a CSV file and reads all the loan application data from it.
        /// </summary>
        /// <param name="path">The file path to our CSV file (like "data.csv")</param>
        /// <returns>A special ML.NET data structure that contains all our loan applications</returns>
        public IDataView LoadData(string path)
        {
            // This tells ML.NET:
            // - Read from this file path
            // - The data type is InputData (our loan application structure)
            // - hasHeader: true = the first row contains column names (MonthlyIncome, LoanAmount, etc.)
            // - separatorChar: ',' = values are separated by commas (that's what CSV means!)
            return _mlContext.Data.LoadFromTextFile<InputData>(
                path,
                hasHeader: true,        // First row has column names
                separatorChar: ',');    // Comma-separated values
        }
    }
}
