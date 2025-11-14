using Microsoft.ML;
using LoanApprovalML.DataModels;

namespace LoanApprovalML.Services
{
    public class DataLoader
    {
        private readonly MLContext _mlContext;

        public DataLoader(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        public IDataView LoadData(string path)
        {
            return _mlContext.Data.LoadFromTextFile<InputData>(
                path,
                hasHeader: true,
                separatorChar: ',');
        }
    }
}
