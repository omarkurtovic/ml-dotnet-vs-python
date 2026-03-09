using Microsoft.EntityFrameworkCore;
using SharedCL.SentimentAnalysis.Models;

namespace CSharpModelTrainerApi.Database
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
        }
        public virtual DbSet<SentimentAnalysisModel> SentimentAnalysisModels { get; set; }
    }
}
