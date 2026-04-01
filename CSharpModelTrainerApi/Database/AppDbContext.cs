using Microsoft.EntityFrameworkCore;
using SharedCL.LungCancerPrediction.Models;
using SharedCL.SentimentAnalysis.Models;

namespace CSharpModelTrainerApi.Database
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
        }
        public virtual DbSet<SentimentAnalysisModel> SentimentAnalysisModels { get; set; }
        public virtual DbSet<LungCancerModel> LungCancerModels { get; set; }
        public virtual DbSet<LungCancerModelEpochData> LungCancerModelEpochData { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            modelBuilder.Entity<LungCancerModel>()
                .HasMany(l => l.EpochData)
                .WithOne(e => e.LungCancerModel)
                .HasForeignKey(e => e.LungCancerModelId);
        }
    }
}
