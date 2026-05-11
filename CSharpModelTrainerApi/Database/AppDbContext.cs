using CSharpModelTrainerApi.LungCancerPrediction.Models;
using CSharpModelTrainerApi.SentimentAnalysis.Models;
using Microsoft.EntityFrameworkCore;

namespace CSharpModelTrainerApi.Database
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
        }
        public virtual DbSet<SAModel> SentimentAnalysisModels { get; set; }
        public virtual DbSet<LCModel> LungCancerModels { get; set; }
        public virtual DbSet<LCEpochData> LungCancerModelEpochData { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            modelBuilder.Entity<LCModel>()
                .HasMany(l => l.EpochData)
                .WithOne(e => e.LCModel)
                .HasForeignKey(e => e.LCModelId);
        }
    }
}
