using CSharpModelTrainerApi.LungCancerPrediction.Models;
using Microsoft.EntityFrameworkCore;

namespace CSharpModelTrainerApi.Database
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
        }
        public virtual DbSet<LCModel> LCModels { get; set; }
        public virtual DbSet<LCEpochData> LCEpochData { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            modelBuilder.Entity<LCModel>()
                .HasMany(l => l.EpochData)
                .WithOne(e => e.LCModel)
                .HasForeignKey(e => e.LCModelId);

            modelBuilder.Entity<LCEpochData>()
                .HasMany(l => l.LCPredictions)
                .WithOne(p => p.LCEpochData)
                .HasForeignKey(p => p.LCEpochDataId);
        }
    }
}
