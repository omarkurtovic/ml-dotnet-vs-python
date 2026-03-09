using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CSharpModelTrainerApi.Migrations
{
    /// <inheritdoc />
    public partial class Init : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "SentimentAnalysisModels",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    Language = table.Column<int>(type: "INTEGER", nullable: false),
                    TrainerAlgorithm = table.Column<int>(type: "INTEGER", nullable: false),
                    TrainingAccuracy = table.Column<double>(type: "REAL", nullable: true),
                    TrainingF1Score = table.Column<double>(type: "REAL", nullable: true),
                    TrainingAreaUnderRocCurve = table.Column<double>(type: "REAL", nullable: true),
                    TrainingPositivePrecision = table.Column<double>(type: "REAL", nullable: true),
                    TrainingPositiveRecall = table.Column<double>(type: "REAL", nullable: true),
                    TestingAccuracy = table.Column<double>(type: "REAL", nullable: true),
                    TestingF1Score = table.Column<double>(type: "REAL", nullable: true),
                    TestingAreaUnderRocCurve = table.Column<double>(type: "REAL", nullable: true),
                    TestingPositivePrecision = table.Column<double>(type: "REAL", nullable: true),
                    TestingPositiveRecall = table.Column<double>(type: "REAL", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_SentimentAnalysisModels", x => x.Id);
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "SentimentAnalysisModels");
        }
    }
}
