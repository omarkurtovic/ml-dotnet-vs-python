using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CSharpModelTrainerApi.Migrations
{
    /// <inheritdoc />
    public partial class init : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "LungCancerModels",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    Language = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_LungCancerModels", x => x.Id);
                });

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

            migrationBuilder.CreateTable(
                name: "LungCancerModelEpochData",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    LungCancerModelId = table.Column<int>(type: "INTEGER", nullable: false),
                    Epoch = table.Column<int>(type: "INTEGER", nullable: false),
                    TrainingLoss = table.Column<double>(type: "REAL", nullable: true),
                    TrainingAccuracy = table.Column<double>(type: "REAL", nullable: true),
                    ValidationAccuracy = table.Column<double>(type: "REAL", nullable: true),
                    ValidationLoss = table.Column<double>(type: "REAL", nullable: true),
                    BenignPrecision = table.Column<double>(type: "REAL", nullable: true),
                    BenignRecall = table.Column<double>(type: "REAL", nullable: true),
                    BenignF1Score = table.Column<double>(type: "REAL", nullable: true),
                    MalignantPrecision = table.Column<double>(type: "REAL", nullable: true),
                    MalignantRecall = table.Column<double>(type: "REAL", nullable: true),
                    MalignantF1Score = table.Column<double>(type: "REAL", nullable: true),
                    NormalPrecision = table.Column<double>(type: "REAL", nullable: true),
                    NormalRecall = table.Column<double>(type: "REAL", nullable: true),
                    NormalF1Score = table.Column<double>(type: "REAL", nullable: true),
                    MacroPrecision = table.Column<double>(type: "REAL", nullable: true),
                    MacroRecall = table.Column<double>(type: "REAL", nullable: true),
                    MacroF1Score = table.Column<double>(type: "REAL", nullable: true),
                    WeightedPrecision = table.Column<double>(type: "REAL", nullable: true),
                    WeightedRecall = table.Column<double>(type: "REAL", nullable: true),
                    WeightedF1Score = table.Column<double>(type: "REAL", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_LungCancerModelEpochData", x => x.Id);
                    table.ForeignKey(
                        name: "FK_LungCancerModelEpochData_LungCancerModels_LungCancerModelId",
                        column: x => x.LungCancerModelId,
                        principalTable: "LungCancerModels",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_LungCancerModelEpochData_LungCancerModelId",
                table: "LungCancerModelEpochData",
                column: "LungCancerModelId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "LungCancerModelEpochData");

            migrationBuilder.DropTable(
                name: "SentimentAnalysisModels");

            migrationBuilder.DropTable(
                name: "LungCancerModels");
        }
    }
}
