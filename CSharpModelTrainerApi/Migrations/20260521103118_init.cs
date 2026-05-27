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
                name: "LCModels",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    Language = table.Column<int>(type: "INTEGER", nullable: false),
                    TrainingTimeInSeconds = table.Column<int>(type: "INTEGER", nullable: false),
                    HardwareInfo = table.Column<string>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_LCModels", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "LCEpochData",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    LCModelId = table.Column<int>(type: "INTEGER", nullable: false),
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
                    table.PrimaryKey("PK_LCEpochData", x => x.Id);
                    table.ForeignKey(
                        name: "FK_LCEpochData_LCModels_LCModelId",
                        column: x => x.LCModelId,
                        principalTable: "LCModels",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_LCEpochData_LCModelId",
                table: "LCEpochData",
                column: "LCModelId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "LCEpochData");

            migrationBuilder.DropTable(
                name: "LCModels");
        }
    }
}
