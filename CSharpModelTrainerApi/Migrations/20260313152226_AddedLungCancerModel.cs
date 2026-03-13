using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CSharpModelTrainerApi.Migrations
{
    /// <inheritdoc />
    public partial class AddedLungCancerModel : Migration
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
                    Language = table.Column<int>(type: "INTEGER", nullable: false),
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
                    MacroF1Score = table.Column<double>(type: "REAL", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_LungCancerModels", x => x.Id);
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "LungCancerModels");
        }
    }
}
