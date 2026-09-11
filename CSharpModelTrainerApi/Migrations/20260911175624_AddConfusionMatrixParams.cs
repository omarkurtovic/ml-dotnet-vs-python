using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CSharpModelTrainerApi.Migrations
{
    /// <inheritdoc />
    public partial class AddConfusionMatrixParams : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "TrueBenignPredBenig",
                table: "LCEpochData",
                type: "INTEGER",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "TrueBenignPredMalig",
                table: "LCEpochData",
                type: "INTEGER",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "TrueBenignPredNorm",
                table: "LCEpochData",
                type: "INTEGER",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "TrueMaligPredBenig",
                table: "LCEpochData",
                type: "INTEGER",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "TrueMaligPredMalig",
                table: "LCEpochData",
                type: "INTEGER",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "TrueMaligPredNorm",
                table: "LCEpochData",
                type: "INTEGER",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "TrueNormPredBenig",
                table: "LCEpochData",
                type: "INTEGER",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "TrueNormPredMalig",
                table: "LCEpochData",
                type: "INTEGER",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "TrueNormPredNorm",
                table: "LCEpochData",
                type: "INTEGER",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "TrueBenignPredBenig",
                table: "LCEpochData");

            migrationBuilder.DropColumn(
                name: "TrueBenignPredMalig",
                table: "LCEpochData");

            migrationBuilder.DropColumn(
                name: "TrueBenignPredNorm",
                table: "LCEpochData");

            migrationBuilder.DropColumn(
                name: "TrueMaligPredBenig",
                table: "LCEpochData");

            migrationBuilder.DropColumn(
                name: "TrueMaligPredMalig",
                table: "LCEpochData");

            migrationBuilder.DropColumn(
                name: "TrueMaligPredNorm",
                table: "LCEpochData");

            migrationBuilder.DropColumn(
                name: "TrueNormPredBenig",
                table: "LCEpochData");

            migrationBuilder.DropColumn(
                name: "TrueNormPredMalig",
                table: "LCEpochData");

            migrationBuilder.DropColumn(
                name: "TrueNormPredNorm",
                table: "LCEpochData");
        }
    }
}
