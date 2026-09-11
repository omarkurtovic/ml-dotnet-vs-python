using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CSharpModelTrainerApi.Migrations
{
    /// <inheritdoc />
    public partial class UpdateConfusionMatrixParams : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "TrueNormPredNorm",
                table: "LCEpochData",
                newName: "TrueNormalPredNormal");

            migrationBuilder.RenameColumn(
                name: "TrueNormPredMalig",
                table: "LCEpochData",
                newName: "TrueNormalPredMalignant");

            migrationBuilder.RenameColumn(
                name: "TrueNormPredBenig",
                table: "LCEpochData",
                newName: "TrueNormalPredBenign");

            migrationBuilder.RenameColumn(
                name: "TrueMaligPredNorm",
                table: "LCEpochData",
                newName: "TrueMalignantPredNormal");

            migrationBuilder.RenameColumn(
                name: "TrueMaligPredMalig",
                table: "LCEpochData",
                newName: "TrueMalignantPredMalignant");

            migrationBuilder.RenameColumn(
                name: "TrueMaligPredBenig",
                table: "LCEpochData",
                newName: "TrueMalignantPredBenign");

            migrationBuilder.RenameColumn(
                name: "TrueBenignPredNorm",
                table: "LCEpochData",
                newName: "TrueBenignPredNormal");

            migrationBuilder.RenameColumn(
                name: "TrueBenignPredMalig",
                table: "LCEpochData",
                newName: "TrueBenignPredMalignant");

            migrationBuilder.RenameColumn(
                name: "TrueBenignPredBenig",
                table: "LCEpochData",
                newName: "TrueBenignPredBenign");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "TrueNormalPredNormal",
                table: "LCEpochData",
                newName: "TrueNormPredNorm");

            migrationBuilder.RenameColumn(
                name: "TrueNormalPredMalignant",
                table: "LCEpochData",
                newName: "TrueNormPredMalig");

            migrationBuilder.RenameColumn(
                name: "TrueNormalPredBenign",
                table: "LCEpochData",
                newName: "TrueNormPredBenig");

            migrationBuilder.RenameColumn(
                name: "TrueMalignantPredNormal",
                table: "LCEpochData",
                newName: "TrueMaligPredNorm");

            migrationBuilder.RenameColumn(
                name: "TrueMalignantPredMalignant",
                table: "LCEpochData",
                newName: "TrueMaligPredMalig");

            migrationBuilder.RenameColumn(
                name: "TrueMalignantPredBenign",
                table: "LCEpochData",
                newName: "TrueMaligPredBenig");

            migrationBuilder.RenameColumn(
                name: "TrueBenignPredNormal",
                table: "LCEpochData",
                newName: "TrueBenignPredNorm");

            migrationBuilder.RenameColumn(
                name: "TrueBenignPredMalignant",
                table: "LCEpochData",
                newName: "TrueBenignPredMalig");

            migrationBuilder.RenameColumn(
                name: "TrueBenignPredBenign",
                table: "LCEpochData",
                newName: "TrueBenignPredBenig");
        }
    }
}
