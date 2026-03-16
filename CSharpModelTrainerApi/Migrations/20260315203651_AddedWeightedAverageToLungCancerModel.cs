using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CSharpModelTrainerApi.Migrations
{
    /// <inheritdoc />
    public partial class AddedWeightedAverageToLungCancerModel : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<double>(
                name: "WeightedF1Score",
                table: "LungCancerModels",
                type: "REAL",
                nullable: true);

            migrationBuilder.AddColumn<double>(
                name: "WeightedPrecision",
                table: "LungCancerModels",
                type: "REAL",
                nullable: true);

            migrationBuilder.AddColumn<double>(
                name: "WeightedRecall",
                table: "LungCancerModels",
                type: "REAL",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "WeightedF1Score",
                table: "LungCancerModels");

            migrationBuilder.DropColumn(
                name: "WeightedPrecision",
                table: "LungCancerModels");

            migrationBuilder.DropColumn(
                name: "WeightedRecall",
                table: "LungCancerModels");
        }
    }
}
