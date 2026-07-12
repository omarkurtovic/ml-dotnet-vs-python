using CSharpModelTrainerApi.Enums;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CSharpModelTrainerApi.Migrations
{
    /// <inheritdoc />
    public partial class AddModelStatus : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "ModelStatus",
                table: "LCModels",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "TotalEpochs",
                table: "LCModels",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.Sql(
                $"UPDATE LCModels SET ModelStatus = {(int)ModelStatus.Trained}");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ModelStatus",
                table: "LCModels");

            migrationBuilder.DropColumn(
                name: "TotalEpochs",
                table: "LCModels");
        }
    }
}
