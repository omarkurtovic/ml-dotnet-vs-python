using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CSharpModelTrainerApi.Migrations
{
    /// <inheritdoc />
    public partial class AddedMoreTimeStats : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "DataLoadingTimeInSeconds",
                table: "LCModels",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "ValidationTimeInSeconds",
                table: "LCModels",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "DataLoadingTimeInSeconds",
                table: "LCModels");

            migrationBuilder.DropColumn(
                name: "ValidationTimeInSeconds",
                table: "LCModels");
        }
    }
}
