using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CSharpModelTrainerApi.Migrations
{
    /// <inheritdoc />
    public partial class ChangedTrainingTimeToSeconds : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "TrainingTime",
                table: "LungCancerModels");

            migrationBuilder.AddColumn<int>(
                name: "TrainingTimeInSeconds",
                table: "LungCancerModels",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "TrainingTimeInSeconds",
                table: "LungCancerModels");

            migrationBuilder.AddColumn<TimeSpan>(
                name: "TrainingTime",
                table: "LungCancerModels",
                type: "TEXT",
                nullable: false,
                defaultValue: new TimeSpan(0, 0, 0, 0, 0));
        }
    }
}
