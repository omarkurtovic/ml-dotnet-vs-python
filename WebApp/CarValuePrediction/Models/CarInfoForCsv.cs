using CsvHelper.Configuration.Attributes;
using Microsoft.ML.Data;

namespace WebApp.CarValuePrediction.Models
{
    public class CarInfoForCsv
    {
        public float Price { get; set; }

        public string Manufacturer { get; set; } = null!;

        public string Model { get; set; } = null!;

        [Name("Prod. year")]
        public float ProdYear { get; set; }

        public string Category { get; set; } = null!;

        [Name("Leather interior")]
        public string LeatherInterior { get; set; } = null!;

        [Name("Fuel type")]
        public string FuelType { get; set; } = null!;

        [Name("Engine volume")]
        public string EngineVolume { get; set; } = null!;

        public string Mileage { get; set; } = null!;

        public float Cylinders { get; set; }

        [Name("Gear box type")]
        public string GearBoxType { get; set; } = null!;

        [Name("Drive wheels")]
        public string DriveWheels { get; set; } = null!;

        public string Doors { get; set; } = null!;

        public string Wheel { get; set; } = null!;

        public string Color { get; set; } = null!;

        public float Airbags { get; set; }


    }    
}
