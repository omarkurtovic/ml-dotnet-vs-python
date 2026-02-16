using CsvHelper.Configuration.Attributes;

namespace WebApp.CarValuePrediction.Models
{
    public class CarInfoDto
    {
        public string Manufacturer { get; set; } = null!;

        public string Model { get; set; } = null!;
        public int ProdYear { get; set; }

        public string Category { get; set; } = null!;
        public bool LeatherInterior { get; set; }

        public string FuelType { get; set; } = null!;

        public string EngineVolume { get; set; } = null!;

        public int Mileage { get; set; }

        public int Cylinders { get; set; }

        public string GearBoxType { get; set; } = null!;

        public string DriveWheels { get; set; } = null!;

        public int Doors { get; set; }

        public bool RightWheel { get; set; }

        public string Color { get; set; } = null!;

        public int Airbags { get; set; }
    }
}
