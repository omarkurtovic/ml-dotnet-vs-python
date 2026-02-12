using System;
using System.Collections.Generic;
using System.Text;

namespace CSharpModelTrainer.CarValuePrediction.Models
{
    public class CarInfoEncoded
    {
        public float Price { get; set; }

        public float[] Manufacturer { get; set; } = null!;

        public float[] Model { get; set; } = null!;

        public float ProdYear { get; set; }

        public float[] Category { get; set; } = null!;

        public float[] LeatherInterior { get; set; } = null!;

        public float[] FuelType { get; set; } = null!;
        public float[] EngineVolume { get; set; } = null!;
        public float Mileage { get; set; }

        public float Cylinders { get; set; }

        public float[] GearBoxType { get; set; } = null!;

        public float[] DriveWheels { get; set; } = null!;

        public float[] Doors { get; set; } = null!;

        public float[] Wheel { get; set; } = null!;

        public float[] Color { get; set; } = null!;
        public float Airbags { get; set; }
    }
}
