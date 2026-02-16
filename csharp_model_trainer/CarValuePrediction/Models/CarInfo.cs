using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace CSharpModelTrainer.CarValuePrediction.Models
{
    public class CarInfo
    {

        [LoadColumn(1)]
        public float Price { get; set; }

        [LoadColumn(3)]
        public string Manufacturer { get; set; } = null!;

        [LoadColumn(4)]
        public string Model { get; set; } = null!;

        [LoadColumn(5)]
        public float ProdYear { get; set; }

        [LoadColumn(6)]
        public string Category { get; set; } = null!;

        [LoadColumn(7)]
        public string LeatherInterior { get; set; } = null!;

        [LoadColumn(8)]
        public string FuelType { get; set; } = null!;

        [LoadColumn(9)]
        public string EngineVolume { get; set; } = null!;

        [LoadColumn(10)]
        public string MileageKm { get; set; } = null!;

        [LoadColumn(11)]
        public float Cylinders { get; set; }

        [LoadColumn(12)]
        public string GearBoxType { get; set; } = null!;

        [LoadColumn(13)]
        public string DriveWheels { get; set; } = null!;

        [LoadColumn(14)]
        public string Doors { get; set; } = null!;

        [LoadColumn(15)]
        public string Wheel { get; set; } = null!;

        [LoadColumn(16)]
        public string Color { get; set; } = null!;

        [LoadColumn(17)]
        public float Airbags { get; set; }

        public override string ToString()
        {
            return $"Price: {Price}, {Manufacturer} {Model}, Year: {ProdYear}, Cat: {Category}, Leather: {LeatherInterior}, Fuel: {FuelType}, Engine: {EngineVolume}, Mileage: {MileageKm}, Cyl: {Cylinders}, Gear: {GearBoxType}, Drive: {DriveWheels}, Doors: {Doors}, Wheel: {Wheel}, Color: {Color}, Airbags: {Airbags}";
        }
    }
}
