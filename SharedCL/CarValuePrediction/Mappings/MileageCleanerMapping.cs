using Microsoft.ML.Transforms;
using SharedCL.CarValuePrediction.Models;

namespace SharedCL.CarValuePrediction.Mappings
{
    [CustomMappingFactoryAttribute("MileageCleaner")]
    public class MileageCleanerMapping : CustomMappingFactory<CarInfoInput, MileageClean>
    {
        public override Action<CarInfoInput, MileageClean> GetMapping()
        {
            return (input, output) =>
            {
                var mileageStr = input.MileageKm?.Replace(" km", "").Trim();
                output.Mileage = float.TryParse(mileageStr, out var mileage) ? mileage : 0f;
            };
        }
    }
}
