using CSharpModelTrainerApi.LungCancerPrediction.Datasets;
using static TorchSharp.torch;

namespace CSharpTests
{
    public class LCTrainDatasetTests
    {

        [Test]
        public void GetClassWeights_ThreeClasses_ZeroLabels_ReturnsCorrectWeights()
        {
            var dataset = new LungCancerTrainDataset([], [], ["Class1", "Class2", "Class3"]);

            var result = dataset.GetClassWeights();

            using (Assert.EnterMultipleScope())
            {
                Assert.That(result.Count, Is.EqualTo(3));
                Assert.That(result[0], Is.EqualTo(0));
                Assert.That(result[1], Is.EqualTo(0));
                Assert.That(result[2], Is.EqualTo(0));
            }
        }

        [Test]
        public void GetClassWeights_ZeroClasses_SomeLabels_ReturnsCorrectWeights()
        {
            var dataset = new LungCancerTrainDataset([], [0, 1, 2], []);

            var result = dataset.GetClassWeights();

            using (Assert.EnterMultipleScope())
            {
                Assert.That(result.Count, Is.EqualTo(0));
            }
        }

        [Test]
        public void GetClassWeights_MissingClass_AvoidsDivideByZero()
        {
            var dataset = new LungCancerTrainDataset([], [0, 0, 2, 2], ["Class0", "Class1", "Class2"]);

            var result = dataset.GetClassWeights();

            using (Assert.EnterMultipleScope())
            {
                Assert.That(result.Count, Is.EqualTo(3));
                Assert.That(result[0], Is.EqualTo(4f / 6f));
                Assert.That(result[1], Is.EqualTo(0f));
                Assert.That(result[2], Is.EqualTo(4f / 6f));
            }
        }

        [Test]
        public void GetClassWeights_UnbalancedClasses_ReturnsCorrectWeights()
        {
            var dataset = new LungCancerTrainDataset([], [0, 0, 1, 2, 2], ["Class1", "Class2", "Class3"]);
            var result = dataset.GetClassWeights();
            using (Assert.EnterMultipleScope())
            {
                Assert.That(result.Count, Is.EqualTo(3));
                Assert.That(result[0], Is.EqualTo(5f / 6));
                Assert.That(result[1], Is.EqualTo(5f / 3));
                Assert.That(result[2], Is.EqualTo(5f / 6));
            }
        }
    }
}
