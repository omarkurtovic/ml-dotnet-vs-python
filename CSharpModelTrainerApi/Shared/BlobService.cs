using Azure.Storage;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;

namespace CSharpModelTrainerApi.Shared
{
    public class BlobService(BlobServiceClient blobServiceClient)
    {
        private const string ContainerName = "data";

        public async Task EnsureDataDownloadedAsync(string localPath, string blobFolder)
        {
            if (Directory.Exists(localPath) && Directory.GetFiles(localPath, "*", SearchOption.AllDirectories).Length > 0)
                return;

            Directory.CreateDirectory(localPath);
            var container = blobServiceClient.GetBlobContainerClient(ContainerName);

            await foreach (var blob in container.GetBlobsAsync(BlobTraits.None, BlobStates.None, blobFolder, CancellationToken.None))
            {
                var relativePath = blob.Name.Substring(blobFolder.Length).TrimStart('/');
                var localFile = Path.Combine(localPath, relativePath);
                Directory.CreateDirectory(Path.GetDirectoryName(localFile)!);

                var blobClient = container.GetBlobClient(blob.Name);
                await blobClient.DownloadToAsync(localFile);
            }
        }
    }
}
