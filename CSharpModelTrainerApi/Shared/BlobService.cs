using Azure.Storage;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;

namespace CSharpModelTrainerApi.Shared
{
    public class BlobService(BlobServiceClient blobServiceClient)
    {
        private const string ContainerName = "datacontainer";
        private const string BlobRootFolder = "data";

        public static string GetBasePath()
        {
            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
            if (Directory.Exists(Path.Combine(repoRoot, "data")))
                return repoRoot;

            return AppDomain.CurrentDomain.BaseDirectory;
        }

        public async Task EnsureDataDownloadedAsync(string localPath, string blobFolder)
        {
            if (Directory.Exists(localPath) && Directory.GetFiles(localPath, "*", SearchOption.AllDirectories).Length > 0)
                return;

            Directory.CreateDirectory(localPath);
            var prefix = $"{BlobRootFolder}/{blobFolder}";
            var container = blobServiceClient.GetBlobContainerClient(ContainerName);

            await foreach (var blob in container.GetBlobsAsync(BlobTraits.None, BlobStates.None, prefix, CancellationToken.None))
            {
                var relativePath = blob.Name.Substring(prefix.Length).TrimStart('/');
                var localFile = Path.Combine(localPath, relativePath);
                Directory.CreateDirectory(Path.GetDirectoryName(localFile)!);

                var blobClient = container.GetBlobClient(blob.Name);
                await blobClient.DownloadToAsync(localFile);
            }
        }
    }
}
