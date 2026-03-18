using Azure.Storage;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;

namespace CSharpModelTrainerApi.Shared
{
    public class BlobService(BlobServiceClient blobServiceClient)
    {
        private const string ContainerName = "datacontainer";
        private const string BlobRootFolder = "data";
        private const string ModelBlobRootFolder = "models";

        private static string? GetRepoRoot()
        {
            var repoRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", ".."));
            return Directory.Exists(Path.Combine(repoRoot, "models")) ? repoRoot : null;
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

        public async Task<string> EnsureModelDownloadedAsync(string blobSubPath)
        {
            var repoRoot = GetRepoRoot();
            if (repoRoot != null)
            {
                var repoPath = Path.Combine(repoRoot, "models", blobSubPath);
                if (File.Exists(repoPath))
                    return repoPath;
            }

            var cachePath = Path.Combine(Path.GetTempPath(), "mlapp-models", blobSubPath);

            if (File.Exists(cachePath))
                return cachePath;

            Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);

            var blobName = $"{ModelBlobRootFolder}/{blobSubPath}";
            var container = blobServiceClient.GetBlobContainerClient(ContainerName);
            var blobClient = container.GetBlobClient(blobName);

            await blobClient.DownloadToAsync(cachePath);
            return cachePath;
        }

        public async Task UploadModelAsync(string localFilePath, string blobSubPath)
        {
            var blobName = $"{ModelBlobRootFolder}/{blobSubPath}";
            var container = blobServiceClient.GetBlobContainerClient(ContainerName);
            await container.CreateIfNotExistsAsync();

            var blobClient = container.GetBlobClient(blobName);
            await blobClient.UploadAsync(localFilePath, overwrite: true);
        }
    }
}
