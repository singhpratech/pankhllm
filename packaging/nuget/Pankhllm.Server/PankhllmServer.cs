using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Pankhllm;

/// Starts the bundled router binary as a child process, for apps that host it in-process.
public static class PankhllmServer
{
    public static Process Start(string configPath, int port = 4000, string host = "127.0.0.1")
    {
        var rid = (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "win" : RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "osx" : "linux")
                  + "-" + RuntimeInformation.OSArchitecture.ToString().ToLowerInvariant();
        var exe = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "pankhllm.exe" : "pankhllm";
        var path = Environment.GetEnvironmentVariable("PANKHLLM_BINARY")
                   ?? Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", exe);
        if (!File.Exists(path)) throw new FileNotFoundException($"pankhllm binary not found for {rid}", path);
        var psi = new ProcessStartInfo(path, $"--config \"{configPath}\" serve --host {host} --port {port}") { UseShellExecute = false };
        return Process.Start(psi) ?? throw new InvalidOperationException("could not start pankhllm");
    }
}
