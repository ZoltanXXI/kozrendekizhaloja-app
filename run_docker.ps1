Param(
    [string]$OpenAIKey = ""
)

Write-Host "== Közrendek Ízhálója — Docker helper =="

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker nem található a PATH-on. Telepítsd a Dockert és próbáld újra."
    exit 1
}

Write-Host "Building Docker image 'kozerendek-izhaloja:latest'..."
docker build -t kozerendek-izhaloja:latest .

$envArg = @()
if ($OpenAIKey -ne "") {
    Write-Host "Using provided OPENAI_API_KEY (from parameter)."
    $envArg = @("-e", "OPENAI_API_KEY=$OpenAIKey")
} else {
    Write-Warning "No OPENAI_API_KEY provided as parameter. You can set it interactively or the container may fail when calling OpenAI."
}

Write-Host "Running container on port 8501 (host) -> 8501 (container)..."
# Note: map current directory into /app if you want live file changes visible
docker run --rm -p 8501:8501 $envArg kozerendek-izhaloja:latest

Write-Host "Container stopped."