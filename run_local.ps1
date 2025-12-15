Param(
    [string]$OpenAIKey = ""
)

Write-Host "== Közrendek Ízhálója — Local venv runner =="

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv .venv
}

Write-Host "Installing requirements into .venv..."
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if ($OpenAIKey -ne "") {
    Write-Host "Using provided OPENAI_API_KEY (from parameter)."
    $env:OPENAI_API_KEY = $OpenAIKey
} else {
    Write-Warning "No OPENAI_API_KEY provided as parameter. Ensure you set it in the environment before running."
}

Write-Host "Launching Streamlit (http://localhost:8501) ..."
& .\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
