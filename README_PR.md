# SnakeGo HL Pipeline - PR Package

## How to use

1. Clone the framework repo:
   git clone https://github.com/SAST-agent/AgentBenchFramework.git
   cd AgentBenchFramework

2. Copy everything from this package on top:
   # In PowerShell, from this folder:
   Copy-Item snakego .\snakego -Recurse -Force
   Copy-Item src .\src -Recurse -Force
   Copy-Item .gitignore .\.gitignore -Force

3. Clean caches and commit:
   Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
   git checkout -b feat/snakego-hl
   git add -A
   git commit -m "feat: SnakeGo HL pipeline (official engine + adapters)"
   git push origin feat/snakego-hl

4. Create PR on GitHub.
