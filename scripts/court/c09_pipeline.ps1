# C09 end to end on new clips, with no annotation: only the court type is given.
#
#   1. court keypoints (C09, random-measurement model) -> homography for EVERY frame (--stride 1)
#   2. clip-camera verification (C07) -> H_ptz, frames the camera cannot explain are dropped
#   2b. completion: holes filled by following the camera, every frame polished on its lines and
#       averaged with its neighbours carried by the image motion (c09_complete_court.py)
#   3. tracking with the best configuration (configs/player.yaml: player detector + C06 court mask)
#   4. 2D map (C08): positions in metres from the C07 homographies, video broadcast + court plan
#
# Usage (repo root):
#   powershell -File scripts/court/c09_pipeline.ps1                       # the 6 test clips
#   powershell -File scripts/court/c09_pipeline.ps1 -Clips v_5ekaksddqrc_c003=FIBA
#   powershell -File scripts/court/c09_pipeline.ps1 -Clips v_xxx_c001=configs/courts/high_school_nfhs.yaml
#
# Every step is skipped when its output already exists (re-run after an interruption: it resumes).
# Tracking runs the detector live on CPU (~1 h per clip) unless outputs/detections/yolo26m_player
# already holds the clip's detections.

param(
    [string[]]$Clips = @(
        "v_00HRwkvvjtQ_c001=NCAA", "v_00HRwkvvjtQ_c005=NCAA",
        "v_5ekaksddqrc_c003=FIBA", "v_4r8QL_wglzQ_c001=FIBA",
        "v_BgwzTUxJaeU_c008=FIBA_PRE2010", "v_BgwzTUxJaeU_c012=FIBA_PRE2010"),
    [string]$Weights = "outputs/court_runs/c09_randdims/weights/best.pt",
    [string]$Run = "outputs/runs/013_c09_pipeline",
    [string]$Config = "configs/player.yaml"
)

$ErrorActionPreference = "Continue"   # tqdm writes progress to stderr: not an error
$py = ".venv/Scripts/python"
$homDir = "outputs/court_c09_pipeline"
$verDir = "outputs/court_verified/c09_pipeline"
$compDir = "outputs/court_complete/c09_pipeline"
$cache = "outputs/detections/yolo26m_player"

foreach ($item in $Clips) {
    $clip, $court = $item -split "=", 2
    Write-Host "`n=== $clip ($court) ===" -ForegroundColor Cyan

    # 1. homographies, every frame
    if (-not (Test-Path "$homDir/$clip/homographies.npz")) {
        & $py scripts/court/c09_detect_homographies.py --weights $Weights --imgsz 640 --stride 1 `
            --sequence $clip --league $court --out $homDir
    }

    # 2. clip-camera verification
    if (-not (Test-Path "$verDir/$clip/report.json")) {
        & $py scripts/court/verify_clip_camera.py --in $homDir --sequence $clip --out $verDir
    }
    $verdict = (Get-Content "$verDir/$clip/report.json" -Raw | ConvertFrom-Json).verdict
    if ($verdict -ne "ok") {
        Write-Host "C07 verdict '$verdict': no trusted court for $clip, no map (tracking still runs)" -ForegroundColor Yellow
    } elseif (-not (Test-Path "$compDir/$clip/homographies.npz")) {
        # 2b. completion and polish (~0.3 s per frame)
        & $py scripts/court/c09_complete_court.py --in $verDir --sequence $clip --out $compDir
    }

    # 3. tracking (cached detections when available)
    if (-not (Test-Path "$Run/results/$clip.txt")) {
        $det = @()
        if (Test-Path "$cache/$clip.txt") { $det = @("detector.type=cached", "detector.cache_dir=$cache") }
        & $py -m hooptrack.cli track --config $Config --set "data.only=[$clip]" "output.dir=$Run" @det --eval
    }

    # 4. 2D map from the completed homographies
    if ($verdict -eq "ok" -and -not (Test-Path "$Run/topdown/$clip.mp4")) {
        & $py -m hooptrack.cli map --config $Config --run $Run `
            --set "data.only=[$clip]" "output.dir=$Run" "topdown.dir=$compDir" --video
    }
}

Write-Host "`nDone. Maps: $Run/topdown/<clip>.mp4 · stats: $Run/topdown/<clip>_stats.csv" -ForegroundColor Green
