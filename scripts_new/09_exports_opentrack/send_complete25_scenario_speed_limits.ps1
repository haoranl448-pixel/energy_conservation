param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(16, 19, 21, 35, 40, 44, 49, 50, 53, 57, 60, 66)]
    [int]$TripNo,

    [Parameter(Mandatory = $true)]
    [ValidateSet("history", "standard", "energy-first", "dwell5")]
    [string]$Scenario,

    [switch]$Send
)

$ErrorActionPreference = "Stop"
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$tripTag = "trip{0:D3}" -f $TripNo
$tripRoot = if ($TripNo -eq 53) {
    Join-Path $projectRoot "output\schedule\batch_trip_reports_complete25_replacement53\$tripTag"
} else {
    Join-Path $projectRoot "output\schedule\batch_trip_reports_complete25_new12\$tripTag"
}
$routeMap = Join-Path $projectRoot "output\opentrack_route_map_newline\priority_dp_trip011_route_map.csv"
$outputDir = Join-Path $projectRoot "output\opentrack_speed_limits_complete25"
$python = "C:\Users\bit11\AppData\Local\Programs\Python\Python311\python.exe"

if (-not (Test-Path -LiteralPath $routeMap -PathType Leaf)) {
    throw "Route map not found: $routeMap"
}
if (-not (Test-Path -LiteralPath $tripRoot -PathType Container)) {
    throw "Trip directory not found: $tripRoot"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = "python"
}
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

if ($Scenario -eq "history") {
    $script = Join-Path $PSScriptRoot "make_opentrack_history_speed_limits_from_lookup.py"
    $lookup = Join-Path $projectRoot "output\schedule\batch_trip_reports_trip1_125\historical_speed_lookup.csv"
    $traceability = Join-Path $projectRoot "data\data_processed_step2_v3_all_curve_quality_traceability\trip_traceability_manifest_v1.csv"
    $trainId = "priority_history_$tripTag"
    $outputCsv = Join-Path $outputDir "speed_limits_history_$tripTag.csv"
    $arguments = @(
        $script,
        "--speed-lookup-csv", $lookup,
        "--route-map", $routeMap,
        "--output-csv", $outputCsv,
        "--trip-no", $TripNo,
        "--traceability-manifest", $traceability,
        "--method", "cruise-avg",
        "--train-id", $trainId,
        "--range-mode", "single-route",
        "--sleep", "1"
    )
} else {
    $script = Join-Path $PSScriptRoot "make_opentrack_speed_limits.py"
    $menu = Join-Path $tripRoot ("ato_class_energy_menu{0}_new_v3.csv" -f $TripNo)
    $scenarioConfig = @{
        "standard" = @("Final_Planning_Comparison.csv", "priority_dp_$tripTag", "priority_dp")
        "energy-first" = @("Final_Planning_Comparison_Energy_First.csv", "energy_first_dp_$tripTag", "energy_first")
        "dwell5" = @("Final_Planning_Comparison_Real_Priority_Dwell_5pct.csv", "real_priority_dwell_5pct_dp_$tripTag", "real_priority_dwell_5pct")
    }
    $config = $scenarioConfig[$Scenario]
    $plan = Join-Path $tripRoot $config[0]
    $trainId = $config[1]
    $outputCsv = Join-Path $outputDir ("speed_limits_{0}_{1}.csv" -f $config[2], $tripTag)
    foreach ($requiredPath in @($menu, $plan)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Required input not found: $requiredPath"
        }
    }
    $arguments = @(
        $script,
        "--plan-csv", $plan,
        "--energy-menu", $menu,
        "--route-map", $routeMap,
        "--output-csv", $outputCsv,
        "--train-id", $trainId,
        "--range-mode", "single-route",
        "--speed-method", "cruise-avg",
        "--opentrack-speed-unit", "kmh",
        "--sleep", "1",
        "--send-retries", "8",
        "--retry-delay", "1"
    )
}

if ($Send) {
    $arguments += "--send"
    Write-Host "Sending $Scenario limits for $tripTag to $trainId"
} else {
    Write-Host "Preview only: generating $outputCsv"
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Speed-limit command failed with exit code $LASTEXITCODE"
}
