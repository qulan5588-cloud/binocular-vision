param(
    [switch]$Watch,
    [int]$IntervalMs = 800
)

function Get-Chain {
    param([string]$InstanceId)
    $chain = [System.Collections.ArrayList]@()
    $cur = (Get-PnpDeviceProperty -InstanceId $InstanceId -KeyName 'DEVPKEY_Device_Parent' -ErrorAction SilentlyContinue).Data
    $lv = 0
    while ($cur -and $lv -lt 8) {
        $d = Get-PnpDevice -InstanceId $cur -ErrorAction SilentlyContinue
        if (-not $d) { break }
        $loc = (Get-PnpDeviceProperty -InstanceId $cur -KeyName 'DEVPKEY_Device_LocationInfo' -ErrorAction SilentlyContinue).Data
        $svc = (Get-PnpDeviceProperty -InstanceId $cur -KeyName 'DEVPKEY_Device_Service' -ErrorAction SilentlyContinue).Data
        [void]$chain.Add([pscustomobject]@{
            Level   = $lv
            Name    = $d.FriendlyName
            Id      = $cur
            Loc     = $loc
            Service = $svc
        })
        $cur = (Get-PnpDeviceProperty -InstanceId $cur -KeyName 'DEVPKEY_Device_Parent' -ErrorAction SilentlyContinue).Data
        $lv++
    }
    return $chain
}

function Get-Report {
    $all  = @(Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue)
    $devs = @($all | Where-Object { $_.InstanceId -match '2BC5' })
    $bad  = @($all | Where-Object { $_.Class -eq 'Unknown' })

    $r = [pscustomobject]@{
        Time       = (Get-Date).ToString('HH:mm:ss')
        Online     = ($devs.Count -gt 0)
        Depth      = $false
        Color      = $false
        HubCount   = 0
        ExtraHub   = ''
        ChainLines = [System.Collections.ArrayList]@()
        UnknownDev = $bad.Count
        Verdict    = ''
    }

    if (-not $r.Online) {
        $r.Verdict = 'NOT_DETECTED'
        return $r
    }

    $depth = $devs | Where-Object { $_.InstanceId -like '*PID_060F*' -and $_.InstanceId -notlike '*MI_*' } | Select-Object -First 1
    $color = $devs | Where-Object { $_.InstanceId -like '*PID_050F*' -and $_.InstanceId -notlike '*MI_*' } | Select-Object -First 1
    if ($depth) { $r.Depth = $true }
    if ($color) { $r.Color = $true }

    $base = $depth
    if (-not $base) { $base = $devs | Select-Object -First 1 }

    if ($base) {
        $chain = Get-Chain -InstanceId $base.InstanceId
        foreach ($c in $chain) {
            $isHub  = ($c.Service -match '^usbhub')
            $isRoot = ($c.Id -like '*ROOT_HUB*')
            $isCam  = ($c.Id -like '*VID_05E3&PID_0608*')
            $tag = ''
            if ($isHub -and -not $isRoot) {
                $r.HubCount++
                if ($isCam) { $tag = ' [inside camera]' }
                else {
                    $tag = ' [EXTRA]'
                    if ($r.ExtraHub -eq '') { $r.ExtraHub = $c.Name }
                    else { $r.ExtraHub = $r.ExtraHub + ', ' + $c.Name }
                }
            }
            if ($isRoot) { $tag = ' [root]' }
            [void]$r.ChainLines.Add(('  L{0}  {1,-34} {2}{3}' -f $c.Level, $c.Name, $c.Loc, $tag))
        }
    }

    if ($r.HubCount -ge 2) { $r.Verdict = 'EXTRA_HUB' }
    elseif ($r.Online)     { $r.Verdict = 'OK' }
    return $r
}

function Show-Report {
    param($r)
    Write-Host ''
    Write-Host ('[{0}] ===== Camera USB check =====' -f $r.Time) -ForegroundColor Cyan
    if (-not $r.Online) {
        Write-Host '  Orbbec device ....... NOT FOUND' -ForegroundColor Red
        Write-Host ('  Unknown devices ..... {0}' -f $r.UnknownDev) -ForegroundColor Yellow
        Write-Host '  => no enumeration at all: cable carries no data, or not seated, or camera hung' -ForegroundColor Red
        return
    }
    Write-Host '  Orbbec device ....... FOUND' -ForegroundColor Green
    $d = 'MISSING'; if ($r.Depth) { $d = 'OK' }
    $c = 'MISSING'; if ($r.Color) { $c = 'OK' }
    Write-Host ('  Depth 2BC5:060F ..... {0}' -f $d)
    Write-Host ('  Color 2BC5:050F ..... {0}' -f $c)
    Write-Host ('  Hub layers .......... {0}' -f $r.HubCount)
    Write-Host '  Chain (device -> host):'
    $r.ChainLines | ForEach-Object { Write-Host $_ }
    if ($r.Verdict -eq 'OK') {
        Write-Host '  => GOOD: only the in-camera hub, ready to capture' -ForegroundColor Green
    } else {
        Write-Host ('  => BAD: extra hub in path -> {0}' -f $r.ExtraHub) -ForegroundColor Red
    }
}

if ($Watch) {
    Write-Host 'Watching USB. Plug / unplug the camera now. Ctrl+C to stop.' -ForegroundColor Yellow
    $lastKey = ''
    while ($true) {
        $r = Get-Report
        $key = '{0}|{1}|{2}|{3}' -f $r.Online, $r.Depth, $r.Color, $r.Verdict
        if ($key -ne $lastKey) {
            Show-Report -r $r
            $lastKey = $key
        }
        Start-Sleep -Milliseconds $IntervalMs
    }
} else {
    Show-Report -r (Get-Report)
}
