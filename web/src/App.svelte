<script>
  import { onMount } from 'svelte';
  import Wizard from './lib/Wizard.svelte';
  import { del, get, url } from './lib/api.js';

  // ─── Reactive state ───
  let currentTime = $state('');
  let activeTab = $state('live');

  // ─── Live data, from /api/dashboard ───
  // The same variables the mockup declared; the values come off the pipeline
  // now instead of Math.random(). The markup below is Mridhula's.
  let d = $state(null);
  let prev = $state(null);
  let view = $state('loading');          // loading | wizard | dashboard
  let runs = $state([]);
  let runId = $state(null);
  let error = $state(null);

  const POLL_MS = 2000;
  const LIVE_POLL_MS = 1000;      // a run in flight is worth watching closely

  async function refresh() {
    try {
      const next = await get(`/api/dashboard${runId ? `?run_id=${runId}` : ''}`);
      prev = d;
      d = next;
      runId = d.run_id ?? runId;
      error = null;
    } catch (e) {
      error = e.message;
    }
  }

  async function loadRuns() {
    try {
      runs = await get('/api/runs');
    } catch (e) {
      runs = [];
    }
  }

  async function onDone(id) {
    runId = id;
    await Promise.all([loadRuns(), refresh()]);
    view = 'dashboard';
    activeTab = 'live';
  }

  async function dropRun() {
    if (!runId) return;
    await del(`/api/runs/${runId}`).catch((e) => (error = e.message));
    runId = null;
    d = prev = null;
    await loadRuns();
    view = runs.length ? 'dashboard' : 'wizard';
    if (view === 'dashboard') refresh();
  }

  onMount(() => {
    const clockInterval = setInterval(() => {
      currentTime = new Date().toLocaleTimeString('en-IN', { hour12: false });
    }, 1000);
    currentTime = new Date().toLocaleTimeString('en-IN', { hour12: false });

    loadRuns().then(async () => {
      if (!runs.length) return (view = 'wizard');
      await refresh();
      view = d?.run_id ? 'dashboard' : 'wizard';
    });

    let dataInterval = setInterval(tick, POLL_MS);
    let period = POLL_MS;
    function tick() {
      if (view === 'dashboard') refresh();
      const want = running ? LIVE_POLL_MS : POLL_MS;
      if (want !== period) {
        period = want;
        clearInterval(dataInterval);
        dataInterval = setInterval(tick, period);
      }
    }
    return () => { clearInterval(clockInterval); clearInterval(dataInterval); };
  });

  // ─── The tiles ───
  let occupancy = $derived(d?.occupancy ?? 0);
  let entriesTotal = $derived(d?.entries ?? 0);
  let exitsTotal = $derived(d?.exits ?? 0);
  let avgDwell = $derived(Math.round((d?.avg_dwell_s ?? 0) / 6) / 10);
  let conversionRate = $derived(d?.conversion_rate ?? 0);
  let fps = $derived(d?.fps ?? 0);
  let plLatency = $derived(d?.system?.latency_ms ?? 0);
  let cpuPct = $derived(d?.system?.cpu_pct ?? 0);
  let memoryMb = $derived(d?.system?.memory_mb ?? 0);
  let streams = $derived(d?.system?.streams ?? 0);
  let uptime = $derived(d?.system?.uptime ?? '—');
  let storeId = $derived(d?.store_id ?? '—');
  let backendName = $derived(d?.backend ?? 'idle');   // before the first run
  let lostRevenue = $derived(Math.round(d?.lost_revenue ?? 0));
  let targetWait = $derived(d?.target_wait_s ?? 180);
  let counters = $derived(d?.counters ?? 2);
  let running = $derived(d?.state === 'running');
  let liveStream = $state('overhead');
  let clips = $derived(d?.clips ?? []);

  let footfallSpark = $derived(d?.footfall_spark ?? new Array(12).fill(0));
  let hourlyFootfall = $derived((d?.footfall_hourly ?? []).map((h) => ({ h: String(h.hour), v: h.entries })));
  let zoneDwell = $derived((d?.zones ?? []).map((z) => ({
    name: z.name ?? '—', dwell: Math.round(z.median_dwell_s ?? 0), count: z.visits ?? 0,
  })));
  let queue = $derived(d?.queue ?? []);
  let shelfFills = $derived(d?.shelf ?? []);
  let alerts = $derived((d?.alerts ?? []).map((a) => ({
    severity: a.severity, rule: a.rule, message: a.message, time: clockOf(a.t),
  })));
  let heatmapData = $derived(d?.heatmap ?? new Array(192).fill(0));
  let funnel = $derived(d?.funnel ?? []);
  let zoneLabels = $derived(d?.zone_labels ?? []);
  let stockouts = $derived((d?.stockouts ?? []).map((s) => ({
    facing: s.facing, sku: s.sku, status: s.status,
    duration: s.status === 'active' ? fmtDuration(s.duration_s) : '—',
    revenue: `₹${Math.round(s.lost)}`,
  })));

  // Status for top metric cards: neutral (empty string) by default,
  // switches to amber or red only when attention is needed.
  let occupancyStatus = $derived(
    occupancy >= 22 ? 'amber' : ''
  );
  let footfallStatus = $derived(
    netOccupancy >= 22 ? 'amber' : ''
  );
  let dwellStatus = $derived(
    avgDwell >= 6.8 ? 'red' : avgDwell >= 5.5 ? 'amber' : ''
  );
  let conversionStatus = $derived(
    conversionRate < 9.5 ? 'red' : conversionRate < 11.0 ? 'amber' : ''
  );

  // ─── Derived values ───
  let maxFootfall = $derived(Math.max(...footfallSpark, 1));
  let maxHourly = $derived(Math.max(...hourlyFootfall.map(h => h.v), 1));
  let netOccupancy = $derived(entriesTotal - exitsTotal);
  let peakHour = $derived(hourlyFootfall.reduce((a, b) => (b.v > (a?.v ?? -1) ? b : a), null));
  let avgHour = $derived(hourlyFootfall.length
    ? (hourlyFootfall.reduce((s, h) => s + h.v, 0) / hourlyFootfall.length).toFixed(1) : '0.0');
  let maxQueue = $derived(Math.max(...queue.map(q => q.count), 4));
  // Real movement since the last poll, so the arrows mean something.
  let dwellDelta = $derived(prev ? avgDwell - Math.round((prev.avg_dwell_s ?? 0) / 6) / 10 : 0);
  let conversionDelta = $derived(prev ? conversionRate - (prev.conversion_rate ?? 0) : 0);

  function heatColor(value) {
    if (value < 0.1) return '#f1f5f9';
    if (value < 0.25) return '#bae6fd';
    if (value < 0.5) return '#38bdf8';
    if (value < 0.75) return '#f59e0b';
    return '#ef4444';
  }
  function formatWait(seconds) {
    const m = Math.floor((seconds || 0) / 60);
    const s = Math.round((seconds || 0) % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  function fmtDuration(s) {
    s = Math.max(0, Math.round(s || 0));
    return s >= 3600
      ? `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`
      : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
  }

  function clockOf(t) {
    return t ? new Date(t * 1000).toLocaleTimeString('en-IN', { hour12: false }) : '—';
  }
</script>

<!-- ═══════════════ HEADER ═══════════════ -->
<header class="header">
  <div class="header-left">
    <div class="header-logo">RA</div>
    <div>
      <div class="header-title">Retail Analytics</div>
      <div class="header-subtitle">{backendName} · Store {storeId}</div>
    </div>
  </div>
  <div class="header-right">
    <div class="nav-tabs">
      <button class="nav-tab" class:active={activeTab === 'live'} onclick={() => activeTab = 'live'}>Live</button>
      <button class="nav-tab" class:active={activeTab === 'history'} onclick={() => activeTab = 'history'}>History</button>
      <button class="nav-tab" class:active={activeTab === 'demo'} onclick={() => activeTab = 'demo'}>Demo</button>
      <button class="nav-tab" class:active={view === 'wizard'} onclick={() => view = 'wizard'}>New run</button>
    </div>
    {#if runs.length > 1}
      <select class="run-picker" bind:value={runId} onchange={refresh}>
        {#each runs as r}<option value={r.run_id}>{r.run_id}</option>{/each}
      </select>
    {/if}
    <span class="badge badge-backend">{backendName} · {plLatency}ms</span>
    <span class="badge badge-privacy">🔒 DPDP</span>
    <span class="badge badge-live"><span class="dot"></span> {running ? 'PROCESSING' : 'LIVE'}</span>
    <span class="mono" style="font-size: 13px; color: var(--text-muted);">{currentTime}</span>
  </div>
</header>

{#if error}
  <div class="page-error">{error}</div>
{/if}

{#if view === 'wizard'}
  <Wizard ondone={onDone} />
{:else if view === 'loading'}
  <p class="page-note">Loading…</p>
{:else}

<!-- ═══════════════ DASHBOARD GRID ═══════════════ -->
<main class="dashboard-grid">

  <!-- Row 1: Key metrics -->
  <div class="card" id="tile-occupancy">
    <div class="card-header">
      <span class="card-title">Occupancy</span>
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
    </div>
    <div class="stat-value {occupancyStatus}">{occupancy}</div>
    <div class="stat-label">people in store now</div>
    <div class="sparkline-bar">
      {#each footfallSpark as val, i}
        <div class="bar {occupancyStatus}" style="height: {(val / maxFootfall) * 100}%; opacity: {0.4 + (i / footfallSpark.length) * 0.6};"></div>
      {/each}
    </div>
  </div>

  <div class="card" id="tile-footfall">
    <div class="card-header">
      <span class="card-title">Footfall</span>
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    </div>
    <div class="flex-between" style="align-items: flex-end;">
      <div>
        <div class="stat-value {footfallStatus}">{entriesTotal}</div>
        <div class="stat-label">entries today</div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 28px; font-weight: 800; color: var(--text-primary); letter-spacing: -1px;">{exitsTotal}</div>
        <div class="stat-label">exits</div>
      </div>
    </div>
    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border);">
      <span class="stat-label">Net in store: </span>
      <span class="mono" style="font-weight: 700; color: {footfallStatus === 'red' ? 'var(--accent-red)' : footfallStatus === 'amber' ? 'var(--accent-amber)' : 'var(--text-primary)'};">{netOccupancy}</span>
    </div>
  </div>

  <div class="card" id="tile-dwell">
    <div class="card-header">
      <span class="card-title">Avg Dwell Time</span>
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
    </div>
    <div class="stat-value {dwellStatus}">{avgDwell}<span style="font-size: 20px; color: var(--text-muted); font-weight: 500;">min</span></div>
    <div class="stat-label">across all zones {#if dwellDelta}<span class="stat-delta {dwellDelta > 0 ? 'up' : 'down'}">{dwellDelta > 0 ? '▲' : '▼'} {Math.abs(dwellDelta).toFixed(1)}</span>{/if}</div>
  </div>

  <div class="card" id="tile-conversion">
    <div class="card-header">
      <span class="card-title">Conversion Rate</span>
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/></svg>
    </div>
    <div class="stat-value {conversionStatus}">{conversionRate}<span style="font-size: 18px; color: var(--text-muted); font-weight: 500;">%</span></div>
    <div class="stat-label">visitors → buyers {#if conversionDelta}<span class="stat-delta {conversionDelta > 0 ? 'up' : 'down'}">{conversionDelta > 0 ? '▲' : '▼'} {Math.abs(conversionDelta).toFixed(1)}</span>{/if}</div>
  </div>

  <!-- Row 2: Zone dwell + Queue + Alerts -->
  <div class="card span-2" id="panel-zones">
    <div class="card-header">
      <span class="card-title">Zone Headcount & Dwell</span>
    </div>
    <div style="display: flex; flex-direction: column; gap: 6px;">
      {#each zoneDwell as zone}
        <div class="lane-row">
          <span class="lane-label">{zone.name}</span>
          <div class="lane-bar-track">
            <div class="lane-bar-fill" style="width: {Math.min(100, zone.count / 8 * 100)}%; background: var(--accent-blue);"></div>
          </div>
          <span class="lane-count">{zone.count}</span>
          <span class="lane-wait">{zone.dwell}s avg</span>
        </div>
      {/each}
    </div>
  </div>

  <div class="card" id="panel-queue">
    <div class="card-header">
      <span class="card-title">Queue Status</span>
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/></svg>
    </div>
    <div style="display: flex; flex-direction: column; gap: 12px;">
      {#each queue as lane}
        {@const late = lane.pred_wait_s > targetWait}
        <div>
          <div class="flex-between" style="margin-bottom: 6px;">
            <span class="lane-label">Lane {lane.lane}</span>
            <span class="lane-count" style="color: {late ? 'var(--accent-red)' : 'var(--accent-green)'};">{lane.count.toFixed(1)} ppl</span>
          </div>
          <div class="lane-bar-track">
            <div class="lane-bar-fill" style="width: {Math.min(100, lane.count / maxQueue * 100)}%; background: {late ? 'var(--accent-red)' : 'var(--accent-amber)'};"></div>
          </div>
          <div style="margin-top: 4px; font-size: 12px; color: var(--text-muted);">Est. wait: <span class="mono" style="color: {late ? 'var(--accent-red)' : 'var(--text-secondary)'};">{formatWait(lane.pred_wait_s)}</span></div>
        </div>
      {/each}
      <div style="padding-top: 8px; border-top: 1px solid var(--border); font-size: 12px; color: var(--text-muted);">
        Target: <span class="mono" style="color: var(--text-secondary);">{formatWait(targetWait)}</span> · Counters: <span class="mono" style="color: var(--text-secondary);">{counters}</span>
      </div>
    </div>
  </div>

  <div class="card row-2" id="panel-alerts">
    <div class="card-header">
      <span class="card-title">Alerts</span>
      <span style="font-size: 11px; color: var(--text-dim); font-family: var(--font-mono);">{alerts.length}</span>
    </div>
    <div class="alert-feed">
      {#each alerts as alert, i}
        <div class="alert-item" style="animation-delay: {i * 50}ms;">
          <span class="alert-severity {alert.severity}"></span>
          <span class="alert-text">{alert.message}</span>
          <span class="alert-time">{alert.time}</span>
        </div>
      {/each}
    </div>
  </div>

  <!-- Row 3: Shelf + Heatmap + Hourly chart -->
  <div class="card span-2" id="panel-shelf">
    <div class="card-header">
      <span class="card-title">Shelf Inventory (Live)</span>
      <div style="display: flex; gap: 8px; align-items: center;">
        <span style="font-size: 11px; color: var(--text-muted);">Lost revenue: </span>
        <span class="mono" style="font-size: 14px; font-weight: 700; color: var(--accent-red);">₹{lostRevenue.toLocaleString()}</span>
      </div>
    </div>
    <div class="shelf-grid">
      {#each shelfFills as cell}
        <div class="shelf-cell {cell.state}" title="{cell.id}: fill {cell.fill}/255">
          {cell.id}
        </div>
      {/each}
    </div>
    <div class="shelf-legend">
      <span><span class="dot" style="background: var(--accent-green);"></span> Full</span>
      <span><span class="dot" style="background: var(--accent-amber);"></span> Partial</span>
      <span><span class="dot" style="background: #ea580c;"></span> Low</span>
      <span><span class="dot" style="background: var(--accent-red);"></span> Empty</span>
    </div>

    <!-- Stock-out table -->
    <div style="margin-top: 16px; border-top: 1px solid var(--border); padding-top: 12px;">
      <div style="font-size: 12px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px;">Active Stock-outs</div>
      <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
        <thead>
          <tr style="color: var(--text-dim); text-align: left;">
            <th style="padding: 4px 8px; font-weight: 500;">Facing</th>
            <th style="padding: 4px 8px; font-weight: 500;">SKU</th>
            <th style="padding: 4px 8px; font-weight: 500;">Duration</th>
            <th style="padding: 4px 8px; font-weight: 500;">Est. Lost</th>
            <th style="padding: 4px 8px; font-weight: 500;">Status</th>
          </tr>
        </thead>
        <tbody>
          {#each stockouts as row}
            <tr style="border-top: 1px solid var(--border);">
              <td class="mono" style="padding: 6px 8px; color: var(--text-primary); font-weight: 600;">{row.facing}</td>
              <td style="padding: 6px 8px; color: var(--text-secondary);">{row.sku}</td>
              <td class="mono" style="padding: 6px 8px; color: {row.status === 'active' ? 'var(--accent-red)' : 'var(--text-dim)'};">{row.duration}</td>
              <td class="mono" style="padding: 6px 8px; color: var(--accent-amber);">{row.revenue}</td>
              <td style="padding: 6px 8px;">
                {#if row.status === 'active'}
                  <span style="color: var(--accent-red); font-weight: 600; font-size: 11px; text-transform: uppercase;">● Active</span>
                {:else}
                  <span style="color: var(--accent-green); font-size: 11px; text-transform: uppercase;">✓ Resolved</span>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Heatmap card (already in row with shelf, alerts is row-2 above) -->
  <div class="card" id="panel-heatmap">
    <div class="card-header">
      <span class="card-title">Floor Heatmap</span>
    </div>
    <div class="heatmap-container">
      <div class="heatmap-grid">
        {#each heatmapData as heat}
          <div class="heatmap-cell" style="background: {heatColor(heat)};"></div>
        {/each}
      </div>
      <div class="heatmap-zones">
        {#each zoneLabels as z}
          <span class="heatmap-zone-label" style="left: {z.left}%; top: {z.top}%;">{z.name}</span>
        {/each}
      </div>
    </div>
    <div style="margin-top: 8px; display: flex; justify-content: space-between; font-size: 10px; color: var(--text-dim);">
      <span>Low traffic</span>
      <div style="flex: 1; margin: 0 8px; height: 6px; display: flex; align-self: center;">
        <div style="flex: 1; background: #bae6fd;"></div>
        <div style="flex: 1; background: #38bdf8;"></div>
        <div style="flex: 1; background: #f59e0b;"></div>
        <div style="flex: 1; background: #ef4444;"></div>
      </div>
      <span>High traffic</span>
    </div>
  </div>

  <!-- Hourly footfall chart -->
  <div class="card span-2" id="panel-hourly">
    <div class="card-header">
      <span class="card-title">Footfall by Hour</span>
      <span style="font-size: 11px; color: var(--text-dim);">Today</span>
    </div>
    <div class="chart-bars">
      {#each hourlyFootfall as h}
        <div class="bar-group">
          <div class="bar" style="height: {(h.v / maxHourly) * 100}%; background: var(--accent-blue);"></div>
          <span class="bar-label">{h.h}</span>
        </div>
      {/each}
    </div>
    <div style="margin-top: 8px; display: flex; justify-content: space-between; font-size: 11px; color: var(--text-dim);">
      <span>Peak: <span class="mono" style="color: var(--accent-blue); font-weight: 600;">{peakHour ? `${peakHour.h}:00 (${peakHour.v})` : '—'}</span></span>
      <span>Avg: <span class="mono" style="color: var(--text-secondary);">{avgHour}</span></span>
    </div>
  </div>

  <!-- Conversion funnel -->
  <div class="card span-2" id="panel-funnel">
    <div class="card-header">
      <span class="card-title">Conversion Funnel</span>
    </div>
    <div class="funnel">
      {#each funnel as step, i}
        <div class="funnel-step">
          <span class="funnel-label">{step.label}</span>
          <div class="funnel-bar-track">
            <div class="funnel-bar-fill" style="width: {step.pct}%; background: {
              i === 0 ? 'var(--accent-blue)' :
              i === 1 ? 'var(--accent-cyan)' :
              i === 2 ? 'var(--accent-amber)' :
              'var(--accent-green)'
            };">
              {step.value}
            </div>
          </div>
          <span class="funnel-pct" style="color: {
            i === 0 ? 'var(--accent-blue)' :
            i === 1 ? 'var(--accent-cyan)' :
            i === 2 ? 'var(--accent-amber)' :
            'var(--accent-green)'
          };">{step.pct}%</span>
        </div>
      {/each}
    </div>
  </div>

  <!-- System status tile -->
  <div class="card" id="panel-system">
    <div class="card-header">
      <span class="card-title">System</span>
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M20 15h2M9 2v2M9 20v2M2 9h2M20 9h2"/></svg>
    </div>
    <div style="display: flex; flex-direction: column; gap: 10px; font-size: 13px;">
      <div class="flex-between">
        <span class="text-muted">FPS</span>
        <span class="mono" style="font-weight: 600; color: var(--accent-green);">{fps}</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">PL Latency</span>
        <span class="mono" style="font-weight: 600; color: var(--accent-cyan);">{plLatency}ms</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">CPU (PS)</span>
        <span class="mono" style="font-weight: 600; color: var(--accent-green);">{cpuPct}%</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">Memory</span>
        <span class="mono" style="font-weight: 600; color: var(--text-secondary);">{memoryMb} MB</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">Streams</span>
        <span class="mono" style="font-weight: 600; color: var(--text-secondary);">{streams} active</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">Backend</span>
        <span style="font-size: 11px; padding: 2px 8px; background: #dcfce7; color: #15803d; font-weight: 600; border: 1px solid #bbf7d0;">{backendName}</span>
      </div>
      <div style="margin-top: 4px; padding-top: 8px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-dim);">
        Store: {storeId} · Uptime: {uptime} · <button class="text-btn" onclick={dropRun}>delete run</button>
      </div>
    </div>
  </div>

  <div class="card span-2" id="panel-live">
    <div class="card-header">
      <span class="card-title">Live View</span>
      <div style="display: flex; gap: 8px; align-items: center;">
        <button class="nav-tab" class:active={liveStream === 'overhead'}
                onclick={() => liveStream = 'overhead'}>Overhead</button>
        <button class="nav-tab" class:active={liveStream === 'shelf'}
                onclick={() => liveStream = 'shelf'}>Shelf</button>
      </div>
    </div>
    {#if running}
      <img class="live-frame" alt="frame being processed"
           src={url(`/api/runs/${runId}/live.mjpg?stream=${liveStream}`)} />
      <div style="margin-top: 8px; font-size: 11px; color: var(--text-dim);">
        {d?.processed ?? 0}{d?.total ? ` / ${d.total}` : ''} frames · {d?.fps ?? 0} fps ·
        boxes are detections, never identities. No frame is written to disk.
      </div>
    {:else if clips.includes(liveStream)}
      <!-- svelte-ignore a11y_media_has_caption -->
      <video class="live-frame" autoplay loop muted playsinline
             src={url(`/api/runs/${runId}/video?stream=${liveStream}`)}></video>
      <div style="margin-top: 8px; font-size: 11px; color: var(--text-dim);">
        Run finished — replaying the {liveStream} clip. Detections are drawn live
        while a run is in flight.
      </div>
    {:else}
      <div class="live-idle">Idle — start a run to watch the pipeline work.</div>
    {/if}
  </div>

  <div class="card" id="panel-privacy">
    <div class="card-header">
      <span class="card-title">Privacy & DPDP</span>
      <span style="font-size: 16px;">🔒</span>
    </div>
    <div style="display: flex; flex-direction: column; gap: 8px; font-size: 12px; color: var(--text-secondary); line-height: 1.6;">
      <div style="display: flex; align-items: center; gap: 6px;">
        <span style="color: var(--accent-green);">✓</span> No frames stored
      </div>
      <div style="display: flex; align-items: center; gap: 6px;">
        <span style="color: var(--accent-green);">✓</span> No biometrics / face data
      </div>
      <div style="display: flex; align-items: center; gap: 6px;">
        <span style="color: var(--accent-green);">✓</span> No persistent track IDs
      </div>
      <div style="display: flex; align-items: center; gap: 6px;">
        <span style="color: var(--accent-green);">✓</span> Aggregate counts only
      </div>
      <div style="display: flex; align-items: center; gap: 6px;">
        <span style="color: var(--accent-green);">✓</span> 7-day raw retention
      </div>
      <div style="display: flex; align-items: center; gap: 6px;">
        <span style="color: var(--accent-green);">✓</span> Fully offline capable
      </div>
      <div style="margin-top: 4px; padding-top: 8px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-dim);">
        DPDP Act 2023 compliant
      </div>
    </div>
  </div>
</main>
{/if}

<style>
  /* Only the two controls the mockup had no place for. Everything else is
     app.css, exactly as it comes off the branch. */
  .run-picker {
    background: var(--bg-elevated); border: 1px solid var(--border);
    color: var(--text-secondary); padding: 5px 8px;
    font-family: var(--font-mono); font-size: 11px;
  }
  .page-error {
    margin: 16px 24px; padding: 10px 14px; font-size: 13px;
    background: #fee2e2; color: var(--accent-red); border: 1px solid #fecaca;
  }
  .page-note { margin: 40px; color: var(--text-muted); }
  .live-frame { width: 100%; display: block; background: #0f172a;
                aspect-ratio: 4 / 3; object-fit: contain; }
  .live-idle {
    aspect-ratio: 4 / 3; display: grid; place-items: center;
    background: var(--bg-elevated); color: var(--text-dim); font-size: 13px;
  }
  .text-btn {
    background: none; border: none; color: var(--text-dim); cursor: pointer;
    font-size: 11px; text-decoration: underline; padding: 0;
  }
</style>
