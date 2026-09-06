<script>
  import { onMount } from 'svelte';
  import Wizard from './lib/Wizard.svelte';
  import { clockOf, del, fmtDuration, fmtWait, get } from './lib/api.js';

  // ─── State: one document from /api/dashboard, polled ───
  let view = $state('loading');        // loading | wizard | dashboard
  let runs = $state([]);
  let runId = $state(null);
  let d = $state(null);
  let error = $state(null);
  let currentTime = $state('');

  const POLL_MS = 2000;

  async function refresh() {
    try {
      d = await get(`/api/dashboard${runId ? `?run_id=${runId}` : ''}`);
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
  }

  async function dropRun() {
    if (!runId) return;
    await del(`/api/runs/${runId}`).catch((e) => (error = e.message));
    runId = null;
    d = null;
    await loadRuns();
    view = runs.length ? 'dashboard' : 'wizard';
    if (view === 'dashboard') refresh();
  }

  onMount(() => {
    const clock = setInterval(
      () => (currentTime = new Date().toLocaleTimeString('en-IN', { hour12: false })), 1000);
    currentTime = new Date().toLocaleTimeString('en-IN', { hour12: false });

    loadRuns().then(async () => {
      if (!runs.length) return (view = 'wizard');
      await refresh();
      view = d?.run_id ? 'dashboard' : 'wizard';
    });

    const tick = setInterval(() => view === 'dashboard' && refresh(), POLL_MS);
    return () => { clearInterval(clock); clearInterval(tick); };
  });

  // ─── Derived: the tiles, straight off the document ───
  let occupancy = $derived(d?.occupancy ?? 0);
  let entriesTotal = $derived(d?.entries ?? 0);
  let exitsTotal = $derived(d?.exits ?? 0);
  let netOccupancy = $derived(entriesTotal - exitsTotal);
  let avgDwell = $derived(((d?.avg_dwell_s ?? 0) / 60).toFixed(1));
  let conversionRate = $derived(d?.conversion_rate ?? 0);
  let lostRevenue = $derived(Math.round(d?.lost_revenue ?? 0));
  let footfallSpark = $derived(d?.footfall_spark ?? new Array(12).fill(0));
  let hourlyFootfall = $derived((d?.footfall_hourly ?? []).map((h) => ({ h: h.hour, v: h.entries })));
  let zoneDwell = $derived((d?.zones ?? []).map((z) => ({
    name: z.name ?? '—', count: z.visits ?? 0, dwell: Math.round(z.median_dwell_s ?? 0),
  })));
  let queue = $derived(d?.queue ?? []);
  let shelfFills = $derived(d?.shelf ?? []);
  let stockouts = $derived(d?.stockouts ?? []);
  let alerts = $derived(d?.alerts ?? []);
  let heatmapData = $derived(d?.heatmap ?? new Array(192).fill(0));
  let funnel = $derived(d?.funnel ?? []);
  let running = $derived(d?.state === 'running');

  let maxFootfall = $derived(Math.max(...footfallSpark, 1));
  let maxHourly = $derived(Math.max(...hourlyFootfall.map((h) => h.v), 1));
  let peakHour = $derived(hourlyFootfall.reduce((a, b) => (b.v > (a?.v ?? -1) ? b : a), null));
  let avgHour = $derived(hourlyFootfall.length
    ? (hourlyFootfall.reduce((s, h) => s + h.v, 0) / hourlyFootfall.length).toFixed(1) : '0');
  let maxQueue = $derived(Math.max(...queue.map((q) => q.count), 4));
  let targetWait = $derived(d?.target_wait_s ?? 180);

  function heatColor(value) {
    if (value < 0.1) return 'rgba(59, 130, 246, 0.05)';
    if (value < 0.25) return `rgba(59, 130, 246, ${0.1 + value * 0.4})`;
    if (value < 0.5) return `rgba(34, 211, 238, ${0.2 + value * 0.5})`;
    if (value < 0.75) return `rgba(245, 158, 11, ${0.3 + value * 0.5})`;
    return `rgba(239, 68, 68, ${0.5 + value * 0.4})`;
  }
</script>

<!-- ═══════════════ HEADER ═══════════════ -->
<header class="header">
  <div class="header-left">
    <div class="header-logo">RA</div>
    <div>
      <div class="header-title">Retail Analytics</div>
      <div class="header-subtitle">Edge AI · {d?.store_id ?? 'no store'} · no cloud</div>
    </div>
  </div>
  <div class="header-right">
    {#if runs.length}
      <select class="run-picker" bind:value={runId} onchange={refresh}>
        {#each runs as r}
          <option value={r.run_id}>{r.run_id}</option>
        {/each}
      </select>
    {/if}
    <div class="nav-tabs">
      <button class="nav-tab" class:active={view === 'dashboard'}
              disabled={!runId} onclick={() => (view = 'dashboard')}>Dashboard</button>
      <button class="nav-tab" class:active={view === 'wizard'}
              onclick={() => (view = 'wizard')}>New run</button>
    </div>
    <span class="badge badge-backend">{d?.backend ?? 'idle'}</span>
    <span class="badge badge-privacy">🔒 DPDP</span>
    {#if running}
      <span class="badge badge-live"><span class="dot"></span> PROCESSING</span>
    {/if}
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
    <div class="stat-value cyan">{occupancy}</div>
    <div class="stat-label">people in store at the last sample</div>
    <div class="sparkline-bar">
      {#each footfallSpark as val, i}
        <div class="bar cyan" style="height: {(val / maxFootfall) * 100}%; opacity: {0.4 + (i / footfallSpark.length) * 0.6};"></div>
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
        <div class="stat-value green">{entriesTotal}</div>
        <div class="stat-label">entries</div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 28px; font-weight: 800; color: var(--accent-amber); letter-spacing: -1px;">{exitsTotal}</div>
        <div class="stat-label">exits</div>
      </div>
    </div>
    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border);">
      <span class="stat-label">Net in store: </span>
      <span class="mono" style="font-weight: 700; color: var(--accent-cyan);">{netOccupancy}</span>
    </div>
  </div>

  <div class="card" id="tile-dwell">
    <div class="card-header">
      <span class="card-title">Avg Dwell Time</span>
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
    </div>
    <div class="stat-value blue">{avgDwell}<span style="font-size: 20px; color: var(--text-muted); font-weight: 500;">min</span></div>
    <div class="stat-label">across all zones</div>
  </div>

  <div class="card" id="tile-conversion">
    <div class="card-header">
      <span class="card-title">Conversion Rate</span>
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/></svg>
    </div>
    <div class="stat-value purple">{conversionRate}<span style="font-size: 18px; color: var(--text-muted); font-weight: 500;">%</span></div>
    <div class="stat-label">visitors → buyers (POS is simulated)</div>
  </div>

  <!-- Row 2: Zone dwell + Queue + Alerts -->
  <div class="card span-2" id="panel-zones">
    <div class="card-header">
      <span class="card-title">Zone Visits & Dwell</span>
    </div>
    <div style="display: flex; flex-direction: column; gap: 6px;">
      {#each zoneDwell as zone}
        <div class="lane-row">
          <span class="lane-label">{zone.name}</span>
          <div class="lane-bar-track">
            <div class="lane-bar-fill" style="width: {Math.min(100, zone.count / 8 * 100)}%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));"></div>
          </div>
          <span class="lane-count">{zone.count}</span>
          <span class="lane-wait">{zone.dwell}s med</span>
        </div>
      {:else}
        <span class="stat-label">No zone visits yet.</span>
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
          <div style="margin-top: 4px; font-size: 12px; color: var(--text-muted);">Est. wait: <span class="mono" style="color: {late ? 'var(--accent-red)' : 'var(--text-secondary)'};">{fmtWait(lane.pred_wait_s)}</span></div>
        </div>
      {:else}
        <span class="stat-label">No queue samples yet.</span>
      {/each}
      <div style="padding-top: 8px; border-top: 1px solid var(--border); font-size: 12px; color: var(--text-muted);">
        Target: <span class="mono" style="color: var(--text-secondary);">{fmtWait(targetWait)}</span> · Counters: <span class="mono" style="color: var(--text-secondary);">{d?.counters ?? '—'}</span>
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
          <span class="alert-time">{clockOf(alert.t)}</span>
        </div>
      {:else}
        <span class="stat-label">Nothing to report.</span>
      {/each}
    </div>
  </div>

  <!-- Row 3: Shelf + Heatmap + Hourly chart -->
  <div class="card span-2" id="panel-shelf">
    <div class="card-header">
      <span class="card-title">Shelf Inventory</span>
      <div style="display: flex; gap: 8px; align-items: center;">
        <span style="font-size: 11px; color: var(--text-muted);">Lost revenue: </span>
        <span class="mono" style="font-size: 14px; font-weight: 700; color: var(--accent-red);">₹{lostRevenue.toLocaleString()}</span>
      </div>
    </div>
    <div class="shelf-grid">
      {#each shelfFills as cell}
        <div class="shelf-cell {cell.state}" title="{cell.name ?? cell.id}: fill {cell.fill}/255">
          {cell.id}
        </div>
      {:else}
        <span class="stat-label">No shelf stream in this run.</span>
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
      <div style="font-size: 12px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px;">Stock-outs</div>
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
              <td class="mono" style="padding: 6px 8px; color: {row.status === 'active' ? 'var(--accent-red)' : 'var(--text-dim)'};">{fmtDuration(row.duration_s)}</td>
              <td class="mono" style="padding: 6px 8px; color: var(--accent-amber);">₹{row.lost}</td>
              <td style="padding: 6px 8px;">
                {#if row.status === 'active'}
                  <span style="color: var(--accent-red); font-weight: 600; font-size: 11px; text-transform: uppercase;">● Active</span>
                {:else}
                  <span style="color: var(--accent-green); font-size: 11px; text-transform: uppercase;">✓ Resolved</span>
                {/if}
              </td>
            </tr>
          {:else}
            <tr><td colspan="5" style="padding: 8px; color: var(--text-dim);">Every facing stayed stocked.</td></tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Heatmap card -->
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
    </div>
    <div style="margin-top: 8px; display: flex; justify-content: space-between; font-size: 10px; color: var(--text-dim);">
      <span>Low traffic</span>
      <div style="flex: 1; margin: 0 8px; height: 6px; border-radius: 3px; background: linear-gradient(90deg, rgba(59,130,246,0.1), var(--accent-cyan), var(--accent-amber), var(--accent-red)); align-self: center;"></div>
      <span>High traffic</span>
    </div>
  </div>

  <!-- Hourly footfall chart -->
  <div class="card span-2" id="panel-hourly">
    <div class="card-header">
      <span class="card-title">Footfall by Hour</span>
      <span style="font-size: 11px; color: var(--text-dim);">Run day</span>
    </div>
    <div class="chart-bars">
      {#each hourlyFootfall as h}
        <div class="bar-group">
          <div class="bar" style="height: {(h.v / maxHourly) * 100}%; background: linear-gradient(180deg, var(--accent-blue), rgba(59,130,246,0.3));"></div>
          <span class="bar-label">{h.h}</span>
        </div>
      {:else}
        <span class="stat-label">No entries counted yet.</span>
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
            <div class="funnel-bar-fill" style="width: {Math.max(step.pct, 3)}%; background: linear-gradient(90deg, {
              i === 0 ? 'var(--accent-blue)' :
              i === 1 ? 'var(--accent-cyan)' :
              i === 2 ? 'var(--accent-amber)' :
              'var(--accent-green)'
            }, {
              i === 0 ? 'rgba(59,130,246,0.6)' :
              i === 1 ? 'rgba(34,211,238,0.6)' :
              i === 2 ? 'rgba(245,158,11,0.6)' :
              'rgba(34,197,94,0.6)'
            });">
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
        <span class="text-muted">Processing FPS</span>
        <span class="mono" style="font-weight: 600; color: var(--accent-green);">{d?.fps ?? 0}</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">Frames</span>
        <span class="mono" style="font-weight: 600; color: var(--accent-cyan);">{d?.processed ?? 0}{d?.total ? ` / ${d.total}` : ''}</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">Detector</span>
        <span style="font-size: 11px; padding: 2px 8px; border-radius: 100px; background: rgba(34,197,94,0.1); color: var(--accent-green); font-weight: 600;">{d?.backend ?? '—'}</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">State</span>
        <span class="mono" style="font-weight: 600; color: var(--text-secondary);">{d?.state ?? '—'}</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">Cloud</span>
        <span class="mono" style="font-weight: 600; color: var(--text-secondary);">not required</span>
      </div>
      <div style="margin-top: 4px; padding-top: 8px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-dim); display: flex; justify-content: space-between; align-items: center;">
        <span class="mono">{runId ?? '—'}</span>
        <button class="text-btn" onclick={dropRun}>delete run</button>
      </div>
    </div>
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
        <span style="color: var(--accent-green);">✓</span> Inference on this device
      </div>
      <div style="margin-top: 4px; padding-top: 8px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-dim);">
        DPDP Act 2023 compliant
      </div>
    </div>
  </div>
</main>
{/if}

<style>
  .run-picker {
    background: var(--bg-elevated); border: 1px solid var(--border);
    color: var(--text-secondary); border-radius: var(--radius-sm);
    padding: 5px 8px; font-family: var(--font-mono); font-size: 11px;
  }
  .page-error {
    margin: 16px 24px; padding: 10px 14px; border-radius: var(--radius-md);
    background: var(--accent-red-glow); color: var(--accent-red); font-size: 13px;
  }
  .page-note { margin: 40px; color: var(--text-muted); }
  .text-btn {
    background: none; border: none; color: var(--text-dim); cursor: pointer;
    font-size: 11px; text-decoration: underline;
  }
  .nav-tab:disabled { opacity: 0.4; cursor: default; }
</style>
