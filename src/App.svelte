<script>
  import { onMount } from 'svelte';

  // ─── Reactive state ───
  let currentTime = $state('');
  let activeTab = $state('live');

  // ─── Fake live data (simulated) ───
  let occupancy = $state(17);
  let entriesTotal = $state(342);
  let exitsTotal = $state(298);
  let avgDwell = $state(4.2);
  let queueCount1 = $state(5);
  let queueCount2 = $state(3);
  let queueWait1 = $state(142);
  let queueWait2 = $state(88);
  let fps = $state(14.8);
  let plLatency = $state(4.8);
  let conversionRate = $state(12.4);
  let lostRevenue = $state(1260);

  // Footfall sparkline data (last 12 intervals)
  let footfallSpark = $state([12, 18, 15, 22, 28, 25, 19, 31, 27, 23, 20, 17]);

  // Hourly footfall chart
  let hourlyFootfall = $state([
    { h: '9', v: 22 }, { h: '10', v: 38 }, { h: '11', v: 52 },
    { h: '12', v: 67 }, { h: '13', v: 58 }, { h: '14', v: 45 },
    { h: '15', v: 61 }, { h: '16', v: 54 }, { h: '17', v: 42 },
    { h: '18', v: 35 },
  ]);

  // Zone dwell (seconds) 
  let zoneDwell = $state([
    { name: 'Entrance', dwell: 8, count: 5 },
    { name: 'Aisle 1', dwell: 45, count: 3 },
    { name: 'Aisle 2', dwell: 62, count: 4 },
    { name: 'Aisle 3', dwell: 28, count: 2 },
    { name: 'Checkout', dwell: 95, count: 6 },
  ]);

  // Shelf ROI fills (8x4 = 32 facings)
  let shelfFills = $state(generateShelfData());

  // Alerts
  let alerts = $state([
    { severity: 'critical', rule: 'stockout', message: 'Stock-out: MAGGI-70G at facing A3', time: '18:28:04', stream: 'shelf' },
    { severity: 'warning', rule: 'queue', message: 'Lane 1 wait exceeds 2min — open counter 2?', time: '18:27:32', stream: 'overhead' },
    { severity: 'info', rule: 'occupancy', message: 'Occupancy exceeded 15 in Zone Aisle 2', time: '18:26:11', stream: 'overhead' },
    { severity: 'warning', rule: 'planogram', message: 'Planogram violation: A5 expected LAYS-50G, observed empty', time: '18:25:47', stream: 'shelf' },
    { severity: 'critical', rule: 'stockout', message: 'Stock-out: DAIRY-MILK-38G at facing B2', time: '18:24:19', stream: 'shelf' },
    { severity: 'info', rule: 'conversion', message: 'Conversion rate dropped below 10% in last 15min', time: '18:23:05', stream: '-' },
  ]);

  // Heatmap grid (16x12)
  let heatmapData = $state(generateHeatmap());

  // Conversion funnel
  let funnel = $state([
    { label: 'Footfall', value: 342, pct: 100 },
    { label: 'Browsed', value: 218, pct: 63.7 },
    { label: 'Picked up', value: 89, pct: 26.0 },
    { label: 'Purchased', value: 42, pct: 12.3 },
  ]);

  // Stock-out table
  let stockouts = $state([
    { facing: 'A3', sku: 'MAGGI-70G', duration: '12m 04s', revenue: '₹504', status: 'active' },
    { facing: 'B2', sku: 'DAIRY-MILK-38G', duration: '8m 31s', revenue: '₹380', status: 'active' },
    { facing: 'C1', sku: 'LAYS-CLASSIC-50G', duration: '—', revenue: '₹0', status: 'resolved' },
  ]);

  function generateShelfData() {
    const ids = [];
    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 8; col++) {
        const label = String.fromCharCode(65 + row) + (col + 1);
        const fill = Math.random();
        let state;
        if (fill > 0.7) state = 'full';
        else if (fill > 0.4) state = 'partial';
        else if (fill > 0.15) state = 'low';
        else state = 'empty';
        ids.push({ id: label, fill: Math.round(fill * 255), state });
      }
    }
    // Force a couple stockouts for realism
    ids[2].state = 'empty'; ids[2].fill = 12;
    ids[9].state = 'empty'; ids[9].fill = 8;
    ids[18].state = 'low';  ids[18].fill = 45;
    return ids;
  }

  function generateHeatmap() {
    const cells = [];
    for (let y = 0; y < 12; y++) {
      for (let x = 0; x < 16; x++) {
        // Simulate hot spots near entrance (bottom-center) and aisles
        let heat = 0;
        // entrance hotspot
        const dx1 = x - 8, dy1 = y - 11;
        heat += Math.max(0, 1 - Math.sqrt(dx1*dx1 + dy1*dy1) / 5);
        // aisle 2 hotspot
        const dx2 = x - 5, dy2 = y - 5;
        heat += Math.max(0, 0.8 - Math.sqrt(dx2*dx2 + dy2*dy2) / 4);
        // checkout hotspot
        const dx3 = x - 13, dy3 = y - 3;
        heat += Math.max(0, 0.6 - Math.sqrt(dx3*dx3 + dy3*dy3) / 3);
        // Add some noise
        heat += (Math.random() - 0.5) * 0.15;
        heat = Math.max(0, Math.min(1, heat));
        cells.push(heat);
      }
    }
    return cells;
  }

  function heatColor(value) {
    if (value < 0.1) return '#f1f5f9';
    if (value < 0.25) return '#bae6fd';
    if (value < 0.5) return '#38bdf8';
    if (value < 0.75) return '#f59e0b';
    return '#ef4444';
  }

  function formatWait(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  // ─── Live update simulation ───
  onMount(() => {
    // Clock
    const clockInterval = setInterval(() => {
      currentTime = new Date().toLocaleTimeString('en-IN', { hour12: false });
    }, 1000);
    currentTime = new Date().toLocaleTimeString('en-IN', { hour12: false });

    // Simulate data changes every 2 seconds
    const dataInterval = setInterval(() => {
      // Jitter occupancy
      occupancy = Math.max(3, Math.min(30, occupancy + Math.floor(Math.random() * 5) - 2));
      
      // Sometimes add an entry or exit
      if (Math.random() > 0.5) entriesTotal += 1;
      if (Math.random() > 0.6) exitsTotal += 1;

      // Jitter dwell
      avgDwell = Math.round((avgDwell + (Math.random() - 0.5) * 0.6) * 10) / 10;
      avgDwell = Math.max(1.5, Math.min(8, avgDwell));

      // Queue jitter
      queueCount1 = Math.max(0, Math.min(10, queueCount1 + Math.floor(Math.random() * 3) - 1));
      queueCount2 = Math.max(0, Math.min(8, queueCount2 + Math.floor(Math.random() * 3) - 1));
      queueWait1 = Math.max(30, Math.min(240, queueWait1 + Math.floor(Math.random() * 21) - 10));
      queueWait2 = Math.max(20, Math.min(180, queueWait2 + Math.floor(Math.random() * 21) - 10));

      // Sparkline shift
      footfallSpark = [...footfallSpark.slice(1), Math.max(5, Math.floor(Math.random() * 35))];

      // FPS jitter
      fps = Math.round((14.5 + Math.random() * 1) * 10) / 10;

      // Lost revenue tick
      lostRevenue = lostRevenue + Math.floor(Math.random() * 28);

      // Shelf jitter (occasionally)
      if (Math.random() > 0.8) {
        const idx = Math.floor(Math.random() * shelfFills.length);
        let newFill = Math.max(0, Math.min(255, shelfFills[idx].fill + Math.floor(Math.random() * 40) - 20));
        let newState;
        if (newFill > 180) newState = 'full';
        else if (newFill > 100) newState = 'partial';
        else if (newFill > 40) newState = 'low';
        else newState = 'empty';
        shelfFills[idx] = { ...shelfFills[idx], fill: newFill, state: newState };
        shelfFills = [...shelfFills]; // trigger reactivity
      }

      // Conversion jitter 
      conversionRate = Math.round((conversionRate + (Math.random() - 0.5) * 0.8) * 10) / 10;
      conversionRate = Math.max(8, Math.min(18, conversionRate));

    }, 2000);

    // Add alerts occasionally
    const alertInterval = setInterval(() => {
      const alertPool = [
        { severity: 'info', rule: 'occupancy', message: `Zone Aisle ${1 + Math.floor(Math.random() * 3)} occupancy: ${5 + Math.floor(Math.random() * 10)}`, stream: 'overhead' },
        { severity: 'warning', rule: 'queue', message: `Lane 1 predicted wait: ${formatWait(90 + Math.floor(Math.random() * 120))}`, stream: 'overhead' },
        { severity: 'info', rule: 'tripwire', message: `Entry detected (tripwire door), total: ${entriesTotal}`, stream: 'overhead' },
        { severity: 'warning', rule: 'shelf_fill', message: `Low fill on facing ${String.fromCharCode(65 + Math.floor(Math.random() * 4))}${1 + Math.floor(Math.random() * 8)}: ${20 + Math.floor(Math.random() * 30)}%`, stream: 'shelf' },
      ];
      const newAlert = { ...alertPool[Math.floor(Math.random() * alertPool.length)], time: new Date().toLocaleTimeString('en-IN', { hour12: false }) };
      alerts = [newAlert, ...alerts.slice(0, 9)];
    }, 5000);

    return () => {
      clearInterval(clockInterval);
      clearInterval(dataInterval);
      clearInterval(alertInterval);
    };
  });

  // ─── Derived values ───
  let maxFootfall = $derived(Math.max(...footfallSpark, 1));
  let maxHourly = $derived(Math.max(...hourlyFootfall.map(h => h.v), 1));
  let netOccupancy = $derived(entriesTotal - exitsTotal);

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
</script>

<!-- ═══════════════ HEADER ═══════════════ -->
<header class="header">
  <div class="header-left">
    <div class="header-logo">RA</div>
    <div>
      <div class="header-title">Retail Analytics</div>
      <div class="header-subtitle">PYNQ-Z2 · Store demo-01</div>
    </div>
  </div>
  <div class="header-right">
    <div class="nav-tabs">
      <button class="nav-tab" class:active={activeTab === 'live'} onclick={() => activeTab = 'live'}>Live</button>
      <button class="nav-tab" class:active={activeTab === 'history'} onclick={() => activeTab = 'history'}>History</button>
      <button class="nav-tab" class:active={activeTab === 'demo'} onclick={() => activeTab = 'demo'}>Demo</button>
    </div>
    <span class="badge badge-backend">PL (FPGA) · {plLatency}ms</span>
    <span class="badge badge-privacy">🔒 DPDP</span>
    <span class="badge badge-live"><span class="dot"></span> LIVE</span>
    <span class="mono" style="font-size: 13px; color: var(--text-muted);">{currentTime}</span>
  </div>
</header>

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
    <div class="stat-label">across all zones <span class="stat-delta up">▲ 0.3</span></div>
  </div>

  <div class="card" id="tile-conversion">
    <div class="card-header">
      <span class="card-title">Conversion Rate</span>
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/></svg>
    </div>
    <div class="stat-value {conversionStatus}">{conversionRate}<span style="font-size: 18px; color: var(--text-muted); font-weight: 500;">%</span></div>
    <div class="stat-label">visitors → buyers <span class="stat-delta down">▼ 1.2</span></div>
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
      <div>
        <div class="flex-between" style="margin-bottom: 6px;">
          <span class="lane-label">Lane 1</span>
          <span class="lane-count" style="color: {queueWait1 > 120 ? 'var(--accent-red)' : 'var(--accent-green)'};">{queueCount1} ppl</span>
        </div>
        <div class="lane-bar-track">
          <div class="lane-bar-fill" style="width: {queueCount1 / 10 * 100}%; background: {queueWait1 > 120 ? 'var(--accent-red)' : 'var(--accent-amber)'};"></div>
        </div>
        <div style="margin-top: 4px; font-size: 12px; color: var(--text-muted);">Est. wait: <span class="mono" style="color: {queueWait1 > 120 ? 'var(--accent-red)' : 'var(--text-secondary)'};">{formatWait(queueWait1)}</span></div>
      </div>
      <div>
        <div class="flex-between" style="margin-bottom: 6px;">
          <span class="lane-label">Lane 2</span>
          <span class="lane-count" style="color: var(--accent-green);">{queueCount2} ppl</span>
        </div>
        <div class="lane-bar-track">
          <div class="lane-bar-fill" style="width: {queueCount2 / 10 * 100}%; background: var(--accent-green);"></div>
        </div>
        <div style="margin-top: 4px; font-size: 12px; color: var(--text-muted);">Est. wait: <span class="mono" style="color: var(--text-secondary);">{formatWait(queueWait2)}</span></div>
      </div>
      <div style="padding-top: 8px; border-top: 1px solid var(--border); font-size: 12px; color: var(--text-muted);">
        Target: <span class="mono" style="color: var(--text-secondary);">3:00</span> · Counters: <span class="mono" style="color: var(--text-secondary);">2</span>
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
          <span class="alert-text">{@html alert.message}</span>
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
        <span class="heatmap-zone-label" style="left: 35%; bottom: 4%;">Entrance</span>
        <span class="heatmap-zone-label" style="left: 15%; top: 32%;">Aisle 2</span>
        <span class="heatmap-zone-label" style="right: 8%; top: 15%;">Checkout</span>
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
      <span>Peak: <span class="mono" style="color: var(--accent-blue); font-weight: 600;">12:00 (67)</span></span>
      <span>Avg: <span class="mono" style="color: var(--text-secondary);">42.2</span></span>
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
        <span class="mono" style="font-weight: 600; color: var(--accent-green);">12%</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">Memory</span>
        <span class="mono" style="font-weight: 600; color: var(--text-secondary);">187 MB</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">Streams</span>
        <span class="mono" style="font-weight: 600; color: var(--text-secondary);">2 active</span>
      </div>
      <div class="flex-between">
        <span class="text-muted">Backend</span>
        <span style="font-size: 11px; padding: 2px 8px; background: #dcfce7; color: #15803d; font-weight: 600; border: 1px solid #bbf7d0;">Reference (CPU)</span>
      </div>
      <div style="margin-top: 4px; padding-top: 8px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-dim);">
        Store: demo-01 · Uptime: 2h 14m
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
        <span style="color: var(--accent-green);">✓</span> Fully offline capable
      </div>
      <div style="margin-top: 4px; padding-top: 8px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-dim);">
        DPDP Act 2023 compliant
      </div>
    </div>
  </div>
</main>
