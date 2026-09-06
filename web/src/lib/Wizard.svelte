<script>
  // Clips -> check the view -> process. Three steps, one Next each.
  import { get, post, upload, url } from './api.js';

  let { ondone } = $props();

  let step = $state(1);
  let files = $state([]);              // { file, role }
  let busy = $state(false);
  let error = $state(null);
  let uploadPct = $state(0);

  let run = $state(null);
  let shelfRows = $state(2);
  let shelfCols = $state(3);
  let counters = $state(2);
  let targetWait = $state(180);
  let backend = $state('');            // '' = let the box decide
  let door = $state([[0, 400], [639, 400]]);
  let progress = $state(null);
  let poll = null;

  const roleOf = (i) => (i === 0 ? 'overhead' : 'shelf');
  let hasOverhead = $derived(files.some((f) => f.role === 'overhead'));
  let hasShelf = $derived(files.some((f) => f.role === 'shelf'));
  let pct = $derived(
    progress?.total ? Math.min(100, (progress.processed / progress.total) * 100) : null,
  );

  function addFiles(list) {
    const incoming = [...list].filter((f) => f.size > 0);
    files = [...files, ...incoming.map((file, i) => ({ file, role: roleOf(files.length + i) }))];
    error = null;
  }

  function onDrop(e) {
    e.preventDefault();
    addFiles(e.dataTransfer.files);
  }

  async function doUpload() {
    if (!files.length) return;
    busy = true;
    error = null;
    try {
      run = await upload(files.map((f) => f.file), files.map((f) => f.role),
                         (p) => (uploadPct = p * 100));
      step = 2;
    } catch (e) {
      error = e.message;
    } finally {
      busy = false;
    }
  }

  async function start() {
    busy = true;
    error = null;
    try {
      progress = await post(`/api/runs/${run.run_id}/start`, {
        shelf_rows: shelfRows, shelf_cols: shelfCols, counters,
        target_wait_s: targetWait, backend: backend || null,
        door_line: hasOverhead ? door : null,
      });
      step = 3;
      poll = setInterval(watch, 700);
    } catch (e) {
      error = e.message;
    } finally {
      busy = false;
    }
  }

  async function watch() {
    try {
      progress = await get(`/api/runs/${run.run_id}`);
    } catch (e) {
      return;                          // a dropped poll is not a failed run
    }
    if (progress.state === 'done' || progress.state === 'error') {
      clearInterval(poll);
      poll = null;
      if (progress.state === 'done') ondone?.(run.run_id);
      else error = progress.error;
    }
  }

  // ---- the draggable door line ----
  let dragging = null;

  function svgPoint(evt, svg) {
    const p = svg.createSVGPoint();
    p.x = evt.clientX;
    p.y = evt.clientY;
    const { x, y } = p.matrixTransform(svg.getScreenCTM().inverse());
    return [Math.max(0, Math.min(639, Math.round(x))), Math.max(0, Math.min(479, Math.round(y)))];
  }

  function onMove(e) {
    if (dragging === null) return;
    const next = [...door];
    next[dragging] = svgPoint(e, e.currentTarget);
    door = next;
  }

  const shelfBoxes = $derived.by(() => {
    // Mirrors tools/autoconfig.rois(), drawn at 640x480 instead of PL scale.
    const out = [];
    const mx = Math.round(640 * 0.08), my = Math.round(480 * 0.08);
    const cw = Math.floor((640 - 2 * mx) / shelfCols), ch = Math.floor((480 - 2 * my) / shelfRows);
    for (let r = 0; r < shelfRows; r++)
      for (let c = 0; c < shelfCols; c++)
        out.push({ id: 'ABCDEFGH'[r] + (c + 1), x: mx + c * cw + cw / 12, y: my + r * ch + ch / 12,
                   w: cw - cw / 6, h: ch - ch / 6 });
    return out;
  });
</script>

<div class="wizard">
  <div class="wiz-steps">
    {#each ['Clips', 'Check the view', 'Process'] as label, i}
      <div class="wiz-step" class:active={step === i + 1} class:done={step > i + 1}>
        <span class="wiz-num">{step > i + 1 ? '✓' : i + 1}</span>{label}
      </div>
    {/each}
  </div>

  {#if error}
    <div class="wiz-error">{error}</div>
  {/if}

  <!-- ── 1. clips ── -->
  {#if step === 1}
    <div class="card">
      <div class="card-header"><span class="card-title">Upload footage</span></div>
      <label class="dropzone" ondragover={(e) => e.preventDefault()} ondrop={onDrop}>
        <input type="file" accept="video/*" multiple hidden
               onchange={(e) => addFiles(e.target.files)} />
        <strong>Drop video files here</strong>
        <span>or click to choose · mp4, mov, mkv, avi, mjpeg</span>
      </label>

      {#if files.length}
        <table class="wiz-table">
          <thead>
            <tr><th>File</th><th>Size</th><th>Camera</th><th></th></tr>
          </thead>
          <tbody>
            {#each files as f, i}
              <tr>
                <td class="mono">{f.file.name}</td>
                <td class="mono">{(f.file.size / 1048576).toFixed(1)} MB</td>
                <td>
                  <select bind:value={files[i].role}>
                    <option value="overhead">Overhead / floor</option>
                    <option value="shelf">Shelf facing</option>
                  </select>
                </td>
                <td><button class="link" onclick={() => (files = files.filter((_, j) => j !== i))}>remove</button></td>
              </tr>
            {/each}
          </tbody>
        </table>
        <p class="hint">
          Several overhead clips play back to back as one continuous stream.
          {#if !hasShelf}No shelf clip: the shelf stream stays on the USB camera for the live demo.{/if}
        </p>
      {/if}

      {#if busy}
        <div class="bar"><div class="bar-fill" style="width: {uploadPct}%"></div></div>
      {/if}
      <div class="wiz-actions">
        <button class="btn" disabled={!files.length || busy} onclick={doUpload}>
          {busy ? `Uploading ${uploadPct.toFixed(0)}%` : 'Next'}
        </button>
      </div>
    </div>
  {/if}

  <!-- ── 2. check the view ── -->
  {#if step === 2}
    <div class="wiz-two">
      {#if hasOverhead}
        <div class="card">
          <div class="card-header">
            <span class="card-title">Entry line</span>
            <span class="hint">drag the ends onto the doorway</span>
          </div>
          <div class="preview">
            <img src={url(`/api/runs/${run.run_id}/preview?role=overhead`)} alt="first overhead frame" />
            <svg viewBox="0 0 640 480" role="group" aria-label="entry line" onpointermove={onMove}
                 onpointerup={() => (dragging = null)} onpointerleave={() => (dragging = null)}>
              <line x1={door[0][0]} y1={door[0][1]} x2={door[1][0]} y2={door[1][1]}
                    stroke="var(--accent-cyan)" stroke-width="4" />
              {#each door as p, i}
                <circle cx={p[0]} cy={p[1]} r="12" fill="var(--accent-cyan)" role="button"
                        tabindex="0" aria-label="entry line end {i + 1}"
                        onpointerdown={() => (dragging = i)} style="cursor: grab" />
              {/each}
            </svg>
          </div>
          <p class="hint">Anyone crossing it counts as an entry or an exit. Nothing else is stored.</p>
        </div>
      {/if}

      {#if hasShelf}
        <div class="card">
          <div class="card-header">
            <span class="card-title">Shelf facings</span>
            <span class="hint">{shelfRows * shelfCols} facings</span>
          </div>
          <div class="preview">
            <img src={url(`/api/runs/${run.run_id}/preview?role=shelf`)} alt="first shelf frame" />
            <svg viewBox="0 0 640 480">
              {#each shelfBoxes as b}
                <rect x={b.x} y={b.y} width={b.w} height={b.h} fill="rgba(34,211,238,0.08)"
                      stroke="var(--accent-cyan)" stroke-width="2" />
                <text x={b.x + 6} y={b.y + 22} fill="var(--accent-cyan)" font-size="18">{b.id}</text>
              {/each}
            </svg>
          </div>
          <div class="wiz-fields">
            <label>Rows <input type="number" min="1" max="8" bind:value={shelfRows} /></label>
            <label>Columns <input type="number" min="1" max="8" bind:value={shelfCols} /></label>
          </div>
        </div>
      {/if}

      <div class="card">
        <div class="card-header"><span class="card-title">Store</span></div>
        <div class="wiz-fields">
          <label>Checkout counters <input type="number" min="1" max="8" bind:value={counters} /></label>
          <label>Target wait (s) <input type="number" min="30" step="30" bind:value={targetWait} /></label>
          <label>Detector
            <select bind:value={backend}>
              <option value="">Auto (YOLO if this box has it)</option>
              <option value="yolo">YOLO — people detector</option>
              <option value="reference">Background subtraction (CPU)</option>
            </select>
          </label>
        </div>
        <div class="wiz-actions">
          <button class="link" onclick={() => (step = 1)}>Back</button>
          <button class="btn" disabled={busy} onclick={start}>Process</button>
        </div>
      </div>
    </div>
  {/if}

  <!-- ── 3. processing ── -->
  {#if step === 3}
    <div class="card">
      <div class="card-header">
        <span class="card-title">Processing</span>
        <span class="badge badge-backend">{progress?.backend ?? '…'}</span>
      </div>
      <div class="stat-value cyan">{pct === null ? '…' : pct.toFixed(0) + '%'}</div>
      <div class="stat-label">
        {progress?.processed ?? 0} frames{progress?.total ? ` of ${progress.total}` : ''}
        · {progress?.fps ?? 0} fps
      </div>
      <div class="bar"><div class="bar-fill" style="width: {pct ?? 5}%"></div></div>
      <p class="hint">
        Inference runs on this box. The clips never leave it, and no frame is written to disk.
      </p>
    </div>
  {/if}
</div>

<style>
  .wizard { max-width: 1100px; margin: 32px auto; padding: 0 20px; display: flex;
            flex-direction: column; gap: 18px; }
  .wiz-steps { display: flex; gap: 10px; }
  .wiz-step { display: flex; align-items: center; gap: 8px; font-size: 13px;
              color: var(--text-dim); padding: 8px 14px; border-radius: 100px;
              border: 1px solid var(--border); }
  .wiz-step.active { color: var(--text-primary); border-color: var(--accent-cyan); }
  .wiz-step.done { color: var(--accent-green); }
  .wiz-num { display: grid; place-items: center; width: 20px; height: 20px;
             border-radius: 50%; background: var(--bg-elevated); font-size: 11px; }
  .wiz-step.active .wiz-num { background: var(--accent-cyan); color: #fff; }
  .wiz-two { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }
  .wiz-error { padding: 12px 16px; font-size: 13px; background: #fee2e2;
               color: var(--accent-red); border: 1px solid #fecaca; }
  .dropzone { display: flex; flex-direction: column; align-items: center; gap: 6px;
              padding: 42px; border: 1.5px dashed var(--border-active); cursor: pointer;
              border-radius: var(--radius-lg); color: var(--text-muted); font-size: 13px; }
  .dropzone:hover { border-color: var(--accent-cyan); background: var(--bg-card-hover); }
  .dropzone strong { color: var(--text-primary); font-size: 15px; }
  .wiz-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 16px; }
  .wiz-table th { text-align: left; font-weight: 500; color: var(--text-dim);
                  padding: 4px 8px; font-size: 11px; text-transform: uppercase; }
  .wiz-table td { padding: 8px; border-top: 1px solid var(--border); color: var(--text-secondary); }
  .wiz-fields { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 12px; }
  .wiz-fields label { display: flex; flex-direction: column; gap: 5px; font-size: 12px;
                      color: var(--text-muted); }
  input, select { background: var(--bg-elevated); border: 1px solid var(--border);
                  color: var(--text-primary); border-radius: var(--radius-sm);
                  padding: 7px 10px; font-family: var(--font-mono); font-size: 13px; }
  .wiz-actions { display: flex; justify-content: flex-end; align-items: center;
                 gap: 14px; margin-top: 18px; }
  .btn { background: var(--accent-cyan); color: #fff; border: none; font-weight: 700;
         padding: 10px 26px; border-radius: var(--radius-md); cursor: pointer;
         font-size: 14px; transition: filter var(--transition-fast); }
  .btn:hover:not(:disabled) { filter: brightness(1.12); }
  .btn:disabled { opacity: 0.45; cursor: default; }
  .link { background: none; border: none; color: var(--text-muted); cursor: pointer;
          font-size: 13px; text-decoration: underline; }
  .hint { font-size: 12px; color: var(--text-dim); margin-top: 10px; line-height: 1.5; }
  .preview { position: relative; border-radius: var(--radius-md); overflow: hidden;
             background: #000; aspect-ratio: 4 / 3; }
  .preview img, .preview svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  .preview svg { touch-action: none; }
  .bar { height: 8px; border-radius: 4px; background: var(--bg-elevated);
         overflow: hidden; margin-top: 14px; }
  .bar-fill { height: 100%; background: var(--accent-blue);
              transition: width var(--transition-normal); }
</style>
