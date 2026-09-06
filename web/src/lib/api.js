// Every call goes to the box the dashboard is served from. In `npm run dev`
// the page is on :5173 and the API on :8000, so the base is explicit there.
const BASE = import.meta.env.DEV ? 'http://localhost:8000' : '';

export const url = (path) => BASE + path;

export async function get(path) {
  const r = await fetch(url(path));
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function post(path, body) {
  const r = await fetch(url(path), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
  const text = await r.text();
  if (!r.ok) throw new Error(detail(text) || `${path}: ${r.status}`);
  return text ? JSON.parse(text) : null;
}

export async function del(path) {
  const r = await fetch(url(path), { method: 'DELETE' });
  if (!r.ok) throw new Error(detail(await r.text()) || `${path}: ${r.status}`);
  return r.json();
}

function detail(text) {
  try { return JSON.parse(text).detail; } catch { return text; }
}

// fetch() cannot report upload progress and these are video files, so XHR it is.
export function upload(files, roles, onProgress) {
  const form = new FormData();
  files.forEach((f, i) => {
    form.append('files', f);
    form.append('roles', roles[i]);
  });
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url('/api/runs'));
    xhr.upload.onprogress = (e) =>
      e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status < 400) return resolve(JSON.parse(xhr.responseText));
      reject(new Error(detail(xhr.responseText) || `upload failed: ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error('upload failed: no response from the server'));
    xhr.send(form);
  });
}

export const fmtWait = (s) =>
  `${Math.floor((s || 0) / 60)}:${String(Math.round((s || 0) % 60)).padStart(2, '0')}`;

export const fmtDuration = (s) => {
  s = Math.max(0, Math.round(s || 0));
  return s >= 3600
    ? `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
    : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
};

export const clockOf = (t) =>
  t ? new Date(t * 1000).toLocaleTimeString('en-IN', { hour12: false }) : '—';
