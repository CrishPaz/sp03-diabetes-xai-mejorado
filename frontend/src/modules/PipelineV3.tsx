import React, { useEffect, useMemo, useRef, useState } from 'react';

type Json = any;
type Section = 'datos' | 'entrenamiento' | 'evaluacion' | 'explicabilidad' | 'inferencia' | 'reportes' | 'optuna';
const SECTIONS: [Section, string][] = [
  ['datos', 'F1 · Datos'], ['entrenamiento', 'F2 · Entrenamiento'], ['evaluacion', 'F3 · Evaluación'],
  ['explicabilidad', 'F4 · Explicabilidad'], ['inferencia', 'F6 · Inferencia'], ['reportes', 'F5 · Reportes'], ['optuna', 'F7 · Optuna'],
];
const COLORS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'];
const MODEL_NAMES: Record<string, string> = { lr: 'Reg. logística', rf: 'Random Forest', xgb: 'XGBoost', tabnet: 'TabNet', stack: 'Stacking' };
const fmt = (v: any, d = 3) => (typeof v === 'number' ? v.toFixed(d) : v ?? '—');

function useApi(apiBase: string) {
  return async (path: string, init?: RequestInit) => {
    const r = await fetch(`${apiBase}${path}`, init);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body));
    return body;
  };
}

const Card: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
    <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
    {children}
  </div>
);

const Interp: React.FC<{ i?: Json }> = ({ i }) =>
  i ? (
    <dl className="grid gap-1 text-[11px] text-slate-400 border-l-2 border-sky-600/60 pl-3">
      <div><dt className="inline font-semibold text-slate-300">Qué muestra: </dt><dd className="inline">{i.que_muestra}</dd></div>
      <div><dt className="inline font-semibold text-slate-300">Cómo leerla: </dt><dd className="inline">{i.como_leerla}</dd></div>
      <div><dt className="inline font-semibold text-slate-300">Interpretación: </dt><dd className="inline">{i.interpretacion}</dd></div>
      <div><dt className="inline font-semibold text-slate-300">Limitaciones: </dt><dd className="inline">{i.limitaciones}</dd></div>
    </dl>
  ) : null;

const Table: React.FC<{ cols: string[]; rows: any[][] }> = ({ cols, rows }) => (
  <div className="overflow-x-auto">
    <table className="w-full text-xs text-slate-300">
      <thead><tr>{cols.map((c) => <th key={c} className="text-left font-medium text-slate-400 border-b border-slate-800 px-2 py-1">{c}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i} className="border-b border-slate-800/50">{r.map((v, j) => <td key={j} className="px-2 py-1 tabular-nums">{typeof v === 'object' && v !== null ? JSON.stringify(v) : fmt(v)}</td>)}</tr>)}</tbody>
    </table>
  </div>
);

const LineChart: React.FC<{ series: { name: string; x: number[]; y: number[] }[]; xLabel: string; yLabel: string; diagonal?: boolean; yMin?: number; yMax?: number }> = ({ series, xLabel, yLabel, diagonal, yMin, yMax }) => {
  const W = 360, H = 240, L = 40, B = 30;
  const xs = series.flatMap((s) => s.x), ys = series.flatMap((s) => s.y);
  const x0 = Math.min(...xs, 0), x1 = Math.max(...xs, 1e-9), y0 = yMin ?? Math.min(...ys, 0), y1 = yMax ?? Math.max(...ys, 1e-9);
  const X = (v: number) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - 10);
  const Y = (v: number) => H - B - ((v - y0) / (y1 - y0 || 1)) * (H - B - 10);
  return (
    <svg viewBox={`0 0 ${W} ${H + 18 * Math.ceil(series.length / 3)}`} className="w-full max-w-md" role="img" aria-label={yLabel}>
      <line x1={L} y1={H - B} x2={W - 10} y2={H - B} stroke="#475569" />
      <line x1={L} y1={10} x2={L} y2={H - B} stroke="#475569" />
      {diagonal && <line x1={X(x0)} y1={Y(y0)} x2={X(x1)} y2={Y(y1)} stroke="#64748b" strokeDasharray="3 3" />}
      {series.map((s, k) => (
        <polyline key={s.name} fill="none" stroke={COLORS[k % COLORS.length]} strokeWidth={1.5} points={s.x.map((v, i) => `${X(v)},${Y(Math.max(y0, Math.min(y1, s.y[i])))}`).join(' ')} />
      ))}
      <text x={(W + L) / 2} y={H - 6} fill="#94a3b8" fontSize={10} textAnchor="middle">{xLabel}</text>
      <text x={10} y={H / 2} fill="#94a3b8" fontSize={10} transform={`rotate(-90 10 ${H / 2})`} textAnchor="middle">{yLabel}</text>
      <text x={L - 4} y={Y(y1) + 4} fill="#64748b" fontSize={9} textAnchor="end">{fmt(y1, 2)}</text>
      <text x={L - 4} y={Y(y0)} fill="#64748b" fontSize={9} textAnchor="end">{fmt(y0, 2)}</text>
      {series.map((s, k) => (
        <text key={s.name} x={L + (k % 3) * 110} y={H + 12 + 16 * Math.floor(k / 3)} fill={COLORS[k % COLORS.length]} fontSize={10}>■ {s.name}</text>
      ))}
    </svg>
  );
};

const Bars: React.FC<{ data: Record<string, number>; max?: number }> = ({ data, max }) => {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const m = max ?? Math.max(...entries.map((e) => Math.abs(e[1])), 1e-9);
  return (
    <div className="space-y-1">
      {entries.map(([k, v]) => (
        <div key={k} className="flex items-center gap-2 text-[11px]">
          <span className="w-44 truncate text-slate-400">{k}</span>
          <div className="flex-1 h-2 bg-slate-800 rounded"><div className="h-2 rounded bg-sky-500" style={{ width: `${(Math.abs(v) / m) * 100}%` }} /></div>
          <span className="w-16 text-right tabular-nums text-slate-300">{fmt(v, 4)}</span>
        </div>
      ))}
    </div>
  );
};

const Heatmap: React.FC<{ matrix: number[][]; cols: string[] }> = ({ matrix, cols }) => {
  const cw = 26, ch = Math.max(1, Math.min(3, 300 / matrix.length));
  const max = Math.max(...matrix.flat(), 1e-9);
  return (
    <svg viewBox={`0 0 ${cols.length * cw + 10} ${matrix.length * ch + 90}`} className="w-full max-w-sm" role="img" aria-label="Mapa de atención TabNet">
      {matrix.map((row, i) => row.map((v, j) => <rect key={`${i}-${j}`} x={j * cw} y={i * ch} width={cw - 1} height={ch} fill={`rgba(56,189,248,${v / max})`} />))}
      {cols.map((c, j) => <text key={c} x={j * cw + 10} y={matrix.length * ch + 6} fontSize={8} fill="#94a3b8" transform={`rotate(60 ${j * cw + 10} ${matrix.length * ch + 6})`}>{c}</text>)}
    </svg>
  );
};

export default function PipelineV3({ apiBase }: { apiBase: string }) {
  const api = useApi(apiBase);
  const [section, setSection] = useState<Section>('datos');
  const [error, setError] = useState('');
  const [results, setResults] = useState<Json[]>([]);
  const [key, setKey] = useState('nhanes_2017_2018__B');
  const refreshResults = () => api('/v3/results').then((r) => { setResults(r); if (r.length && !r.find((x: Json) => x.clave === key)) setKey(r[0].clave); }).catch((e) => setError(e.message));
  useEffect(() => { refreshResults(); }, []);
  const guard = (p: Promise<any>) => p.catch((e) => setError(e.message));

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200">
        Prototipo académico, no clínico. Resultados del pipeline v3: partición 70/15/15, CV anidada 5×3 con Optuna, escenarios A (con HbA1c/glucosa, detección) y B (sin biomarcadores glucémicos, predicción temprana). El modelo de supervivencia está deshabilitado: requiere tiempo hasta evento y censura.
      </div>
      <div className="flex flex-wrap gap-1">
        {SECTIONS.map(([k, label]) => (
          <button key={k} onClick={() => { setSection(k); setError(''); }} className={`px-3 py-1.5 rounded-lg text-xs ${section === k ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40' : 'text-slate-400 hover:bg-slate-800'}`}>{label}</button>
        ))}
        {results.length > 0 && !['datos', 'entrenamiento', 'optuna'].includes(section) && (
          <select value={key} onChange={(e) => setKey(e.target.value)} className="ml-auto bg-slate-900 border border-slate-700 rounded-lg text-xs text-slate-200 px-2">
            {results.map((r) => <option key={r.clave} value={r.clave}>{r.cohorte} · escenario {r.escenario}</option>)}
          </select>
        )}
      </div>
      {error && <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-2 text-xs text-rose-200">{error}</div>}
      {section === 'datos' && <DataSection api={api} guard={guard} />}
      {section === 'entrenamiento' && <TrainSection api={api} guard={guard} apiBase={apiBase} onDone={refreshResults} />}
      {section === 'evaluacion' && <EvalSection api={api} guard={guard} k={key} />}
      {section === 'explicabilidad' && <ExplainSection api={api} guard={guard} k={key} />}
      {section === 'inferencia' && <InferenceSection api={api} guard={guard} k={key} />}
      {section === 'reportes' && <ReportSection api={api} guard={guard} k={key} apiBase={apiBase} />}
      {section === 'optuna' && <OptunaSection api={api} guard={guard} />}
    </div>
  );
}

type SecProps = { api: ReturnType<typeof useApi>; guard: (p: Promise<any>) => Promise<any> };

function DataSection({ api, guard }: SecProps) {
  const [cohort, setCohort] = useState('nhanes_2017_2018');
  const [scenario, setScenario] = useState('A');
  const [reg, setReg] = useState<Json>(null);
  const [eda, setEda] = useState<Json>(null);
  const [val, setVal] = useState<Json>(null);
  const [meta, setMeta] = useState<Json>(null);
  useEffect(() => { guard(api('/v3/data/registry').then(setReg)); }, []);
  useEffect(() => { setEda(null); setVal(null); guard(api(`/v3/data/${cohort}/eda?scenario=${scenario}`).then(setEda)); guard(api(`/v3/data/${cohort}/metadata`).then(setMeta)); }, [cohort, scenario]);
  const saveMeta = () => guard(api(`/v3/data/${cohort}/metadata`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ autor: 'investigador', variables: meta.variables }) }).then(setMeta));
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Registro de cohortes y diagrama de exclusiones">
        {reg ? <Table cols={['Cohorte', 'Rol', 'Licencia', 'n oficial', 'Exclusiones']} rows={reg.cohortes.map((c: Json) => [c.nombre, c.rol, c.licencia, c.n_oficial, c.flujo_exclusiones.map((f: Json) => `${f.paso}: ${f.n}`).join(' → ')])} /> : <p className="text-xs text-slate-500">Cargando…</p>}
      </Card>
      <Card title="Cohorte y escenario">
        <div className="flex gap-2">
          <select value={cohort} onChange={(e) => setCohort(e.target.value)} className="bg-slate-950 border border-slate-700 rounded text-xs px-2 py-1 text-slate-200"><option value="nhanes_2017_2018">NHANES 2017–2018</option><option value="pima_indians">Pima Indians</option></select>
          <select value={scenario} onChange={(e) => setScenario(e.target.value)} className="bg-slate-950 border border-slate-700 rounded text-xs px-2 py-1 text-slate-200"><option value="A">A · detección</option><option value="B">B · predicción temprana</option></select>
          <button onClick={() => guard(api(`/v3/data/${cohort}/validate-groups?scenario=${scenario}`, { method: 'POST' }).then(setVal))} className="px-3 py-1 rounded bg-sky-600 text-xs text-white">Validar grupos</button>
        </div>
        {eda && <p className="text-xs text-slate-400">Etiqueta: {eda.auditoria.etiqueta}. Escenario: {eda.auditoria.descripcion_escenario}. n = {eda.auditoria.n_oficial}, eventos = {eda.auditoria.eventos} (prevalencia {fmt(eda.auditoria.prevalencia)}).</p>}
        {val && (<>
          <p className={`text-xs font-semibold ${val.estado === 'PASS' ? 'text-emerald-400' : 'text-rose-400'}`}>Validación de grupos: {val.estado}{val.alertas?.length ? ` · alertas: ${val.alertas.join('; ')}` : ''}</p>
          <Table cols={['Control', 'OK', 'Detalle']} rows={val.controles.map((c: Json) => [c.control, c.ok ? 'sí' : 'no', c.detalle])} />
          <Table cols={['Clase', 'Desarrollo 70 %', 'Validación 15 %', 'Prueba 15 %', 'Total']} rows={val.tabla_particion.map((r: Json) => [r.clase, r.desarrollo, r.validacion, r.prueba, r.total])} />
        </>)}
      </Card>
      {eda && (
        <Card title="Tabla 1 por clase">
          <Table cols={['Variable', 'Con DM2', 'Sin DM2', '% faltantes', 'p (descriptivo)', 'SMD']} rows={eda.tabla1.filas.map((r: Json) => [r.variable, r.con_dm2, r.sin_dm2, r.faltantes_pct, r.p_descriptivo, r.smd])} />
          <Interp i={eda.tabla1.interpretabilidad} />
        </Card>
      )}
      {meta && (
        <Card title={`Editor de metadata (versión ${meta.version})`}>
          <div className="max-h-72 overflow-y-auto">
            <table className="w-full text-[11px] text-slate-300"><thead><tr><th className="text-left">Variable</th><th>Unidad</th><th>Mín</th><th>Máx</th><th>Rol</th></tr></thead>
              <tbody>{meta.variables.map((v: Json, i: number) => (
                <tr key={v.nombre}>
                  <td>{v.nombre}</td>
                  <td><input className="w-16 bg-slate-950 border border-slate-800 rounded px-1" value={v.unidad} onChange={(e) => { const m = structuredClone(meta); m.variables[i].unidad = e.target.value; setMeta(m); }} /></td>
                  {(['rango_min', 'rango_max'] as const).map((f) => <td key={f}><input type="number" className="w-16 bg-slate-950 border border-slate-800 rounded px-1" value={v[f] ?? ''} onChange={(e) => { const m = structuredClone(meta); m.variables[i][f] = e.target.value === '' ? null : Number(e.target.value); setMeta(m); }} /></td>)}
                  <td><select className="bg-slate-950 border border-slate-800 rounded" value={v.rol} onChange={(e) => { const m = structuredClone(meta); m.variables[i].rol = e.target.value; setMeta(m); }}>{['predictor', 'objetivo', 'grupo', 'sensible_solo_auditoria', 'excluida'].map((r) => <option key={r}>{r}</option>)}</select></td>
                </tr>))}</tbody></table>
          </div>
          <button onClick={saveMeta} className="px-3 py-1 rounded bg-emerald-600 text-xs text-white">Guardar nueva versión</button>
          <p className="text-[11px] text-slate-500">El servidor impide marcar el identificador o la etiqueta como predictor, y exige justificación para usar sexo como predictor.</p>
        </Card>
      )}
    </div>
  );
}

function TrainSection({ api, guard, apiBase, onDone }: SecProps & { apiBase: string; onDone: () => void }) {
  const [form, setForm] = useState({ cohort: 'pima_indians', scenario: 'A', models: ['lr', 'rf', 'xgb', 'tabnet', 'stack'], n_trials: '' });
  const [catalog, setCatalog] = useState<Json>(null);
  const [jobList, setJobList] = useState<Json[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [events, setEvents] = useState<Json[]>([]);
  const [state, setState] = useState('');
  const es = useRef<EventSource | null>(null);
  useEffect(() => { guard(api('/v3/models/catalog').then(setCatalog)); guard(api('/v3/train/jobs').then(setJobList)); return () => es.current?.close(); }, []);
  const follow = (id: string) => {
    es.current?.close(); setJobId(id); setEvents([]); setState('running');
    const src = new EventSource(`${apiBase}/v3/train/jobs/${id}/stream`);   // al reconectar se rehidrata desde el log del job
    src.onmessage = (m) => { const ev = JSON.parse(m.data); if (String(ev.event).startsWith('job_')) { setState(ev.event.slice(4)); src.close(); onDone(); } else setEvents((p) => [...p.slice(-400), ev]); };
    es.current = src;
  };
  const launch = () => guard(api('/v3/train/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...form, n_trials: form.n_trials === '' ? null : Number(form.n_trials) }) }).then((j) => follow(j.id)));
  const trials = events.filter((e) => e.event === 'trial_completed');
  const best = trials.reduce((acc: number[], t) => [...acc, Math.max(acc[acc.length - 1] ?? 0, t.valor)], []);
  const folds = events.filter((e) => e.event === 'fold_completed');
  const epoch = events.filter((e) => e.event === "epoch").slice(-1)[0];
  const res = events.filter((e) => e.event === "resource").slice(-1)[0];
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Lanzar entrenamiento (CV anidada 5×3)">
        <div className="flex flex-wrap gap-2 text-xs">
          <select value={form.cohort} onChange={(e) => setForm({ ...form, cohort: e.target.value })} className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200"><option value="nhanes_2017_2018">NHANES</option><option value="pima_indians">Pima</option></select>
          <select value={form.scenario} onChange={(e) => setForm({ ...form, scenario: e.target.value })} className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200"><option value="A">Escenario A</option><option value="B">Escenario B</option></select>
          <input placeholder="n_trials (YAML)" value={form.n_trials} onChange={(e) => setForm({ ...form, n_trials: e.target.value.replace(/\D/g, '') })} className="w-32 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200" />
        </div>
        <div className="flex flex-wrap gap-3 text-xs text-slate-300">
          {catalog?.catalogo.map((m: Json) => (
            <label key={m.id} className={m.habilitado ? '' : 'text-slate-600'} title={m.motivo}>
              <input type="checkbox" disabled={!m.habilitado} checked={form.models.includes(m.id)} onChange={(e) => setForm({ ...form, models: e.target.checked ? [...form.models, m.id] : form.models.filter((x) => x !== m.id) })} /> {m.nombre}{!m.habilitado && ` — ${m.motivo}`}
            </label>
          ))}
        </div>
        <button onClick={launch} className="px-3 py-1.5 rounded bg-sky-600 text-xs text-white">Iniciar trabajo</button>
        <Table cols={['Trabajo', 'Tipo', 'Cohorte', 'Estado', '']} rows={jobList.slice(0, 8).map((j) => [j.id, j.tipo, j.cohorte, j.estado, j.estado === 'running' ? '▶ seguir' : ''])} />
        <div className="flex gap-2">{jobList.filter((j) => j.estado === 'running').map((j) => <button key={j.id} onClick={() => follow(j.id)} className="text-xs text-sky-300 underline">Seguir {j.id}</button>)}</div>
      </Card>
      {jobId && (
        <Card title={`Monitoreo en vivo · ${jobId} · ${state}`}>
          <div className="flex gap-2 text-xs text-slate-400">
            <span>Pliegues completados: {folds.length}</span><span>· Trials: {trials.length}</span>{res && <span>· CPU {fmt(res.cpu, 0)} % · RAM {fmt(res.ram_pct, 0)} %</span>}
            <button onClick={() => guard(api(`/v3/train/jobs/${jobId}/cancel`, { method: 'POST' }).then(() => setState('cancelled')))} className="ml-auto text-rose-300 underline">Cancelar</button>
          </div>
          {best.length > 1 && <LineChart series={[{ name: 'Mejor AUC interno acumulado', x: best.map((_, i) => i), y: best }]} xLabel="Trial" yLabel="ROC-AUC" />}
          {epoch && <LineChart series={[{ name: 'Pérdida TabNet', x: epoch.historial.loss.map((_: any, i: number) => i), y: epoch.historial.loss }, ...(epoch.historial.val_0_auc ? [{ name: 'AUC parada temprana', x: epoch.historial.val_0_auc.map((_: any, i: number) => i), y: epoch.historial.val_0_auc }] : [])]} xLabel="Época" yLabel="Valor" />}
          <pre className="max-h-48 overflow-y-auto text-[10px] text-slate-400 bg-slate-950 p-2 rounded">{events.filter((e) => e.event !== 'trial_completed' && e.event !== 'resource' && e.event !== 'epoch').map((e) => JSON.stringify(e)).join('\n')}</pre>
        </Card>
      )}
    </div>
  );
}

function EvalSection({ api, guard, k }: SecProps & { k: string }) {
  const [r, setR] = useState<Json>(null);
  const [model, setModel] = useState('');
  useEffect(() => { setR(null); guard(api(`/v3/evaluate/${k}`).then((x) => { setR(x); setModel(x.modelo_recomendado); })); }, [k]);
  if (!r) return <p className="text-xs text-slate-500">Cargando resultados…</p>;
  const pm = r.evaluacion_prueba.por_modelo;
  const keys = ['roc_auc', 'pr_auc', 'sensibilidad', 'especificidad', 'vpp', 'vpn', 'f1', 'mcc', 'brier', 'ece', 'pendiente'];
  const cm = pm[model]?.metricas.matriz_confusion;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="CV anidada: AUC externo por modelo (5 pliegues)">
        <Table cols={['Modelo', 'Media', 'DE', 'IC95 t', 'Brecha optimismo', 'Pliegues']} rows={Object.entries(r.cv_anidada.resumen).map(([m, s]: [string, Json]) => [MODEL_NAMES[m] ?? m, s.media, s.de, s.ic95_t.map((v: number) => v.toFixed(3)).join('–'), s.brecha_optimismo, s.auc_externo.map((v: number) => v.toFixed(3)).join(' ')])} />
        <Interp i={r.cv_anidada.interpretabilidad} />
        <p className="text-[11px] text-slate-400">Friedman: {r.cv_anidada.pruebas.friedman ? `χ² = ${fmt(r.cv_anidada.pruebas.friedman.chi2)}, p = ${fmt(r.cv_anidada.pruebas.friedman.p)}` : '—'} · Diferencia crítica de Nemenyi = {r.cv_anidada.pruebas.diferencia_critica_nemenyi} · {r.cv_anidada.pruebas.nota_wilcoxon}</p>
        <Table cols={['Par', 'Dif. media', 't Nadeau–Bengio', 'p', 'p Holm', 'p Nemenyi']} rows={r.cv_anidada.pruebas.pares_nadeau_bengio.map((p: Json) => [`${p.modelo_a} vs ${p.modelo_b}`, p.media_diferencia, p.t, p.p, p.p_holm, p.p_nemenyi])} />
      </Card>
      <Card title="Prueba aislada (lectura única): métricas con IC 95 %">
        <Table cols={['Modelo', ...keys, 'umbral']} rows={Object.entries(pm).map(([m, v]: [string, Json]) => [MODEL_NAMES[m] ?? m, ...keys.map((kk) => v.ic95[kk] ? `${fmt(v.metricas[kk])} [${fmt(v.ic95[kk][0])}–${fmt(v.ic95[kk][1])}]` : v.metricas[kk]), v.umbral])} />
        <Interp i={r.evaluacion_prueba.interpretabilidad} />
        <Table cols={['Par', 'Dif. AUC [IC95]', 'DeLong p Holm', 'McNemar p Holm']} rows={r.evaluacion_prueba.pruebas.pares.map((p: Json) => [`${p.modelo_a} vs ${p.modelo_b}`, `${fmt(p.dif_auc)} [${p.dif_auc_ic95.map((v: number) => v.toFixed(3)).join(', ')}]`, p.delong_p_holm, p.mcnemar_p_holm])} />
        <Interp i={r.evaluacion_prueba.pruebas.interpretabilidad} />
      </Card>
      <Card title="Curvas ROC y precisión-sensibilidad">
        <LineChart diagonal series={Object.entries(pm).map(([m, v]: [string, Json]) => ({ name: MODEL_NAMES[m] ?? m, x: v.curvas.roc.fpr, y: v.curvas.roc.tpr }))} xLabel="1 − especificidad" yLabel="Sensibilidad" />
        <LineChart series={Object.entries(pm).map(([m, v]: [string, Json]) => ({ name: MODEL_NAMES[m] ?? m, x: v.curvas.pr.recall, y: v.curvas.pr.precision }))} xLabel="Sensibilidad" yLabel="VPP" yMin={0} yMax={1} />
      </Card>
      <Card title="Calibración y curva de decisión">
        <LineChart diagonal series={Object.entries(pm).map(([m, v]: [string, Json]) => ({ name: MODEL_NAMES[m] ?? m, x: v.curvas.calibracion.map((c: Json) => c.pred), y: v.curvas.calibracion.map((c: Json) => c.obs) }))} xLabel="Probabilidad predicha" yLabel="Frecuencia observada" yMin={0} yMax={1} />
        <LineChart series={[...Object.entries(pm).map(([m, v]: [string, Json]) => ({ name: MODEL_NAMES[m] ?? m, x: v.curva_decision.map((c: Json) => c.umbral), y: v.curva_decision.map((c: Json) => c.modelo) })), { name: 'Tratar a todos', x: pm[model].curva_decision.map((c: Json) => c.umbral), y: pm[model].curva_decision.map((c: Json) => c.tratar_todos) }]} xLabel="Umbral" yLabel="Beneficio neto" yMin={-0.02} />
        <p className="text-[11px] text-slate-400">Calibración en Validación: {r.calibracion.metodo}. Cobertura conformal empírica en prueba (objetivo {1 - r.calibracion.conformal.alpha}): {Object.entries(r.calibracion.conformal.cobertura_prueba).map(([m, c]) => `${m} ${c}`).join(' · ')}</p>
      </Card>
      <Card title="Matriz de confusión y equidad">
        <select value={model} onChange={(e) => setModel(e.target.value)} className="bg-slate-950 border border-slate-700 rounded text-xs px-2 py-1 text-slate-200">{Object.keys(pm).map((m) => <option key={m} value={m}>{MODEL_NAMES[m] ?? m}</option>)}</select>
        {cm && <Table cols={['', 'Predicho +', 'Predicho −']} rows={[['Real +', cm.tp, `${cm.fn} (casos no detectados)`], ['Real −', `${cm.fp} (derivaciones innecesarias)`, cm.tn]]} />}
        {Object.entries(pm[model].equidad).map(([g, e]: [string, Json]) => (
          <div key={g}>
            <p className={`text-xs ${e.alerta ? 'text-rose-300' : 'text-slate-400'}`}>{g}: brecha de sensibilidad {fmt(e.brecha_sensibilidad)} {e.alerta ? '(alerta > 0.10)' : ''}</p>
            <Table cols={['Grupo', 'n', 'Eventos', 'AUC', 'Sens.', 'Espec.', 'Prob. media', 'Prev. obs.']} rows={e.filas.map((f: Json) => [f.grupo, f.n, f.eventos, f.roc_auc, f.sensibilidad, f.especificidad, f.prob_media, f.prevalencia_obs])} />
          </div>
        ))}
      </Card>
    </div>
  );
}

function ExplainSection({ api, guard, k }: SecProps & { k: string }) {
  const [x, setX] = useState<Json>(null);
  const [feats, setFeats] = useState<string[]>([]);
  useEffect(() => { setX(null); guard(api(`/v3/explain/${k}`).then(setX)); guard(api(`/v3/evaluate/${k}`).then((r) => setFeats(r.auditoria.predictores))); }, [k]);
  if (!x) return <p className="text-xs text-slate-500">Cargando explicaciones…</p>;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="lg:col-span-2 rounded-lg border border-slate-800 bg-slate-900/40 p-3 text-xs text-slate-400">{x.nota_gradcam} {x.advertencia}</div>
      {Object.entries(x.global).map(([name, imp]: [string, Json]) => <Card key={name} title={`Importancia global · ${name}`}><Bars data={imp} /></Card>)}
      <Card title="Concordancia entre métodos">
        <Table cols={['Método A', 'Método B', 'Spearman', 'Jaccard top-5']} rows={x.concordancia.map((c: Json) => [c.metodo_a, c.metodo_b, c.spearman, c.jaccard_top5])} />
        <Interp i={x.interpretabilidad} />
      </Card>
      {x.plausibilidad && <Card title="Plausibilidad clínica (dirección del efecto vs ADA)"><Table cols={['Variable', 'Rango', 'Dirección modelo', 'Dirección ADA', 'Coherente']} rows={x.plausibilidad.map((p: Json) => [p.variable, p.rango_importancia, p.direccion_modelo, p.direccion_ada, p.coherente])} /></Card>}
      {x.estabilidad_pliegues && <Card title="Estabilidad entre pliegues externos (ganancia XGBoost)"><Table cols={['Variable', 'Media', 'Rango', 'CV']} rows={x.estabilidad_pliegues.map((s: Json) => [s.variable, s.media, s.rango.join('–'), s.cv])} /></Card>}
      {x.tabnet && (<>
        <Card title="Mapa de atención TabNet (200 pacientes de prueba, ordenados por probabilidad)"><Heatmap matrix={x.tabnet.mapa_calor.matriz} cols={feats} /></Card>
        <Card title="Máscaras TabNet por paso (paciente de mayor riesgo) y por clase">
          {Object.entries(x.tabnet.paciente_ejemplo.pasos).map(([s, m]: [string, Json]) => <div key={s}><p className="text-[11px] text-slate-400">Paso {s} · dispersión {fmt(x.tabnet.dispersion_por_paso[s], 2)}</p><Bars data={Object.fromEntries(feats.map((f, i) => [f, m[i]]))} max={1} /></div>)}
          <p className="text-[11px] text-slate-400">Máscara media con DM2 vs sin DM2:</p>
          <Table cols={['Variable', 'Con DM2', 'Sin DM2']} rows={feats.map((f, i) => [f, x.tabnet.mascara_media_por_clase.con_dm2[i], x.tabnet.mascara_media_por_clase.sin_dm2[i]])} />
          <p className="text-[11px] text-slate-500">Delta de convergencia de Integrated Gradients: {x.tabnet.delta_convergencia_ig}</p>
        </Card>
      </>)}
    </div>
  );
}

function InferenceSection({ api, guard, k }: SecProps & { k: string }) {
  const [feats, setFeats] = useState<string[]>([]);
  const [vals, setVals] = useState<Record<string, string>>({});
  const [out, setOut] = useState<Json>(null);
  const [mmol, setMmol] = useState(false);
  const [wi, setWi] = useState<Json>(null);
  useEffect(() => { setOut(null); guard(api(`/v3/evaluate/${k}`).then((r) => setFeats(r.auditoria.predictores))); }, [k]);
  const numeric = useMemo(() => Object.fromEntries(Object.entries(vals).filter(([, v]) => v !== '' && !isNaN(Number(v))).map(([kk, v]) => [kk, Number(v)])), [vals]);
  const invalid = Object.entries(vals).filter(([, v]) => v !== '' && isNaN(Number(v))).map(([kk]) => kk);
  const run = () => guard(api('/v3/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: k, features: numeric, glucosa_mmol: mmol }) }).then(setOut));
  const whatIf = () => guard(api('/v3/what-if', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: k, features: numeric, variable: 'imc', values: [22, 25, 28, 31, 34, 37, 40] }) }).then(setWi));
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Datos del paciente (variables vacías se imputan con la mediana de entrenamiento)">
        <div className="grid grid-cols-2 gap-2">
          {feats.map((f) => <label key={f} className="text-[11px] text-slate-400">{f}<input value={vals[f] ?? ''} onChange={(e) => setVals({ ...vals, [f]: e.target.value })} className={`w-full bg-slate-950 border rounded px-2 py-1 text-slate-200 ${invalid.includes(f) ? 'border-rose-500' : 'border-slate-700'}`} /></label>)}
        </div>
        {feats.includes('glucosa') && <label className="text-[11px] text-slate-400"><input type="checkbox" checked={mmol} onChange={(e) => setMmol(e.target.checked)} /> Glucosa en mmol/L (se convierte a mg/dL)</label>}
        <div className="flex gap-2"><button disabled={invalid.length > 0} onClick={run} className="px-3 py-1.5 rounded bg-sky-600 text-xs text-white disabled:opacity-40">Estimar riesgo</button><button onClick={whatIf} className="px-3 py-1.5 rounded border border-slate-600 text-xs text-slate-300">¿Qué pasaría si cambia el IMC?</button></div>
      </Card>
      {out && (
        <Card title={`Riesgo estimado · ${MODEL_NAMES[out.modelo] ?? out.modelo}`}>
          <p className="text-2xl font-semibold text-slate-100">{(out.probabilidad * 100).toFixed(1)} %</p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded border border-slate-700 p-2"><p className="text-slate-400">Capa 1 · riesgo del modelo</p><p className="text-slate-100 font-semibold">{out.severidad_modelo}</p></div>
            <div className="rounded border border-slate-700 p-2"><p className="text-slate-400">Capa 2 · categoría ADA</p><p className="text-slate-100 font-semibold">{out.categoria_ada?.categoria ?? 'sin HbA1c/glucosa'}</p></div>
          </div>
          {out.discordancia_capas && <p className="text-xs text-amber-300">Discordancia entre capas: revisar.</p>}
          {out.fuera_de_distribucion.alerta && <p className="text-xs text-rose-300">Fuera de distribución (Mahalanobis {out.fuera_de_distribucion.distancia_mahalanobis} &gt; {out.fuera_de_distribucion.umbral}): estimación poco fiable.</p>}
          {out.confiabilidad_reducida && <p className="text-xs text-amber-300">Variables imputadas: {out.variables_imputadas.join(', ')}.</p>}
          <p className="text-xs text-slate-400">Conjunto conformal (α = {out.conjunto_conformal.alpha}): {out.conjunto_conformal.lectura}. Dispersión entre modelos: {out.dispersion_modelos}{out.desacuerdo_modelos ? ' (desacuerdo)' : ''}.</p>
          <Bars data={out.probabilidades_por_modelo} max={1} />
          <ul className="text-xs text-slate-300 list-disc pl-4">{out.top5_shap.map((e: Json) => <li key={e.variable}>{e.frase}</li>)}</ul>
          <p className="text-xs text-slate-300 whitespace-pre-line">{out.narrativa?.texto}</p>
          <p className="text-[10px] text-slate-500">{out.aviso}</p>
        </Card>
      )}
      {wi && <Card title="Sensibilidad del modelo al IMC (no es recomendación)"><LineChart series={[{ name: 'Probabilidad', x: wi.curva.map((c: Json) => c.valor), y: wi.curva.map((c: Json) => c.probabilidad) }]} xLabel="IMC (kg/m²)" yLabel="Probabilidad" yMin={0} yMax={1} /></Card>}
    </div>
  );
}

function ReportSection({ api, guard, k, apiBase }: SecProps & { k: string; apiBase: string }) {
  const [rep, setRep] = useState<Json>(null);
  const [tipo, setTipo] = useState('cientifico');
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Generar reporte (PDF, Word y Excel con las mismas cifras; figuras a 600 DPI)">
        <select value={tipo} onChange={(e) => setTipo(e.target.value)} className="bg-slate-950 border border-slate-700 rounded text-xs px-2 py-1 text-slate-200"><option value="cientifico">Científico</option><option value="gerencial">Gerencial</option></select>
        <button onClick={() => guard(api('/v3/reports/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: k, tipo }) }).then(setRep))} className="ml-2 px-3 py-1.5 rounded bg-sky-600 text-xs text-white">Generar</button>
        {rep && <div className="flex gap-3 text-xs">{Object.keys(rep.archivos).map((f) => <a key={f} className="text-sky-300 underline" href={`${apiBase}/v3/reports/${k}/download?format=${f}&tipo=${tipo}`}>Descargar {f.toUpperCase()}</a>)}</div>}
      </Card>
      {rep?.figuras.map((f: Json, i: number) => (
        <Card key={f.id} title={`Figura ${i + 1}`}><img src={`${apiBase}/v3/figures/${k}/fig_${f.id}?t=${Date.now()}`} alt={f.leyenda} className="max-w-xs bg-white rounded" /><p className="text-[11px] text-slate-400">{f.leyenda}</p></Card>
      ))}
    </div>
  );
}

function OptunaSection({ api, guard }: SecProps) {
  const [studies, setStudies] = useState<Json[]>([]);
  const [sel, setSel] = useState<Json>(null);
  const [cfg, setCfg] = useState({ cohort: 'pima_indians', scenario: 'A', model: 'xgb', n_trials: 20, sampler: 'tpe', pruner: 'median', multiobjective: false });
  const [msg, setMsg] = useState('');
  const load = () => guard(api('/v3/optuna/studies').then(setStudies));
  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, []);
  useEffect(() => { if (!sel?.nombre) return; const t = setInterval(() => guard(api(`/v3/optuna/studies/${sel.nombre}`).then(setSel)), 4000); return () => clearInterval(t); }, [sel?.nombre]);
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Configurar estudio exploratorio">
        <div className="flex flex-wrap gap-2 text-xs">
          {([['cohort', ['nhanes_2017_2018', 'pima_indians']], ['scenario', ['A', 'B']], ['model', ['lr', 'rf', 'xgb', 'tabnet']], ['sampler', ['tpe', 'random', 'cmaes', 'nsga2']], ['pruner', ['median', 'hyperband', 'none']]] as [keyof typeof cfg, string[]][]).map(([f, opts]) => (
            <select key={f} value={String(cfg[f])} onChange={(e) => setCfg({ ...cfg, [f]: e.target.value })} className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200">{opts.map((o) => <option key={o}>{o}</option>)}</select>
          ))}
          <input type="number" min={1} max={500} value={cfg.n_trials} onChange={(e) => setCfg({ ...cfg, n_trials: Number(e.target.value) })} className="w-20 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200" />
          <label className="text-slate-300"><input type="checkbox" checked={cfg.multiobjective} onChange={(e) => setCfg({ ...cfg, multiobjective: e.target.checked })} /> multiobjetivo (AUC↑, Brier↓; NSGA-II)</label>
        </div>
        <button onClick={() => guard(api('/v3/optuna/studies', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg) }).then((r) => setMsg(`${r.advertencia} Trabajo ${r.id}.`)))} className="px-3 py-1.5 rounded bg-sky-600 text-xs text-white">Ejecutar</button>
        {msg && <p className="text-[11px] text-amber-300">{msg}</p>}
        <div className="max-h-72 overflow-y-auto"><Table cols={['Estudio', 'Trials', 'Mejor', '']} rows={studies.map((s) => [s.nombre, s.n_trials, s.mejor_valor, ''])} /></div>
        <select onChange={(e) => guard(api(`/v3/optuna/studies/${e.target.value}`).then(setSel))} className="bg-slate-950 border border-slate-700 rounded text-xs px-2 py-1 text-slate-200"><option>Elegir estudio…</option>{studies.map((s) => <option key={s.nombre}>{s.nombre}</option>)}</select>
      </Card>
      {sel && (
        <Card title={`Estudio ${sel.nombre}`}>
          <p className="text-xs text-slate-400">Completos {sel.conteos.COMPLETE} · podados {sel.conteos.PRUNED} · fallidos {sel.conteos.FAIL} · en curso {sel.conteos.RUNNING}</p>
          {sel.historial && <LineChart series={[{ name: 'Valor', x: sel.historial.numero, y: sel.historial.valor }, { name: 'Mejor acumulado', x: sel.historial.numero, y: sel.historial.mejor_acumulado }]} xLabel="Trial" yLabel="ROC-AUC interno" />}
          {sel.importancia && Object.keys(sel.importancia).length > 0 && <Bars data={sel.importancia} max={1} />}
          {sel.contorno && <LineChart series={[{ name: `${sel.contorno.x} vs ${sel.contorno.y}`, x: sel.contorno.puntos.map((p: Json) => p.x), y: sel.contorno.puntos.map((p: Json) => p.y) }]} xLabel={sel.contorno.x} yLabel={sel.contorno.y} />}
          {sel.edf && <LineChart series={[{ name: 'EDF', x: sel.edf.valor, y: sel.edf.fraccion }]} xLabel="Valor objetivo" yLabel="Fracción de trials" yMin={0} yMax={1} />}
          {sel.coordenadas_paralelas && <Table cols={Object.keys(sel.coordenadas_paralelas[0] ?? {})} rows={sel.coordenadas_paralelas.slice(-10).map((r: Json) => Object.values(r))} />}
          {sel.pareto && <Table cols={['Trial', 'AUC', 'Brier']} rows={sel.pareto.map((p: Json) => [p.numero, ...p.valores])} />}
          <Interp i={sel.interpretabilidad} />
          <button onClick={() => guard(api(`/v3/optuna/studies/${sel.nombre}/export`, { method: 'POST' }).then((r) => setMsg(`Exportado a ${r.yaml}`)))} className="px-3 py-1.5 rounded border border-slate-600 text-xs text-slate-300">Exportar mejores parámetros a YAML</button>
        </Card>
      )}
    </div>
  );
}
