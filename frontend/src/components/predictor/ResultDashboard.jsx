import React from 'react';
import { motion } from 'framer-motion';
import GaugeChart from 'react-gauge-chart';
import { Bar } from 'react-chartjs-2';
import parse from 'html-react-parser';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
} from 'chart.js';
import {
  AlertTriangle,
  CheckCircle2,
  Mail,
  Copy,
  Check,
  Gauge as GaugeIcon,
  ListChecks,
} from 'lucide-react';
import { fadeUp, stagger } from '../../lib/motion';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

const TONE = {
  danger: { text: 'text-danger-bright', ring: 'border-danger/40', glow: 'bg-danger/10', chip: 'bg-danger/15 text-danger-bright' },
  warn: { text: 'text-warn', ring: 'border-warn/40', glow: 'bg-warn/10', chip: 'bg-warn/15 text-warn' },
  brand: { text: 'text-brand-bright', ring: 'border-brand/40', glow: 'bg-brand/10', chip: 'bg-brand/15 text-brand-bright' },
  safe: { text: 'text-safe-bright', ring: 'border-safe/40', glow: 'bg-safe/10', chip: 'bg-safe/15 text-safe-bright' },
};

function ModelBreakdown({ probs }) {
  const labels = Object.keys(probs);
  const data = {
    labels: labels.map((l) => l.replace('Classifier', '')),
    datasets: [
      {
        data: labels.map((l) => probs[l] * 100),
        backgroundColor: ['#6366f1', '#a855f7', '#22d3ee', '#818cf8'],
        borderRadius: 6,
        maxBarThickness: 26,
      },
    ],
  };
  const options = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (c) => `${c.parsed.x.toFixed(1)}%` } },
    },
    scales: {
      x: { beginAtZero: true, max: 100, ticks: { color: '#aab4c8', callback: (v) => `${v}%` }, grid: { color: 'rgba(36,48,73,0.45)' } },
      y: { ticks: { color: '#aab4c8' }, grid: { display: false } },
    },
  };
  return (
    <div className="h-44">
      <Bar data={data} options={options} />
    </div>
  );
}

export default function ResultDashboard({ result }) {
  const [copied, setCopied] = React.useState(false);
  const tone = TONE[result.risk?.color] || TONE.brand;
  const pct = (result.probability * 100).toFixed(1);
  const atRisk = result.prediction === 1;

  const copyEmail = async () => {
    try {
      // Strip basic HTML tags for a clean clipboard paste.
      const text = (result.email || '').replace(/<[^>]+>/g, '');
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      /* clipboard unavailable */
    }
  };

  return (
    <motion.div
      variants={stagger(0.08)}
      initial="hidden"
      animate="show"
      className="grid gap-6 lg:grid-cols-5"
    >
      {/* Verdict + gauge */}
      <motion.div variants={fadeUp} className={`glass relative overflow-hidden rounded-2xl border ${tone.ring} p-6 lg:col-span-2`}>
        <div className={`absolute -right-12 -top-12 h-40 w-40 rounded-full ${tone.glow} blur-3xl`} />
        <div className="relative">
          <div className="flex items-center justify-between">
            <span className="inline-flex items-center gap-2 text-sm text-ink-muted">
              <GaugeIcon className="h-4 w-4" /> Churn probability
            </span>
            <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${tone.chip}`}>
              {atRisk ? <AlertTriangle className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
              {result.risk?.tier}
            </span>
          </div>

          <div className="mx-auto mt-2 max-w-[280px]">
            <GaugeChart
              id="churn-gauge"
              nrOfLevels={24}
              percent={result.probability}
              textColor="#f8fafc"
              needleColor="#6b7793"
              needleBaseColor="#6b7793"
              colors={['#10b981', '#f59e0b', '#ef4444']}
              arcPadding={0.02}
              formatTextValue={() => `${pct}%`}
            />
          </div>

          <p className={`mt-1 text-center text-sm font-medium ${tone.text}`}>
            {result.risk?.headline}
          </p>
          <div className="mt-4 flex items-center justify-center gap-2 text-xs text-ink-faint">
            <span className="h-1.5 w-1.5 rounded-full bg-brand-bright" />
            Ensemble confidence {result.confidence}%
          </div>
        </div>
      </motion.div>

      {/* Recommended actions */}
      <motion.div variants={fadeUp} className="glass rounded-2xl p-6 lg:col-span-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-ink">
          <ListChecks className="h-4 w-4 text-brand-bright" />
          Recommended playbook
        </div>
        <ul className="mt-4 space-y-3">
          {(result.risk?.actions || []).map((a, i) => (
            <motion.li
              key={i}
              variants={fadeUp}
              className="flex items-start gap-3 rounded-xl border border-border-soft bg-surface/40 p-3 text-sm text-ink-muted"
            >
              <span className={`mt-0.5 flex h-5 w-5 flex-none items-center justify-center rounded-full text-[11px] font-bold ${tone.chip}`}>
                {i + 1}
              </span>
              <span>{a}</span>
            </motion.li>
          ))}
        </ul>
        <div className="mt-5">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-faint">
            Model breakdown
          </div>
          <ModelBreakdown probs={result.model_probabilities} />
        </div>
      </motion.div>

      {/* Explanation */}
      <motion.div variants={fadeUp} className="glass rounded-2xl p-6 lg:col-span-2">
        <div className="text-sm font-semibold text-ink">Why this score</div>
        <p className="mt-3 text-sm leading-relaxed text-ink-muted">{result.explanation}</p>
      </motion.div>

      {/* Email */}
      <motion.div variants={fadeUp} className="glass rounded-2xl p-6 lg:col-span-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm font-semibold text-ink">
            <Mail className="h-4 w-4 text-brand-bright" />
            Personalized retention email
          </div>
          <button
            onClick={copyEmail}
            className="inline-flex items-center gap-1.5 rounded-lg border border-border-soft bg-surface/60 px-3 py-1.5 text-xs font-medium text-ink-muted transition-colors hover:text-ink"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-safe-bright" /> : <Copy className="h-3.5 w-3.5" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
        <div className="mt-4 max-h-80 overflow-y-auto whitespace-pre-line rounded-xl border border-border-soft bg-canvas/60 p-4 text-sm leading-relaxed text-ink-muted">
          {result.email ? parse(result.email) : 'No email generated.'}
        </div>
      </motion.div>
    </motion.div>
  );
}
