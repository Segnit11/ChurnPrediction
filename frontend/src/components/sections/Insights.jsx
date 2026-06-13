import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { Users, TrendingDown, HeartHandshake, Wallet } from 'lucide-react';
import SectionHeading from '../ui/SectionHeading';
import AnimatedNumber from '../ui/AnimatedNumber';
import Reveal from '../ui/Reveal';
import { getAnalytics } from '../../lib/api';
import { fadeUp, stagger, inView } from '../../lib/motion';

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Tooltip, Legend);

// Fallback so the marketing page always looks complete even with no backend.
const FALLBACK = {
  totalCustomers: 165034,
  churned: 34921,
  retained: 130113,
  overallChurnRate: 21.2,
  retentionRate: 78.8,
  avgBalanceChurned: 91108.54,
  avgBalanceRetained: 71153.32,
  byGeography: [
    { label: 'France', churnRate: 16.2 },
    { label: 'Germany', churnRate: 37.8 },
    { label: 'Spain', churnRate: 16.7 },
  ],
  byNumProducts: [
    { label: '1', churnRate: 27.7 },
    { label: '2', churnRate: 8.4 },
    { label: '3', churnRate: 82.7 },
    { label: '4', churnRate: 100 },
  ],
};

const GRID = 'rgba(36,48,73,0.45)';
const TICK = '#aab4c8';

function barOptions(title) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: true, text: title, color: '#f8fafc', font: { size: 14, weight: '600' } },
      tooltip: { callbacks: { label: (c) => `${c.parsed.y.toFixed(1)}% churn` } },
    },
    scales: {
      x: { ticks: { color: TICK }, grid: { display: false } },
      y: {
        beginAtZero: true,
        max: 100,
        ticks: { color: TICK, callback: (v) => `${v}%` },
        grid: { color: GRID },
      },
    },
  };
}

const KPIS = (a) => [
  { icon: Users, label: 'Customers analyzed', value: a.totalCustomers, decimals: 0 },
  { icon: TrendingDown, label: 'Overall churn rate', value: a.overallChurnRate, decimals: 1, suffix: '%', tone: 'text-danger-bright' },
  { icon: HeartHandshake, label: 'Retention rate', value: a.retentionRate, decimals: 1, suffix: '%', tone: 'text-safe-bright' },
  { icon: Wallet, label: 'Avg. balance at risk', value: Math.round(a.avgBalanceChurned), decimals: 0, prefix: '$' },
];

export default function Insights() {
  const [data, setData] = useState(FALLBACK);
  const [live, setLive] = useState(false);

  useEffect(() => {
    let active = true;
    getAnalytics()
      .then((d) => {
        if (active && d && d.totalCustomers) {
          setData(d);
          setLive(true);
        }
      })
      .catch(() => {/* keep fallback */});
    return () => {
      active = false;
    };
  }, []);

  const geoData = {
    labels: data.byGeography.map((g) => g.label),
    datasets: [
      {
        data: data.byGeography.map((g) => g.churnRate),
        backgroundColor: ['#6366f1', '#a855f7', '#22d3ee'],
        borderRadius: 8,
        maxBarThickness: 64,
      },
    ],
  };

  const prodData = {
    labels: data.byNumProducts.map((p) => `${p.label} product${p.label === '1' ? '' : 's'}`),
    datasets: [
      {
        data: data.byNumProducts.map((p) => p.churnRate),
        backgroundColor: data.byNumProducts.map((p) =>
          p.churnRate >= 50 ? '#ef4444' : p.churnRate >= 25 ? '#f59e0b' : '#10b981'
        ),
        borderRadius: 8,
        maxBarThickness: 64,
      },
    ],
  };

  const splitData = {
    labels: ['Retained', 'Churned'],
    datasets: [
      {
        data: [data.retained, data.churned],
        backgroundColor: ['#10b981', '#ef4444'],
        borderColor: '#0f1626',
        borderWidth: 4,
        hoverOffset: 6,
      },
    ],
  };

  return (
    <section id="insights" className="relative px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <SectionHeading
          eyebrow="Portfolio insights"
          title="Where your churn risk really lives"
          subtitle={
            live
              ? 'Computed live from your connected dataset.'
              : 'Sample analytics — connect the API to see your own portfolio.'
          }
        />

        {/* KPI cards */}
        <motion.div
          variants={stagger(0.08)}
          initial="hidden"
          whileInView="show"
          viewport={inView}
          className="mt-14 grid gap-5 sm:grid-cols-2 lg:grid-cols-4"
        >
          {KPIS(data).map((k) => (
            <motion.div
              key={k.label}
              variants={fadeUp}
              className="glass rounded-2xl p-6"
            >
              <span className="inline-flex h-11 w-11 items-center justify-center rounded-xl border border-border-soft bg-surface/60 text-brand-bright">
                <k.icon className="h-5 w-5" />
              </span>
              <div className={`mt-4 font-mono text-3xl font-bold ${k.tone || 'text-ink'}`}>
                <AnimatedNumber
                  value={k.value}
                  decimals={k.decimals}
                  prefix={k.prefix || ''}
                  suffix={k.suffix || ''}
                />
              </div>
              <div className="mt-1 text-sm text-ink-muted">{k.label}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Charts */}
        <div className="mt-8 grid gap-6 lg:grid-cols-3">
          <Reveal className="glass rounded-2xl p-6 lg:col-span-1">
            <div className="mb-4 text-sm font-semibold text-ink">Retained vs. churned</div>
            <div className="relative mx-auto h-64 max-w-xs">
              <Doughnut
                data={splitData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  cutout: '68%',
                  plugins: {
                    legend: { position: 'bottom', labels: { color: TICK, padding: 16 } },
                    tooltip: {
                      callbacks: { label: (c) => `${c.label}: ${c.parsed.toLocaleString()}` },
                    },
                  },
                }}
              />
            </div>
          </Reveal>

          <Reveal className="glass rounded-2xl p-6">
            <div className="h-72">
              <Bar data={geoData} options={barOptions('Churn rate by geography')} />
            </div>
          </Reveal>

          <Reveal className="glass rounded-2xl p-6">
            <div className="h-72">
              <Bar data={prodData} options={barOptions('Churn rate by products held')} />
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
