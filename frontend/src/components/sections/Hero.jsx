import React from 'react';
import { motion } from 'framer-motion';
import {
  ArrowRight,
  PlayCircle,
  TrendingDown,
  Activity,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { fadeUp, scaleIn, stagger } from '../../lib/motion';

const trust = [
  { icon: TrendingDown, label: 'Avg. churn cut', value: 'up to 27%' },
  { icon: Activity, label: 'Model accuracy', value: '4-model stack' },
];

// A miniature, animated dashboard card — the "floating preview" on the right.
function PreviewCard() {
  const assets = [
    { name: 'Loyal', value: '61%', tone: 'text-safe-bright' },
    { name: 'Watch', value: '18%', tone: 'text-brand-bright' },
    { name: 'Elevated', value: '14%', tone: 'text-warn' },
    { name: 'Critical', value: '7%', tone: 'text-danger-bright' },
  ];
  return (
    <motion.div variants={scaleIn} className="relative">
      <div className="absolute inset-0 rounded-3xl bg-gradient-to-r from-brand/20 to-accent/20 blur-3xl" />
      <motion.div
        animate={{ y: [0, -10, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
        className="glass relative rounded-3xl p-7 shadow-2xl"
      >
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-gradient-to-br from-brand to-accent" />
            <div>
              <div className="text-sm font-semibold text-ink">Retention Cockpit</div>
              <div className="text-xs text-ink-faint">Live · updated 2m ago</div>
            </div>
          </div>
          <ShieldCheck className="h-5 w-5 text-safe-bright" />
        </div>

        <div className="mb-1 text-sm text-ink-faint">Customers at risk this cycle</div>
        <div className="mb-1 font-mono text-4xl font-bold text-ink">1,284</div>
        <div className="mb-6 flex items-center gap-2 text-sm">
          <span className="font-medium text-safe-bright">-12.5%</span>
          <span className="text-ink-faint">vs last month</span>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {assets.map((a) => (
            <div key={a.name} className="rounded-xl border border-border-soft bg-surface/50 p-4">
              <div className="mb-1 text-xs text-ink-faint">{a.name}</div>
              <div className={`font-mono text-xl font-semibold ${a.tone}`}>{a.value}</div>
            </div>
          ))}
        </div>

        <div className="mt-5 flex items-center justify-between border-t border-border-soft pt-4 text-sm">
          <span className="text-ink-faint">Revenue protected</span>
          <span className="font-mono font-semibold text-ink">$2.4M</span>
        </div>
      </motion.div>
    </motion.div>
  );
}

export default function Hero() {
  return (
    <section id="top" className="relative overflow-hidden px-6 pt-36 pb-24">
      <div className="mx-auto grid max-w-7xl items-center gap-14 lg:grid-cols-2">
        <motion.div variants={stagger(0.12)} initial="hidden" animate="show" className="space-y-7">
          <motion.span
            variants={fadeUp}
            className="inline-flex items-center gap-2 rounded-full border border-border-soft bg-surface/60 px-4 py-1.5 text-sm text-ink-muted"
          >
            <Sparkles className="h-4 w-4 text-brand-bright" />
            AI-powered retention intelligence
          </motion.span>

          <motion.h1
            variants={fadeUp}
            className="text-4xl font-extrabold leading-[1.08] tracking-tight sm:text-5xl lg:text-6xl"
          >
            <span className="text-ink">Know who&apos;s about to</span>
            <br />
            <span className="text-gradient">leave — before they do.</span>
          </motion.h1>

          <motion.p variants={fadeUp} className="max-w-xl text-lg leading-relaxed text-ink-muted">
            ChurnGuard scores every customer with a four-model ensemble, explains
            the <span className="text-ink">why</span> in plain language, and drafts
            a personalized retention email in one click. Turn raw banking data into
            action that actually keeps customers.
          </motion.p>

          <motion.div variants={fadeUp} className="flex flex-col gap-4 sm:flex-row">
            <a
              href="#predictor"
              className="btn-primary group inline-flex items-center justify-center gap-2 rounded-xl px-8 py-4 font-semibold"
            >
              Predict a customer
              <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
            </a>
            <a
              href="#insights"
              className="inline-flex items-center justify-center gap-2 rounded-xl border border-border-soft bg-surface/40 px-8 py-4 font-semibold text-ink transition-colors hover:bg-surface"
            >
              <PlayCircle className="h-5 w-5" />
              See live insights
            </a>
          </motion.div>

          <motion.div variants={fadeUp} className="grid max-w-md grid-cols-2 gap-3 pt-2">
            {trust.map((t) => (
              <div
                key={t.label}
                className="flex items-center gap-3 rounded-xl border border-border-soft bg-surface/40 px-4 py-3"
              >
                <t.icon className="h-5 w-5 text-brand-bright" />
                <div>
                  <div className="text-xs text-ink-faint">{t.label}</div>
                  <div className="text-sm font-semibold text-ink">{t.value}</div>
                </div>
              </div>
            ))}
          </motion.div>
        </motion.div>

        <motion.div variants={stagger(0.1, 0.2)} initial="hidden" animate="show">
          <PreviewCard />
        </motion.div>
      </div>
    </section>
  );
}
