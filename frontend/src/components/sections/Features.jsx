import React from 'react';
import { motion } from 'framer-motion';
import {
  Layers,
  Brain,
  Mail,
  Gauge,
  ShieldAlert,
  BarChart3,
} from 'lucide-react';
import SectionHeading from '../ui/SectionHeading';
import { fadeUp, stagger, inView } from '../../lib/motion';

const features = [
  {
    icon: Layers,
    title: 'Four-model ensemble',
    body: 'Random Forest, Gradient Boosting and XGBoost feed a stacking classifier — and you see every model’s vote, not just the verdict.',
  },
  {
    icon: Gauge,
    title: 'Probability + confidence',
    body: 'A calibrated churn probability alongside an ensemble-agreement confidence score, so you know how much to trust each call.',
  },
  {
    icon: ShieldAlert,
    title: 'Risk tiers & playbooks',
    body: 'Every customer is mapped to Loyal → Critical with a concrete, ready-to-run retention action plan attached.',
  },
  {
    icon: Brain,
    title: 'Plain-language reasoning',
    body: 'No black box. ChurnGuard explains the drivers behind each score in language a relationship manager can act on.',
  },
  {
    icon: Mail,
    title: 'One-click retention email',
    body: 'Auto-drafts a warm, personalized outreach email with tailored incentives — copy, tweak, send.',
  },
  {
    icon: BarChart3,
    title: 'Portfolio analytics',
    body: 'Live churn breakdowns by geography, age, product count and engagement reveal where risk concentrates.',
  },
];

export default function Features() {
  return (
    <section id="features" className="relative px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <SectionHeading
          eyebrow="Why ChurnGuard"
          title="Everything you need to fight churn"
          subtitle="From raw customer rows to a signed-off retention play — in seconds, not spreadsheets."
        />

        <motion.div
          variants={stagger(0.08)}
          initial="hidden"
          whileInView="show"
          viewport={inView}
          className="mt-14 grid gap-6 sm:grid-cols-2 lg:grid-cols-3"
        >
          {features.map((f) => (
            <motion.div
              key={f.title}
              variants={fadeUp}
              whileHover={{ y: -6 }}
              transition={{ type: 'spring', stiffness: 200, damping: 18 }}
              className="group relative overflow-hidden rounded-2xl border border-border-soft bg-surface/40 p-7"
            >
              <div className="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-brand/10 blur-2xl transition-opacity duration-300 group-hover:bg-accent/20" />
              <div className="relative">
                <span className="inline-flex h-12 w-12 items-center justify-center rounded-xl border border-border-soft bg-gradient-to-br from-brand/20 to-accent/20 text-brand-bright">
                  <f.icon className="h-6 w-6" />
                </span>
                <h3 className="mt-5 text-lg font-semibold text-ink">{f.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-muted">{f.body}</p>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
