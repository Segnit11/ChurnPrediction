import React from 'react';
import { motion } from 'framer-motion';
import { ClipboardList, Cpu, Send } from 'lucide-react';
import SectionHeading from '../ui/SectionHeading';
import { fadeUp, stagger, inView } from '../../lib/motion';

const steps = [
  {
    icon: ClipboardList,
    step: '01',
    title: 'Bring the customer',
    body: 'Pick an existing customer or enter their profile — credit score, balance, products, tenure and engagement.',
  },
  {
    icon: Cpu,
    step: '02',
    title: 'Score the risk',
    body: 'The four-model ensemble returns a churn probability, confidence, risk tier and the drivers behind it.',
  },
  {
    icon: Send,
    step: '03',
    title: 'Act to retain',
    body: 'Follow the recommended playbook and send the auto-drafted, personalized retention email.',
  },
];

export default function HowItWorks() {
  return (
    <section id="how" className="relative px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <SectionHeading
          eyebrow="How it works"
          title="From data to retention in three steps"
        />

        <motion.div
          variants={stagger(0.12)}
          initial="hidden"
          whileInView="show"
          viewport={inView}
          className="relative mt-14 grid gap-6 md:grid-cols-3"
        >
          {/* connecting line on desktop */}
          <div className="absolute left-0 right-0 top-9 hidden h-px bg-gradient-to-r from-transparent via-border to-transparent md:block" />
          {steps.map((s) => (
            <motion.div
              key={s.step}
              variants={fadeUp}
              className="relative rounded-2xl border border-border-soft bg-surface/40 p-7"
            >
              <div className="flex items-center justify-between">
                <span className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-brand to-accent text-white shadow-lg shadow-brand/30">
                  <s.icon className="h-6 w-6" />
                </span>
                <span className="font-mono text-3xl font-bold text-border">{s.step}</span>
              </div>
              <h3 className="mt-5 text-lg font-semibold text-ink">{s.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-muted">{s.body}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
