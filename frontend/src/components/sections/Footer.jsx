import React from 'react';
import { ShieldCheck, Github, ArrowRight } from 'lucide-react';
import Reveal from '../ui/Reveal';

export default function Footer() {
  return (
    <footer className="relative px-6 pb-10 pt-12">
      <div className="mx-auto max-w-7xl">
        {/* CTA band */}
        <Reveal className="relative overflow-hidden rounded-3xl border border-border-soft p-10 text-center sm:p-14">
          <div className="absolute inset-0 -z-10 aurora opacity-70" />
          <h2 className="mx-auto max-w-2xl text-3xl font-bold tracking-tight text-ink sm:text-4xl">
            Stop losing customers you could have kept.
          </h2>
          <p className="mx-auto mt-4 max-w-xl text-ink-muted">
            Score your portfolio, understand the drivers, and act — all in one place.
          </p>
          <a
            href="#predictor"
            className="btn-primary mt-8 inline-flex items-center gap-2 rounded-xl px-8 py-4 font-semibold"
          >
            Try the predictor <ArrowRight className="h-5 w-5" />
          </a>
        </Reveal>

        {/* Bottom bar */}
        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-border-soft pt-8 sm:flex-row">
          <div className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand to-accent">
              <ShieldCheck className="h-4 w-4 text-white" />
            </span>
            <span className="font-semibold text-ink">
              Churn<span className="text-gradient">Guard</span>
            </span>
          </div>
          <p className="text-sm text-ink-faint">
            Built with a four-model ML ensemble · Flask + React
          </p>
          <a
            href="https://github.com/Segnit11/ChurnPrediction"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 text-sm text-ink-muted transition-colors hover:text-ink"
          >
            <Github className="h-4 w-4" /> Source
          </a>
        </div>
      </div>
    </footer>
  );
}
