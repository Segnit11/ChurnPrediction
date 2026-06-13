import React from 'react';
import { motion } from 'framer-motion';
import { fadeUp, stagger, inView } from '../../lib/motion';

// Consistent eyebrow + title + subtitle block used at the top of each section.
export default function SectionHeading({ eyebrow, title, subtitle, center = true }) {
  return (
    <motion.div
      variants={stagger(0.1)}
      initial="hidden"
      whileInView="show"
      viewport={inView}
      className={`max-w-2xl ${center ? 'mx-auto text-center' : ''}`}
    >
      {eyebrow && (
        <motion.span
          variants={fadeUp}
          className="inline-flex items-center gap-2 rounded-full border border-border-soft bg-surface/60 px-3 py-1 text-xs font-medium uppercase tracking-wider text-brand-bright"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-brand-bright" />
          {eyebrow}
        </motion.span>
      )}
      <motion.h2
        variants={fadeUp}
        className="mt-4 text-3xl font-bold tracking-tight text-ink sm:text-4xl"
      >
        {title}
      </motion.h2>
      {subtitle && (
        <motion.p variants={fadeUp} className="mt-4 text-base leading-relaxed text-ink-muted">
          {subtitle}
        </motion.p>
      )}
    </motion.div>
  );
}
