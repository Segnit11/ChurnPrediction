import React from 'react';
import { motion } from 'framer-motion';
import { fadeUp, inView } from '../../lib/motion';

// Scroll-triggered reveal wrapper. Defaults to a spring fade-up.
export default function Reveal({
  children,
  variants = fadeUp,
  className = '',
  as = 'div',
  ...rest
}) {
  const MotionTag = motion[as] || motion.div;
  return (
    <MotionTag
      className={className}
      variants={variants}
      initial="hidden"
      whileInView="show"
      viewport={inView}
      {...rest}
    >
      {children}
    </MotionTag>
  );
}
