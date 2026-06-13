import React, { useEffect, useRef, useState } from 'react';
import { animate, useInView } from 'framer-motion';

// Counts up to `value` when scrolled into view. Respects reduced motion by
// jumping straight to the final value if the animation is neutralised.
export default function AnimatedNumber({
  value = 0,
  decimals = 0,
  prefix = '',
  suffix = '',
  duration = 1.2,
  className = '',
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, amount: 0.5 });
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (!inView) return undefined;
    const controls = animate(0, value, {
      duration,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => setDisplay(v),
    });
    return () => controls.stop();
  }, [inView, value, duration]);

  const formatted = display.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

  return (
    <span ref={ref} className={className}>
      {prefix}
      {formatted}
      {suffix}
    </span>
  );
}
