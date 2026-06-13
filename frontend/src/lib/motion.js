// Shared framer-motion variants. Spring-based, reduced-motion friendly
// (the global CSS rule neutralises durations when prefers-reduced-motion is set).

export const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: {
    opacity: 1,
    y: 0,
    transition: { type: 'spring', stiffness: 120, damping: 18 },
  },
};

export const fadeIn = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: 0.5 } },
};

export const scaleIn = {
  hidden: { opacity: 0, scale: 0.96, y: 30 },
  show: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { type: 'spring', stiffness: 90, damping: 16 },
  },
};

// Parent container that staggers its children's entrance.
export const stagger = (gap = 0.08, delay = 0) => ({
  hidden: {},
  show: {
    transition: { staggerChildren: gap, delayChildren: delay },
  },
});

// Viewport config reused for scroll-triggered reveals.
export const inView = { once: true, amount: 0.25 };
