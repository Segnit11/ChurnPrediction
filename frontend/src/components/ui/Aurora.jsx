import React from 'react';

// Fixed, animated aurora + grid backdrop that sits behind the whole app.
// Adapted from the Magic/21st crypto-hero inspiration, retuned to the
// ChurnGuard brand palette (indigo / violet / cyan).
export default function Aurora() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-canvas">
      {/* soft radial color washes */}
      <div
        className="absolute -inset-24 opacity-60"
        style={{
          background: `
            radial-gradient(ellipse 70% 50% at 50% -10%, rgba(99,102,241,0.30), transparent),
            radial-gradient(ellipse 55% 50% at 85% 30%, rgba(168,85,247,0.22), transparent),
            radial-gradient(ellipse 55% 50% at 15% 75%, rgba(34,211,238,0.18), transparent)
          `,
        }}
      />
      {/* faint grid */}
      <div
        className="absolute inset-0 opacity-[0.35]"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgba(36,48,73,0.5) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(36,48,73,0.5) 1px, transparent 1px)`,
          backgroundSize: '64px 64px',
          maskImage:
            'radial-gradient(ellipse 75% 60% at 50% 0%, #000 35%, transparent 100%)',
          WebkitMaskImage:
            'radial-gradient(ellipse 75% 60% at 50% 0%, #000 35%, transparent 100%)',
        }}
      />
      {/* drifting blobs */}
      <div className="absolute left-1/4 top-0 h-96 w-96 animate-pulse rounded-full bg-brand/20 blur-[120px]" />
      <div className="absolute right-1/4 top-1/3 h-96 w-96 animate-pulse rounded-full bg-accent/20 blur-[120px] [animation-delay:1s]" />
      <div className="absolute bottom-0 left-1/2 h-96 w-96 animate-pulse rounded-full bg-cyan/10 blur-[120px] [animation-delay:2s]" />
    </div>
  );
}
