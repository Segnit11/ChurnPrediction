import React from 'react';
import Aurora from './components/ui/Aurora';
import Navbar from './components/Navbar';
import Hero from './components/sections/Hero';
import Features from './components/sections/Features';
import Insights from './components/sections/Insights';
import HowItWorks from './components/sections/HowItWorks';
import Predictor from './components/sections/Predictor';
import Footer from './components/sections/Footer';

function App() {
  return (
    <div className="relative min-h-dvh font-sans text-ink">
      <Aurora />
      <Navbar />
      <main>
        <Hero />
        <Features />
        <Insights />
        <HowItWorks />
        <Predictor />
      </main>
      <Footer />
    </div>
  );
}

export default App;
