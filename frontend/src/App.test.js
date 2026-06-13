import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the ChurnGuard predictor CTA', () => {
  render(<App />);
  // The hero/nav render primary CTAs referencing the prediction flow.
  const cta = screen.getAllByText(/predict/i);
  expect(cta.length).toBeGreaterThan(0);
});
