import React, { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Loader2, AlertCircle, Wand2 } from 'lucide-react';
import SectionHeading from '../ui/SectionHeading';
import Reveal from '../ui/Reveal';
import ResultDashboard from '../predictor/ResultDashboard';
import { getCustomers, predict } from '../../lib/api';
import { fadeUp } from '../../lib/motion';

// One sensible sample so visitors can try it instantly without a backend lookup.
const SAMPLE = {
  CustomerId: 15634602,
  Surname: 'Hargrave',
  CreditScore: 619,
  Geography: 'France',
  Gender: 'Female',
  Age: 42,
  Tenure: 2,
  Balance: 0,
  NumOfProducts: 1,
  HasCrCard: 1,
  IsActiveMember: 1,
  EstimatedSalary: 101348,
};

const numberFields = [
  { name: 'CreditScore', label: 'Credit score', placeholder: '300–900', min: 300, max: 900 },
  { name: 'Balance', label: 'Balance', placeholder: 'e.g. 75000', min: 0, step: '0.01' },
  { name: 'NumOfProducts', label: 'Number of products', placeholder: '1–4', min: 1, max: 4 },
  { name: 'Age', label: 'Age', placeholder: '18+', min: 18, max: 120 },
  { name: 'EstimatedSalary', label: 'Estimated salary', placeholder: 'e.g. 90000', min: 0, step: '0.01' },
  { name: 'Tenure', label: 'Tenure (years)', placeholder: '0–50', min: 0, max: 50 },
];

const selectFields = [
  { name: 'Geography', label: 'Geography', options: ['France', 'Germany', 'Spain'] },
  { name: 'Gender', label: 'Gender', options: ['Male', 'Female'] },
  { name: 'HasCrCard', label: 'Has credit card', options: [['1', 'Yes'], ['0', 'No']] },
  { name: 'IsActiveMember', label: 'Active member', options: [['1', 'Yes'], ['0', 'No']] },
];

const inputClass =
  'w-full rounded-xl border border-border-soft bg-canvas/60 px-4 py-3 text-sm text-ink placeholder:text-ink-faint outline-none transition-colors focus:border-brand focus:ring-2 focus:ring-brand/30';

function Field({ label, error, children, required }) {
  return (
    <div>
      <label className="mb-1.5 block text-xs font-medium text-ink-muted">
        {label} {required && <span className="text-danger-bright">*</span>}
      </label>
      {children}
      {error && (
        <p className="mt-1 flex items-center gap-1 text-xs text-danger-bright">
          <AlertCircle className="h-3 w-3" /> {error}
        </p>
      )}
    </div>
  );
}

export default function Predictor() {
  const {
    register,
    handleSubmit,
    setValue,
    reset,
    formState: { errors },
  } = useForm({ defaultValues: SAMPLE });

  const [customers, setCustomers] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);

  useEffect(() => {
    getCustomers('', 100)
      .then((d) => setCustomers(d.results || []))
      .catch(() => setCustomers([]));
  }, []);

  const onCustomerChange = (e) => {
    const sel = customers.find((c) => `${c.CustomerId} - ${c.Surname}` === e.target.value);
    if (sel) {
      setValue('CustomerId', sel.CustomerId);
      setValue('Surname', sel.Surname);
    }
  };

  const fillSample = () => {
    reset(SAMPLE);
    setResult(null);
    setApiError(null);
  };

  const onSubmit = async (data) => {
    setLoading(true);
    setApiError(null);
    const payload = {
      CustomerId: parseInt(data.CustomerId, 10) || 0,
      Surname: data.Surname || 'Customer',
      CreditScore: parseInt(data.CreditScore, 10),
      Geography: data.Geography,
      Gender: data.Gender,
      Age: parseInt(data.Age, 10),
      Tenure: parseInt(data.Tenure, 10),
      Balance: parseFloat(data.Balance),
      NumOfProducts: parseInt(data.NumOfProducts, 10),
      HasCrCard: parseInt(data.HasCrCard, 10),
      IsActiveMember: parseInt(data.IsActiveMember, 10),
      EstimatedSalary: parseFloat(data.EstimatedSalary),
    };
    try {
      const res = await predict(payload);
      setResult(res);
      // Bring the freshly rendered results into view.
      requestAnimationFrame(() =>
        document.getElementById('prediction-result')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      );
    } catch (err) {
      const detail = err?.response?.data?.details?.join(' · ') || err?.response?.data?.error;
      setApiError(detail || 'Could not reach the prediction service. Is the backend running on the API URL?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="predictor" className="relative px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <SectionHeading
          eyebrow="Live predictor"
          title="Score a customer right now"
          subtitle="Pick a customer or enter a profile. ChurnGuard returns the risk, the reasoning, and a ready-to-send email."
        />

        <div className="mt-14 grid gap-6 lg:grid-cols-12">
          {/* Form */}
          <Reveal className="glass rounded-2xl p-6 lg:col-span-5">
            <div className="mb-5 flex items-center justify-between">
              <h3 className="text-base font-semibold text-ink">Customer profile</h3>
              <button
                type="button"
                onClick={fillSample}
                className="inline-flex items-center gap-1.5 rounded-lg border border-border-soft bg-surface/60 px-3 py-1.5 text-xs font-medium text-ink-muted transition-colors hover:text-ink"
              >
                <Wand2 className="h-3.5 w-3.5" /> Load sample
              </button>
            </div>

            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <Field label="Select existing customer">
                <select className={inputClass} onChange={onCustomerChange} defaultValue="">
                  <option value="">Type a profile or pick one…</option>
                  {customers.map((c) => (
                    <option key={c.CustomerId} value={`${c.CustomerId} - ${c.Surname}`}>
                      {c.CustomerId} — {c.Surname}
                    </option>
                  ))}
                </select>
              </Field>

              <div className="grid grid-cols-2 gap-4">
                {numberFields.map((f) => (
                  <Field
                    key={f.name}
                    label={f.label}
                    required
                    error={errors[f.name] && 'Required / out of range'}
                  >
                    <input
                      type="number"
                      step={f.step || '1'}
                      placeholder={f.placeholder}
                      className={inputClass}
                      {...register(f.name, {
                        required: true,
                        valueAsNumber: true,
                        min: f.min,
                        max: f.max,
                      })}
                    />
                  </Field>
                ))}
              </div>

              <div className="grid grid-cols-2 gap-4">
                {selectFields.map((f) => (
                  <Field key={f.name} label={f.label} required error={errors[f.name] && 'Required'}>
                    <select className={inputClass} {...register(f.name, { required: true })}>
                      {f.options.map((o) => {
                        const [val, text] = Array.isArray(o) ? o : [o, o];
                        return (
                          <option key={val} value={val}>
                            {text}
                          </option>
                        );
                      })}
                    </select>
                  </Field>
                ))}
              </div>

              <input type="hidden" {...register('CustomerId')} />
              <input type="hidden" {...register('Surname')} />

              <button
                type="submit"
                disabled={loading}
                className="btn-primary inline-flex w-full items-center justify-center gap-2 rounded-xl px-6 py-3.5 font-semibold disabled:cursor-not-allowed disabled:opacity-70"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" /> Scoring…
                  </>
                ) : (
                  <>
                    <Sparkles className="h-5 w-5" /> Predict churn risk
                  </>
                )}
              </button>

              {apiError && (
                <p className="flex items-start gap-2 rounded-xl border border-danger/40 bg-danger/10 p-3 text-sm text-danger-bright">
                  <AlertCircle className="mt-0.5 h-4 w-4 flex-none" /> {apiError}
                </p>
              )}
            </form>
          </Reveal>

          {/* Results / empty state */}
          <div id="prediction-result" className="lg:col-span-7">
            <AnimatePresence mode="wait">
              {result ? (
                <ResultDashboard key="result" result={result} />
              ) : (
                <motion.div
                  key="empty"
                  variants={fadeUp}
                  initial="hidden"
                  animate="show"
                  exit={{ opacity: 0 }}
                  className="glass flex h-full min-h-[420px] flex-col items-center justify-center rounded-2xl p-10 text-center"
                >
                  <span className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-brand/20 to-accent/20 text-brand-bright">
                    <Sparkles className="h-8 w-8" />
                  </span>
                  <h3 className="mt-5 text-lg font-semibold text-ink">Your prediction will appear here</h3>
                  <p className="mt-2 max-w-sm text-sm text-ink-muted">
                    Fill in the profile and hit <span className="text-ink">Predict churn risk</span> to
                    see the probability gauge, model breakdown, recommended playbook and a personalized email.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </section>
  );
}
