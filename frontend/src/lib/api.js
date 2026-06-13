import axios from 'axios';

// Configurable API base — defaults to the local Flask server.
export const API_BASE =
  process.env.REACT_APP_API_URL || 'http://localhost:5001';

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
});

export const getCustomers = (q = '', limit = 100) =>
  api.get('/customers', { params: { q, limit } }).then((r) => r.data);

export const getAnalytics = () => api.get('/analytics').then((r) => r.data);

export const getFeatureImportance = () =>
  api.get('/feature-importance').then((r) => r.data);

export const predict = (payload) =>
  api.post('/predict', payload).then((r) => r.data);
