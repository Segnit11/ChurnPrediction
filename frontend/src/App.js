import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import axios from 'axios';
import GaugeChart from 'react-gauge-chart';
import { Bar } from 'react-chartjs-2';
import parse from 'html-react-parser';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import './App.css';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const { register, handleSubmit, setValue } = useForm();
  const [customers, setCustomers] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(() => 
    window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  useEffect(() => {
    axios.get('http://localhost:5001/customers')
      .then(response => setCustomers(response.data.slice(0, 100)))
      .catch(error => console.error('Error fetching customers:', error));
  }, []);

  useEffect(() => {
    document.body.classList.toggle('dark', isDarkMode);
  }, [isDarkMode]);

  const toggleDarkMode = () => setIsDarkMode(prev => !prev);

  const onSubmit = (data) => {
    const { customer, ...formData } = data;
    const predictionData = {
      CustomerId: parseInt(formData.CustomerId),
      Surname: formData.Surname,
      CreditScore: parseInt(formData.CreditScore),
      Geography: formData.Geography,
      Gender: formData.Gender,
      Age: parseInt(formData.Age),
      Tenure: parseInt(formData.Tenure),
      Balance: parseFloat(formData.Balance),
      NumOfProducts: parseInt(formData.NumOfProducts),
      HasCrCard: parseInt(formData.HasCrCard),
      IsActiveMember: parseInt(formData.IsActiveMember),
      EstimatedSalary: parseFloat(formData.EstimatedSalary)
    };
    axios.post('http://localhost:5001/predict', predictionData)
      .then(response => setPrediction(response.data))
      .catch(error => console.error('Error predicting:', error));
  };

  const handleCustomerChange = (e) => {
    const selected = customers.find(c => `${c.CustomerId} - ${c.Surname}` === e.target.value);
    if (selected) {
      setValue('CustomerId', selected.CustomerId, { shouldValidate: false });
      setValue('Surname', selected.Surname, { shouldValidate: false });
    }
  };

  const getBarChartData = () => {
    if (!prediction) return {};
    const labels = Object.keys(prediction.model_probabilities);
    const data = Object.values(prediction.model_probabilities).map(prob => prob * 100);
    return {
      labels,
      datasets: [
        {
          label: 'Churn Probability (%)',
          data,
          backgroundColor: isDarkMode ? '#34D399' : '#10B981',
          borderColor: isDarkMode ? '#10B981' : '#059669',
          borderWidth: 1,
        },
      ],
    };
  };

  const barChartOptions = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Model Breakdown',
        color: isDarkMode ? '#E5E7EB' : '#1F2937',
        font: { size: 18, weight: '500' },
      },
      tooltip: {
        callbacks: {
          label: (context) => `${context.parsed.x.toFixed(2)}%`,
        },
      },
    },
    scales: {
      x: {
        beginAtZero: true,
        max: 100,
        ticks: {
          color: isDarkMode ? '#D1D5DB' : '#6B7280',
          stepSize: 25,
          callback: (value) => `${value}%`,
        },
        grid: {
          color: isDarkMode ? '#4B5563' : '#E5E7EB',
        },
      },
      y: {
        ticks: {
          color: isDarkMode ? '#D1D5DB' : '#6B7280',
        },
        grid: {
          display: false,
        },
      },
    },
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Churn Prediction</h1>
        <button onClick={toggleDarkMode} className="dark-mode-toggle">
          {isDarkMode ? '☀️' : '🌙'}
        </button>
      </header>

      <main className="app-main">
        <form onSubmit={handleSubmit(onSubmit)} className="prediction-form">
          <div className="form-section">
            <h2>Customer Details</h2>
            <div className="form-grid">
              <div className="form-group">
                <label>Select Customer</label>
                <select {...register('customer')} onChange={handleCustomerChange}>
                  <option value="">Select...</option>
                  {customers.map((c, index) => (
                    <option key={index} value={`${c.CustomerId} - ${c.Surname}`}>
                      {c.CustomerId} - {c.Surname}
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>Credit Score</label>
                <input type="number" {...register('CreditScore', { required: true })} />
              </div>
              <div className="form-group">
                <label>Balance</label>
                <input type="number" {...register('Balance', { required: true })} />
              </div>
              <div className="form-group">
                <label>Geography</label>
                <select {...register('Geography', { required: true })}>
                  <option value="">Select...</option>
                  <option value="France">France</option>
                  <option value="Germany">Germany</option>
                  <option value="Spain">Spain</option>
                </select>
              </div>
              <div className="form-group">
                <label>Number of Products</label>
                <input type="number" {...register('NumOfProducts', { required: true, min: 1 })} />
              </div>
              <div className="form-group">
                <label>Gender</label>
                <select {...register('Gender', { required: true })}>
                  <option value="">Select...</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
              <div className="form-group">
                <label>Has Credit Card</label>
                <select {...register('HasCrCard', { required: true })}>
                  <option value="">Select...</option>
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>
              <div className="form-group">
                <label>Is Active Member</label>
                <select {...register('IsActiveMember', { required: true })}>
                  <option value="">Select...</option>
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>
              <div className="form-group">
                <label>Age</label>
                <input type="number" {...register('Age', { required: true, min: 18 })} />
              </div>
              <div className="form-group">
                <label>Estimated Salary</label>
                <input type="number" {...register('EstimatedSalary', { required: true })} />
              </div>
              <div className="form-group">
                <label>Tenure (Years)</label>
                <input type="number" {...register('Tenure', { required: true, min: 0 })} />
              </div>
            </div>
            <input type="hidden" {...register('CustomerId')} />
            <input type="hidden" {...register('Surname')} />
            <button type="submit" className="submit-btn">Predict</button>
          </div>
        </form>

        {prediction && (
          <section className="results-section">
            <h2>Prediction Results</h2>
            <div className="gauge-container">
              <h3>Churn Probability</h3>
              <GaugeChart
                id="churn-gauge"
                nrOfLevels={20}
                percent={prediction.probability}
                textColor={isDarkMode ? "#E5E7EB" : "#1F2937"}
                needleColor={isDarkMode ? "#9CA3AF" : "#6B7280"}
                colors={isDarkMode ? ['#F87171', '#FBBF24', '#34D399'] : ['#EF4444', '#F59E0B', '#10B981']}
                formatTextValue={value => `${(value).toFixed(2)}%`}
              />
            </div>
            <div className="model-probabilities">
              <div className="bar-chart-container">
                <Bar data={getBarChartData()} options={barChartOptions} />
              </div>
            </div>
            <div className="explanation">
              <h3>Explanation</h3>
              <p>{prediction.explanation}</p>
            </div>
            <div className="email-preview">
              <h3>Personalized Email</h3>
              <div className="email-content">{parse(prediction.email)}</div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;