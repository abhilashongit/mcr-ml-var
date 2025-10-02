import Head from 'next/head'
import { useState, useEffect } from 'react'

export default function Home() {
  const [theme, setTheme] = useState('light')
  const [copySuccess, setCopySuccess] = useState('')

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'light'
    setTheme(savedTheme)
    document.documentElement.setAttribute('data-theme', savedTheme)
  }, [])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
  }

  const copyToClipboard = async (text, buttonId) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopySuccess(buttonId)
      setTimeout(() => setCopySuccess(''), 2000)
    } catch (err) {
      console.error('Failed to copy text: ', err)
    }
  }

  const deploymentCommands = {
    clone: `git clone https://github.com/abhilashongit/mcr-ml-var.git
cd mcr-ml-var`,
    install: `npm install
# or
pip install -r requirements.txt`,
    setup: `cp .env.example .env
# Edit .env with your configuration`,
    deploy: `npm install -g vercel
vercel login
vercel --prod`
  }

  return (
    <div>
      <Head>
        <title>MCR ML-VAR Documentation</title>
        <meta name="description" content="Professional deployment and technical guidance for VAR model implementation" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      {/* Header */}
      <header className="header">
        <div className="container">
          <div className="header-content">
            <div className="logo">
              <h1>MCR ML-VAR</h1>
              <span>Documentation</span>
            </div>
            <nav className="nav">
              <a href="#home" className="nav-link active">Home</a>
              <a href="#deployment" className="nav-link">Deployment</a>
              <a href="#technical" className="nav-link">Technical Details</a>
              <a href="#model" className="nav-link">Model Info</a>
            </nav>
            <div className="header-actions">
              <a href="https://github.com/abhilashongit/mcr-ml-var" target="_blank" className="btn btn--outline btn--sm">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.91 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                GitHub
              </a>
              <button className="theme-toggle" onClick={toggleTheme}>
                {theme === 'light' ? '🌙' : '☀️'}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="hero" id="home">
        <div className="container">
          <div className="hero-content">
            <h1>Machine Learning & Econometric Model Documentation</h1>
            <p>Professional deployment and technical guidance for VAR model implementation with comprehensive scaling strategies</p>
            <div className="hero-actions">
              <a href="#deployment" className="btn btn--primary">Get Started with Deployment</a>
              <a href="https://github.com/abhilashongit/mcr-ml-var" target="_blank" className="btn btn--outline">View Repository</a>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="main">
        <div className="container">
          
          {/* Quick Deployment Guide */}
          <section className="section" id="deployment">
            <h2 className="section-title">Quick Deployment Guide</h2>
            
            <div className="cards">
              <div className="card">
                <h3>1. Clone Repository</h3>
                <p>Download the repository to your local machine</p>
                <div className="code-block">
                  <button 
                    className="copy-btn"
                    onClick={() => copyToClipboard(deploymentCommands.clone, 'clone')}
                  >
                    {copySuccess === 'clone' ? 'Copied!' : 'Copy'}
                  </button>
                  <pre>{deploymentCommands.clone}</pre>
                </div>
              </div>

              <div className="card">
                <h3>2. Install Dependencies</h3>
                <p>Install required packages and dependencies</p>
                <div className="code-block">
                  <button 
                    className="copy-btn"
                    onClick={() => copyToClipboard(deploymentCommands.install, 'install')}
                  >
                    {copySuccess === 'install' ? 'Copied!' : 'Copy'}
                  </button>
                  <pre>{deploymentCommands.install}</pre>
                </div>
              </div>

              <div className="card">
                <h3>3. Environment Setup</h3>
                <p>Configure environment variables for your deployment</p>
                <div className="code-block">
                  <button 
                    className="copy-btn"
                    onClick={() => copyToClipboard(deploymentCommands.setup, 'setup')}
                  >
                    {copySuccess === 'setup' ? 'Copied!' : 'Copy'}
                  </button>
                  <pre>{deploymentCommands.setup}</pre>
                </div>
              </div>

              <div className="card">
                <h3>4. Vercel Deployment</h3>
                <p>Deploy directly to Vercel for production use</p>
                <div className="code-block">
                  <button 
                    className="copy-btn"
                    onClick={() => copyToClipboard(deploymentCommands.deploy, 'deploy')}
                  >
                    {copySuccess === 'deploy' ? 'Copied!' : 'Copy'}
                  </button>
                  <pre>{deploymentCommands.deploy}</pre>
                </div>
              </div>
            </div>
          </section>

          {/* Technical Specifications */}
          <section className="section" id="technical">
            <h2 className="section-title">Technical Infrastructure</h2>
            
            <div className="cards">
              <div className="card">
                <h3>💻 Minimum Requirements</h3>
                <ul>
                  <li><strong>CPU:</strong> 4 cores</li>
                  <li><strong>RAM:</strong> 8GB</li>
                  <li><strong>Storage:</strong> 50GB SSD</li>
                  <li><strong>GPU:</strong> Not required</li>
                </ul>
              </div>

              <div className="card">
                <h3>🚀 Recommended Specs</h3>
                <ul>
                  <li><strong>CPU:</strong> 8 cores</li>
                  <li><strong>RAM:</strong> 16GB</li>
                  <li><strong>Storage:</strong> 100GB SSD</li>
                  <li><strong>GPU:</strong> NVIDIA Tesla T4</li>
                </ul>
              </div>

              <div className="card">
                <h3>⚙️ Runtime Requirements</h3>
                <ul>
                  <li><strong>Node.js:</strong> 18.x or higher</li>
                  <li><strong>Python:</strong> 3.8 or higher</li>
                  <li><strong>Frameworks:</strong> Next.js, React</li>
                  <li><strong>Styling:</strong> Tailwind CSS</li>
                </ul>
              </div>
            </div>
          </section>

          {/* VAR Model Information */}
          <section className="section" id="model">
            <h2 className="section-title">VAR Model Implementation</h2>
            
            <div className="cards">
              <div className="card">
                <h3>📊 What is VAR?</h3>
                <p>Vector Autoregression (VAR) is a multivariate forecasting algorithm used when two or more time series influence each other. It captures linear interdependencies among multiple time series.</p>
              </div>

              <div className="card">
                <h3>🎯 Applications</h3>
                <ul>
                  <li>Financial market analysis</li>
                  <li>Economic forecasting</li>
                  <li>Oil price volatility prediction</li>
                  <li>Portfolio risk management</li>
                  <li>Macroeconomic policy analysis</li>
                </ul>
              </div>

              <div className="card">
                <h3>✨ Key Features</h3>
                <ul>
                  <li>Multivariate time series modeling</li>
                  <li>Lag order selection via AIC/BIC</li>
                  <li>Impulse response functions</li>
                  <li>Forecast error variance decomposition</li>
                  <li>Granger causality testing</li>
                </ul>
              </div>
            </div>
          </section>

        </div>
      </main>
    </div>
  )
}
