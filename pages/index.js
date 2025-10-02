import Head from 'next/head'

export default function Home() {
  return (
    <div>
      <Head>
        <title>MCR ML-VAR Documentation</title>
        <meta name="description" content="Machine Learning and Econometric Model Documentation" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main style={{ padding: '2rem', fontFamily: 'Arial, sans-serif' }}>
        <h1>MCR ML-VAR Documentation</h1>
        <p>Machine Learning and Econometric Vector Autoregression Model Repository</p>
        
        <h2>Quick Start</h2>
        <pre style={{ background: '#f4f4f4', padding: '1rem', borderRadius: '4px' }}>
{`# Clone the repository
git clone https://github.com/abhilashongit/mcr-ml-var.git
cd mcr-ml-var

# Install dependencies
npm install

# Run the model
python main.py`}
        </pre>

        <h2>Technical Specifications</h2>
        <ul>
          <li>CPU: 4+ cores recommended</li>
          <li>RAM: 8GB minimum, 16GB recommended</li>
          <li>Storage: 50GB+ SSD</li>
          <li>Python 3.8+ required</li>
        </ul>

        <h2>Repository</h2>
        <a href="https://github.com/abhilashongit/mcr-ml-var" target="_blank">
          View on GitHub
        </a>
      </main>
    </div>
  )
}
