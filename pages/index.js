import Head from 'next/head'
import { useState, useEffect } from 'react'

export default function Home() {
  const [theme, setTheme] = useState('light')
  const [copySuccess, setCopySuccess] = useState('')

  useEffect(() => {
    // Set theme on load
    const savedTheme = localStorage.getItem('theme') || 'light'
    setTheme(savedTheme)
  }, [])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
  }

  const copyToClipboard = async (text, id) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopySuccess(id)
      setTimeout(() => setCopySuccess(''), 2000)
    } catch (err) {
      console.error('Failed to copy')
    }
  }

  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <Head>
        <title>MCR ML-VAR Documentation</title>
        <meta name="description" content="Machine Learning and Econometric Model Documentation" />
        <style jsx global>{`
          * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }
          
          body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            line-height: 1.6;
            color: #333;
            background: ${theme === 'dark' ? '#0a0a0a' : '#ffffff'};
          }
          
          .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
          }
          
          .header {
            background: ${theme === 'dark' ? '#1a1a1a' : '#ffffff'};
            border-bottom: 1px solid ${theme === 'dark' ? '#333' : '#e5e5e5'};
            padding: 20px 0;
            position: sticky;
            top: 0;
            z-index: 100;
          }
          
          .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
          }
          
          .logo h1 {
            color: #3b82f6;
            font-size: 28px;
            font-weight: 700;
          }
          
          .logo span {
            color: ${theme === 'dark' ? '#888' : '#666'};
            font-size: 14px;
          }
          
          .nav {
            display: flex;
            gap: 30px;
          }
          
          .nav a {
            color: ${theme === 'dark' ? '#ccc' : '#555'};
            text-decoration: none;
            font-weight: 500;
          }
          
          .nav a:hover {
            color: #3b82f6;
          }
          
          .btn {
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 500;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            border: none;
            font-size: 14px;
            transition: all 0.2s;
          }
          
          .btn-primary {
            background: linear-gradient(135deg, #3b82f6, #64ffda);
            color: white;
          }
          
          .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
          }
          
          .btn-outline {
            border: 1px solid ${theme === 'dark' ? '#333' : '#ddd'};
            color: ${theme === 'dark' ? '#ccc' : '#555'};
            background: transparent;
          }
          
          .btn-outline:hover {
            border-color: #3b82f6;
            color: #3b82f6;
          }
          
          .theme-toggle {
            backgroun
